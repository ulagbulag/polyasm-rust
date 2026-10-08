//! MIR rvalue, scalar-expression, and register assignment lowering.

use rustc_middle::mir::{
    AggregateKind, BinOp, CastKind, ConstOperand, Operand, Place, Rvalue, UnOp,
};
use rustc_middle::ty;
use rustc_span::Span;

use super::super::model::{Instruction, Opcode, Register, ValueType};
use super::{FunctionEmitter, LocalShape, PointerProvenance, address_token, encoded_type};

impl<'a, 'tcx> FunctionEmitter<'a, 'tcx> {
    pub(super) fn emit_assignment(
        &mut self,
        destination: Place<'tcx>,
        value: &Rvalue<'tcx>,
        span: Span,
    ) {
        match value {
            Rvalue::Use(operand, _) => self.emit_use(destination, operand, span),
            Rvalue::Ref(_, _, target)
            | Rvalue::RawPtr(_, target)
            | Rvalue::Reborrow(_, _, target) => {
                self.emit_address_of(destination, *target, span);
            }
            Rvalue::CopyForDeref(target) => {
                self.emit_use(destination, &Operand::Copy(*target), span);
            }
            Rvalue::BinaryOp(operation, operands) => {
                if matches!(
                    operation,
                    BinOp::AddWithOverflow | BinOp::MulWithOverflow | BinOp::SubWithOverflow
                ) {
                    self.emit_overflow_binary(
                        destination,
                        *operation,
                        &operands.0,
                        &operands.1,
                        span,
                    );
                } else {
                    self.emit_binary(destination, *operation, &operands.0, &operands.1, span);
                }
            }
            Rvalue::UnaryOp(operation, operand) => {
                let Some(ty) = self.operand_type(operand, span) else { return };
                let Some(destination) = self.place_register(destination, span) else { return };
                let Some(source) = self.operand_register(operand, span) else { return };
                let opcode = match operation {
                    UnOp::Neg => Opcode::Neg,
                    UnOp::Not => Opcode::Not,
                    UnOp::PtrMetadata => {
                        self.error(span, "PolyASM does not support wide-pointer metadata");
                        return;
                    }
                };
                if self.require_register_type(destination, ty, span) {
                    self.code.push(Instruction::unary(opcode, ty, destination, source));
                }
            }
            Rvalue::Cast(kind, operand, _) => self.emit_cast(destination, *kind, operand, span),
            Rvalue::Aggregate(kind, operands) => {
                if matches!(&**kind, AggregateKind::Closure(..))
                    && matches!(self.place_shape(destination, span), Some(LocalShape::Opaque))
                {
                    return;
                }
                if !matches!(&**kind, AggregateKind::Tuple | AggregateKind::Array(_)) {
                    self.error(span, "PolyASM supports only tuple and byte-array aggregates");
                    return;
                }
                let Some(fields) = self.place_fields(destination, span).map(<[Register]>::to_vec)
                else {
                    return;
                };
                if fields.len() != operands.len() {
                    self.error(span, "PolyASM aggregate assignment has the wrong arity");
                    return;
                }
                for (field, operand) in fields.into_iter().zip(operands) {
                    self.emit_move(field, operand, span);
                }
            }
            Rvalue::Repeat(..) | Rvalue::ThreadLocalRef(..) | Rvalue::WrapUnsafeBinder(..) => {
                self.error(span, "rvalue is not supported by native PolyASM codegen");
            }
            Rvalue::Discriminant(..) => {
                self.error(span, "PolyASM does not support enum discriminants");
            }
        }
    }

    pub(super) fn emit_use(
        &mut self,
        destination: Place<'tcx>,
        operand: &Operand<'tcx>,
        span: Span,
    ) {
        if matches!(self.place_shape(destination, span), Some(LocalShape::Pointer { .. })) {
            self.emit_pointer_use(destination, operand, span);
            return;
        }
        let Some(destination) = self.place_register(destination, span) else { return };
        self.emit_move(destination, operand, span);
    }

    pub(super) fn emit_statement_identity(
        &mut self,
        destination: Place<'tcx>,
        operand: &Operand<'tcx>,
        span: Span,
    ) {
        if let Operand::Copy(source) | Operand::Move(source) = operand
            && let Some(destination) = self.resolve_place(destination, span)
            && let Some(source) = self.resolve_place(*source, span)
            && !destination.dereference
            && destination.field.is_none()
            && !source.dereference
            && source.field.is_none()
            && let Some(value) = self.locals[source.local].clone()
        {
            self.locals[destination.local] = Some(value);
            return;
        }
        self.emit_use(destination, operand, span);
    }

    pub(super) fn emit_address_of(
        &mut self,
        destination: Place<'tcx>,
        target: Place<'tcx>,
        span: Span,
    ) {
        let Some((target, target_shape)) = self.address_target(target, span) else { return };
        let LocalShape::Scalar(target_ty) = target_shape else {
            self.error(span, "PolyASM pointer target must be scalar");
            return;
        };
        let Some(destination_shape) = self.place_shape(destination, span) else { return };
        let LocalShape::Pointer { pointee: destination_ty, .. } = destination_shape else {
            self.error(span, "PolyASM address must be assigned to a pointer");
            return;
        };
        if destination_ty != target_ty && destination_ty != ValueType::U32 {
            self.error(span, "PolyASM pointer changes its pointee type without an explicit cast");
            return;
        }
        if !self.set_alias(destination, target, span) {
            return;
        }
        let Some(destination) = self.place_register(destination, span) else { return };
        self.emit_pointer_address(destination, target, span);
    }

    pub(super) fn emit_pointer_use(
        &mut self,
        destination: Place<'tcx>,
        operand: &Operand<'tcx>,
        span: Span,
    ) {
        let Some(provenance) = self.operand_pointer_provenance(operand, span) else { return };
        if !self.set_alias(destination, provenance, span) {
            return;
        }
        let Some(destination) = self.place_register(destination, span) else { return };
        match operand {
            Operand::Constant(_) => self.emit_pointer_address(destination, provenance, span),
            Operand::Copy(place) | Operand::Move(place) => {
                let Some(source) = self.place_register(*place, span) else { return };
                let Some(ty) = self.place_register_type(destination) else {
                    self.error(span, "PolyASM pointer destination names an undeclared register");
                    return;
                };
                if !self.require_register_type(source, ty, span) {
                    return;
                }
                self.code.push(Instruction::unary(Opcode::Move, ty, destination, source));
            }
            Operand::RuntimeChecks(_) => {
                self.error(span, "PolyASM runtime-check operands cannot carry pointer provenance");
            }
        }
    }

    pub(super) fn emit_pointer_address(
        &mut self,
        destination: Register,
        provenance: PointerProvenance,
        span: Span,
    ) {
        let Some(ty) = self.place_register_type(destination) else {
            self.error(span, "PolyASM pointer destination names an undeclared register");
            return;
        };
        match provenance {
            PointerProvenance::Place(target) => {
                let Some(token) = address_token(target) else {
                    self.error(span, "PolyASM pointer token exceeds the bytecode address space");
                    return;
                };
                self.code.push(Instruction::constant(ty, destination, token));
            }
            PointerProvenance::Static(address) => {
                self.code.push(Instruction::data_address(
                    ty,
                    destination,
                    address.symbol,
                    address.offset,
                ));
            }
        }
    }

    pub(super) fn emit_cast(
        &mut self,
        destination: Place<'tcx>,
        kind: CastKind,
        operand: &Operand<'tcx>,
        span: Span,
    ) {
        let Some(source_ty) = self.operand_type(operand, span) else { return };
        let Some(destination_register) = self.place_register(destination, span) else { return };
        let Some(destination_ty) = self.place_register_type(destination_register) else { return };
        let pointer_destination =
            matches!(self.place_shape(destination, span), Some(LocalShape::Pointer { .. }));
        if source_ty != destination_ty {
            self.error(span, "PolyASM cast requires an unsupported scalar-width conversion");
            return;
        }
        if !matches!(
            kind,
            CastKind::FnPtrToPtr
                | CastKind::IntToInt
                | CastKind::PointerExposeProvenance
                | CastKind::PointerWithExposedProvenance
                | CastKind::PtrToPtr
                | CastKind::Transmute
        ) {
            self.error(span, "PolyASM cast is not representable in portable bytecode");
            return;
        }
        if pointer_destination {
            self.emit_pointer_use(destination, operand, span);
            return;
        }
        self.emit_move(destination_register, operand, span);
    }

    pub(super) fn emit_binary(
        &mut self,
        destination: Place<'tcx>,
        operation: BinOp,
        lhs: &Operand<'tcx>,
        rhs: &Operand<'tcx>,
        span: Span,
    ) {
        let Some(ty) = self.operand_type(lhs, span) else { return };
        if self.operand_type(rhs, span) != Some(ty) {
            self.error(span, "PolyASM binary operands have different register types");
            return;
        }
        let opcode = match operation {
            BinOp::Add | BinOp::AddUnchecked => Opcode::Add,
            BinOp::BitAnd => Opcode::And,
            BinOp::Div => Opcode::Div,
            BinOp::Eq => Opcode::CompareEq,
            BinOp::Ge => Opcode::CompareGe,
            BinOp::Gt => Opcode::CompareGt,
            BinOp::Le => Opcode::CompareLe,
            BinOp::Lt => Opcode::CompareLt,
            BinOp::Mul | BinOp::MulUnchecked => Opcode::Mul,
            BinOp::Ne => Opcode::CompareNe,
            BinOp::Rem => Opcode::Rem,
            BinOp::Shl | BinOp::ShlUnchecked => Opcode::ShiftLeft,
            BinOp::Shr | BinOp::ShrUnchecked => Opcode::ShiftRight,
            BinOp::Sub | BinOp::SubUnchecked => Opcode::Sub,
            BinOp::BitOr => Opcode::Or,
            BinOp::BitXor => Opcode::Xor,
            BinOp::AddWithOverflow
            | BinOp::Cmp
            | BinOp::MulWithOverflow
            | BinOp::Offset
            | BinOp::SubWithOverflow => {
                self.error(span, "PolyASM binary operation requires unsupported result semantics");
                return;
            }
        };
        let Some(destination) = self.place_register(destination, span) else { return };
        let result_ty = if matches!(
            opcode,
            Opcode::CompareEq
                | Opcode::CompareGe
                | Opcode::CompareGt
                | Opcode::CompareLe
                | Opcode::CompareLt
                | Opcode::CompareNe
        ) {
            ValueType::U32
        } else {
            ty
        };
        if !self.require_register_type(destination, result_ty, span) {
            return;
        }
        let Some(lhs) = self.operand_register(lhs, span) else { return };
        let Some(rhs) = self.operand_register(rhs, span) else { return };
        self.code.push(Instruction::binary(opcode, ty, destination, lhs, rhs));
    }

    pub(super) fn emit_overflow_binary(
        &mut self,
        destination: Place<'tcx>,
        operation: BinOp,
        lhs: &Operand<'tcx>,
        rhs: &Operand<'tcx>,
        span: Span,
    ) {
        let Some(ty) = self.operand_type(lhs, span) else { return };
        if self.operand_type(rhs, span) != Some(ty) || !ty.is_integer() {
            self.error(span, "PolyASM checked arithmetic requires equal integer operands");
            return;
        }
        let Some(fields) = self.place_fields(destination, span).map(<[Register]>::to_vec) else {
            return;
        };
        let [result, overflow] = fields.as_slice() else {
            self.error(span, "PolyASM checked arithmetic requires a (value, overflow) result");
            return;
        };
        if !self.require_register_type(*result, ty, span)
            || !self.require_register_type(*overflow, ValueType::U32, span)
        {
            return;
        }
        let Some(lhs_register) = self.operand_register(lhs, span) else { return };
        let Some(rhs_register) = self.operand_register(rhs, span) else { return };
        let arithmetic = match operation {
            BinOp::AddWithOverflow => Opcode::Add,
            BinOp::MulWithOverflow => Opcode::Mul,
            BinOp::SubWithOverflow => Opcode::Sub,
            _ => unreachable!(),
        };
        self.code.push(Instruction::binary(arithmetic, ty, *result, lhs_register, rhs_register));

        let signed = matches!(ty, ValueType::I32 | ValueType::I64);
        match (operation, signed) {
            (BinOp::AddWithOverflow, false) => self.code.push(Instruction::binary(
                Opcode::CompareLt,
                ty,
                *overflow,
                *result,
                lhs_register,
            )),
            (BinOp::SubWithOverflow, false) => self.code.push(Instruction::binary(
                Opcode::CompareLt,
                ty,
                *overflow,
                lhs_register,
                rhs_register,
            )),
            (BinOp::AddWithOverflow | BinOp::SubWithOverflow, true) => {
                let first = self.allocate(ty);
                let second = self.allocate(ty);
                let combined = self.allocate(ty);
                let zero = self.allocate(ty);
                if operation == BinOp::AddWithOverflow {
                    self.code.push(Instruction::binary(
                        Opcode::Xor,
                        ty,
                        first,
                        lhs_register,
                        *result,
                    ));
                    self.code.push(Instruction::binary(
                        Opcode::Xor,
                        ty,
                        second,
                        rhs_register,
                        *result,
                    ));
                } else {
                    self.code.push(Instruction::binary(
                        Opcode::Xor,
                        ty,
                        first,
                        lhs_register,
                        rhs_register,
                    ));
                    self.code.push(Instruction::binary(
                        Opcode::Xor,
                        ty,
                        second,
                        lhs_register,
                        *result,
                    ));
                }
                self.code.push(Instruction::binary(Opcode::And, ty, combined, first, second));
                self.code.push(Instruction::constant(ty, zero, 0));
                self.code.push(Instruction::binary(
                    Opcode::CompareLt,
                    ty,
                    *overflow,
                    combined,
                    zero,
                ));
            }
            (BinOp::MulWithOverflow, _) => {
                let constant = match rhs {
                    Operand::Constant(constant) => self.constant_immediate(constant, ty, span),
                    _ => None,
                };
                let Some(constant) = constant else {
                    self.error(
                        span,
                        "PolyASM checked multiplication currently requires a constant multiplier",
                    );
                    return;
                };
                if constant == 0 {
                    self.code.push(Instruction::constant(ValueType::U32, *overflow, 0));
                } else if signed && constant == -1 {
                    self.error(
                        span,
                        "PolyASM checked multiplication by -1 is not yet representable",
                    );
                } else {
                    let quotient = self.allocate(ty);
                    self.code.push(Instruction::binary(
                        Opcode::Div,
                        ty,
                        quotient,
                        *result,
                        rhs_register,
                    ));
                    self.code.push(Instruction::binary(
                        Opcode::CompareNe,
                        ty,
                        *overflow,
                        quotient,
                        lhs_register,
                    ));
                }
            }
            _ => unreachable!(),
        }
    }

    pub(super) fn emit_move(&mut self, destination: Register, operand: &Operand<'tcx>, span: Span) {
        let Some(ty) = self.operand_type(operand, span) else { return };
        if !self.require_register_type(destination, ty, span) {
            return;
        }
        match operand {
            Operand::RuntimeChecks(checks) => {
                self.code.push(Instruction::constant(
                    ty,
                    destination,
                    i64::from(checks.value(self.tcx.sess)),
                ));
            }
            Operand::Constant(constant) => {
                let Some(immediate) = self.constant_immediate(constant, ty, span) else { return };
                self.code.push(Instruction::constant(ty, destination, immediate));
            }
            Operand::Copy(place) | Operand::Move(place) => {
                let Some(source) = self.place_register(*place, span) else { return };
                self.code.push(Instruction::unary(Opcode::Move, ty, destination, source));
            }
        }
    }

    pub(super) fn operand_register(
        &mut self,
        operand: &Operand<'tcx>,
        span: Span,
    ) -> Option<Register> {
        match operand {
            Operand::RuntimeChecks(checks) => {
                let register = self.allocate(ValueType::U32);
                self.code.push(Instruction::constant(
                    ValueType::U32,
                    register,
                    i64::from(checks.value(self.tcx.sess)),
                ));
                Some(register)
            }
            Operand::Constant(constant) => {
                let ty = self.operand_type(operand, span)?;
                let immediate = self.constant_immediate(constant, ty, span)?;
                let register = self.allocate(ty);
                self.code.push(Instruction::constant(ty, register, immediate));
                Some(register)
            }
            Operand::Copy(place) | Operand::Move(place) => self.place_register(*place, span),
        }
    }

    pub(super) fn operand_type(&self, operand: &Operand<'tcx>, span: Span) -> Option<ValueType> {
        match operand {
            Operand::RuntimeChecks(_) => Some(ValueType::U32),
            Operand::Copy(place) | Operand::Move(place) => {
                self.place_shape(*place, span).and_then(|shape| shape.register_type())
            }
            Operand::Constant(constant) => {
                let ty = self.monomorphize(constant.const_.ty());
                encoded_type(self.tcx, ty).or_else(|| {
                    self.error(span, "PolyASM constant is not a scalar value");
                    None
                })
            }
        }
    }

    pub(super) fn constant_immediate(
        &self,
        constant: &ConstOperand<'tcx>,
        ty: ValueType,
        span: Span,
    ) -> Option<i64> {
        let constant_span = constant.span;
        let constant = self.monomorphize(constant.const_);
        let value = constant
            .eval(self.tcx, ty::TypingEnv::fully_monomorphized(), constant_span)
            .ok()
            .and_then(|value| value.try_to_scalar_int());
        let Some(value) = value else {
            self.error(span, "PolyASM constant is not an immediate scalar");
            return None;
        };
        let bits = value.to_bits_unchecked();
        if bits > u128::from(u64::MAX) {
            self.error(span, "PolyASM constants wider than 64 bits are unsupported");
            return None;
        }
        let bits = bits as u64;
        Some(match ty {
            ValueType::I32 => i64::from(bits as u32 as i32),
            ValueType::F32 | ValueType::U32 => i64::from(bits as u32),
            ValueType::F64 | ValueType::I64 | ValueType::U64 => bits as i64,
            ValueType::V128 | ValueType::V256 | ValueType::V512 => {
                self.error(span, "PolyASM vector constants are unsupported");
                return None;
            }
        })
    }
}
