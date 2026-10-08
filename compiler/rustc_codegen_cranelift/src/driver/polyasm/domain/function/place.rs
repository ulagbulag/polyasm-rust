//! Branch patching, place resolution, and pointer-provenance lowering.

use rustc_middle::mir::interpret::{GlobalAlloc, Scalar};
use rustc_middle::mir::{BasicBlock, ConstOperand, Operand, Place, ProjectionElem};
use rustc_middle::ty::{self, Instance};
use rustc_span::Span;

use super::super::model::{Instruction, Register, ValueType};
use super::{
    FunctionEmitter, LocalRegisters, LocalShape, PointerProvenance, SimplePlace, StaticAddress,
};

impl<'a, 'tcx> FunctionEmitter<'a, 'tcx> {
    pub(super) fn push_branch(&mut self, target: BasicBlock) {
        let instruction = self.code.len();
        self.code.push(Instruction::branch(0));
        self.patches.push((instruction, target));
    }

    pub(super) fn push_branch_if(&mut self, condition: Register, target: BasicBlock) {
        let instruction = self.code.len();
        self.code.push(Instruction::branch_if(condition, 0));
        self.patches.push((instruction, target));
    }

    pub(super) fn place_fields(&self, place: Place<'tcx>, span: Span) -> Option<&[Register]> {
        let place = self.resolve_place(place, span)?;
        if place.field.is_some() {
            self.error(span, "PolyASM projected place cannot name an aggregate");
            return None;
        }
        self.local(place.local, span).map(|local| local.fields.as_slice())
    }

    pub(super) fn place_register(&self, place: Place<'tcx>, span: Span) -> Option<Register> {
        let place = self.resolve_place(place, span)?;
        let local = self.local(place.local, span)?;
        match place.field {
            Some(field) => local.fields.get(field).copied().or_else(|| {
                self.error(span, "PolyASM aggregate projection is out of bounds");
                None
            }),
            None if local.fields.len() == 1 => local.fields.first().copied(),
            None => {
                self.error(span, "PolyASM scalar place has a non-scalar type");
                None
            }
        }
    }

    pub(super) fn place_register_type(&self, register: Register) -> Option<ValueType> {
        self.registers.get(usize::from(register.number())).copied()
    }

    pub(super) fn place_shape(&self, place: Place<'tcx>, span: Span) -> Option<LocalShape> {
        let place = self.resolve_place(place, span)?;
        self.place_shape_from_simple(place, span)
    }

    pub(super) fn place_shape_from_simple(
        &self,
        place: SimplePlace,
        span: Span,
    ) -> Option<LocalShape> {
        let local = self.local(place.local, span)?;
        match place.field {
            Some(field) => match &local.shape {
                LocalShape::Tuple(types) => {
                    types.get(field).copied().map(LocalShape::Scalar).or_else(|| {
                        self.error(span, "PolyASM aggregate projection is out of bounds");
                        None
                    })
                }
                _ => {
                    self.error(span, "PolyASM projects a non-aggregate local");
                    None
                }
            },
            None => Some(local.shape.clone()),
        }
    }

    pub(super) fn local(&self, local: usize, span: Span) -> Option<&LocalRegisters> {
        self.locals.get(local).and_then(Option::as_ref).or_else(|| {
            self.error(span, "PolyASM MIR uses an undeclared local");
            None
        })
    }

    pub(super) fn address_target(
        &self,
        place: Place<'tcx>,
        span: Span,
    ) -> Option<(PointerProvenance, LocalShape)> {
        let mut output =
            SimplePlace { dereference: false, field: None, local: place.local.as_usize() };
        for projection in place.projection {
            match projection {
                ProjectionElem::Deref if !output.dereference && output.field.is_none() => {
                    output.dereference = true;
                }
                ProjectionElem::Field(field, _) if output.field.is_none() => {
                    output.field = Some(field.as_usize());
                }
                ProjectionElem::OpaqueCast(_) | ProjectionElem::UnwrapUnsafeBinder(_) => {}
                _ => {
                    self.error(span, "PolyASM place projection is not statically addressable");
                    return None;
                }
            }
        }
        if !output.dereference {
            let shape = self.place_shape_from_simple(output, span)?;
            return Some((PointerProvenance::Place(output), shape));
        }
        if output.field.is_some() {
            self.error(span, "PolyASM cannot project a field before resolving pointer provenance");
            return None;
        }
        let local = self.local(output.local, span)?;
        let LocalShape::Pointer { pointee, .. } = &local.shape else {
            self.error(span, "PolyASM dereferences a non-pointer local");
            return None;
        };
        let provenance = local.alias.or_else(|| {
            self.error(span, "PolyASM pointer has no statically proven provenance");
            None
        })?;
        Some((provenance, LocalShape::Scalar(*pointee)))
    }

    pub(super) fn operand_pointer_provenance(
        &self,
        operand: &Operand<'tcx>,
        span: Span,
    ) -> Option<PointerProvenance> {
        match operand {
            Operand::Constant(constant) => {
                self.constant_static_address(constant, span).map(PointerProvenance::Static)
            }
            Operand::Copy(place) | Operand::Move(place) => {
                let source = self.resolve_place(*place, span)?;
                if source.dereference || source.field.is_some() {
                    self.error(span, "PolyASM pointer operand must be an unprojected local");
                    return None;
                }
                let local = self.local(source.local, span)?;
                if !matches!(&local.shape, LocalShape::Pointer { .. }) {
                    self.error(span, "PolyASM pointer operand has a non-pointer type");
                    return None;
                }
                local.alias.or_else(|| {
                    self.error(span, "PolyASM pointer has no statically proven provenance");
                    None
                })
            }
            Operand::RuntimeChecks(_) => {
                self.error(span, "PolyASM runtime-check operands cannot carry pointer provenance");
                None
            }
        }
    }

    pub(super) fn constant_static_address(
        &self,
        constant: &ConstOperand<'tcx>,
        span: Span,
    ) -> Option<StaticAddress> {
        let constant = self.monomorphize(constant.const_);
        let Some(Scalar::Ptr(pointer, _)) =
            constant.try_eval_scalar(self.tcx, ty::TypingEnv::fully_monomorphized())
        else {
            self.error(span, "PolyASM pointer constant is not a fixed static address");
            return None;
        };
        let (provenance, pointer_offset) = pointer.prov_and_relative_offset();
        let Some(GlobalAlloc::Static(def_id)) =
            self.tcx.try_get_global_alloc(provenance.alloc_id())
        else {
            self.error(span, "PolyASM pointer constant does not name fixed static data");
            return None;
        };
        if self.tcx.is_thread_local_static(def_id) || self.tcx.is_foreign_item(def_id) {
            self.error(span, "PolyASM cannot assign a fixed guest address to TLS or foreign data");
            return None;
        }
        let name = self.tcx.symbol_name(Instance::mono(self.tcx, def_id)).name.to_string();
        let Some(&symbol) = self.indices.get(&name) else {
            self.error(span, "PolyASM static data is outside the emitted graph");
            return None;
        };
        let Some(offset) = u32::try_from(pointer_offset.bytes()).ok() else {
            self.error(span, "PolyASM static pointer offset exceeds guest memory");
            return None;
        };
        Some(StaticAddress { offset, symbol })
    }

    pub(super) fn resolve_place(&self, place: Place<'tcx>, span: Span) -> Option<SimplePlace> {
        let mut output =
            SimplePlace { dereference: false, field: None, local: place.local.as_usize() };
        for projection in place.projection {
            match projection {
                ProjectionElem::Deref if !output.dereference && output.field.is_none() => {
                    output.dereference = true;
                }
                ProjectionElem::Field(field, _) if output.field.is_none() => {
                    output.field = Some(field.as_usize());
                }
                ProjectionElem::OpaqueCast(_) | ProjectionElem::UnwrapUnsafeBinder(_) => {}
                _ => {
                    self.error(span, "PolyASM place projection is not statically addressable");
                    return None;
                }
            }
        }
        if !output.dereference {
            return Some(output);
        }
        if output.field.is_some() {
            self.error(span, "PolyASM cannot project a field before resolving pointer provenance");
            return None;
        }
        match self.local(output.local, span)?.alias {
            Some(PointerProvenance::Place(target)) => Some(target),
            Some(PointerProvenance::Static(_)) => {
                self.error(
                    span,
                    "PolyASM fixed static memory requires an explicit supported memory operation",
                );
                None
            }
            None => {
                self.error(span, "PolyASM pointer has no statically proven provenance");
                None
            }
        }
    }

    pub(super) fn set_alias(
        &mut self,
        destination: Place<'tcx>,
        target: PointerProvenance,
        span: Span,
    ) -> bool {
        let Some(destination) = self.resolve_place(destination, span) else { return false };
        if destination.dereference || destination.field.is_some() {
            self.error(span, "PolyASM pointer destination must be an unprojected local");
            return false;
        }
        let Some(local) = self.locals.get_mut(destination.local).and_then(Option::as_mut) else {
            self.error(span, "PolyASM pointer destination is undeclared");
            return false;
        };
        if !matches!(local.shape, LocalShape::Pointer { .. }) {
            self.error(span, "PolyASM address is assigned to a non-pointer local");
            return false;
        }
        local.alias = Some(target);
        true
    }

    pub(super) fn require_register_type(
        &self,
        register: Register,
        ty: ValueType,
        span: Span,
    ) -> bool {
        if self.registers.get(usize::from(register.number())) == Some(&ty) {
            true
        } else {
            self.error(span, "PolyASM assignment changes a local register type");
            false
        }
    }
}
