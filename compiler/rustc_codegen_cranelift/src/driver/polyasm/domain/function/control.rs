//! MIR terminator, call, property-marker, and endian-operation lowering.

use rustc_codegen_ssa::polyasm::{
    normalized_polyasm_property_holds, normalized_polyasm_statement_property_holds,
    normalized_polyasm_static_schedule, polyasm_statement_selection,
    polyasm_static_callable_selection, polyasm_static_callable_selection_on,
    polyasm_static_clock_hz,
};
use rustc_hir::def::DefKind;
use rustc_middle::mir::{BasicBlock, Location, Operand, Place, RETURN_PLACE, TerminatorKind};
use rustc_middle::mono::resolve_polyasm_callable;
use rustc_middle::ty::{self, Instance};
use rustc_span::{Span, sym};

use super::super::capability::{
    CAP_LINUX_SAFE, CAP_POINTER_ACCELERATORS, CAP_XDP, property_capability,
};
use super::super::model::{Instruction, Opcode, Register, ValueType};
use super::{
    CompilerMarker, FunctionEmitter, PointerProvenance, StatementWarrant, StaticCallableWitness,
    compiler_marker, is_core_endian_helper, pointer_value_type, polyasm_u64_const,
    volatile_value_type,
};

impl<'a, 'tcx> FunctionEmitter<'a, 'tcx> {
    pub(super) fn emit_terminator(
        &mut self,
        terminator: &TerminatorKind<'tcx>,
        location: Location,
        span: Span,
    ) {
        match terminator {
            TerminatorKind::Goto { target }
            | TerminatorKind::Drop { target, .. }
            | TerminatorKind::FalseEdge { real_target: target, .. }
            | TerminatorKind::FalseUnwind { real_target: target, .. } => self.push_branch(*target),
            TerminatorKind::SwitchInt { discr, targets } => {
                let Some(ty) = self.operand_type(discr, span) else { return };
                if !ty.is_integer() {
                    self.error(span, "PolyASM switch discriminator must be integer-like");
                    return;
                }
                let Some(discriminator) = self.operand_register(discr, span) else { return };
                for (value, target) in targets.iter() {
                    if value > u128::from(u64::MAX) {
                        self.error(span, "PolyASM switch value exceeds 64 bits");
                        return;
                    }
                    let expected = self.allocate(ty);
                    self.code.push(Instruction::constant(ty, expected, value as u64 as i64));
                    let equal = self.allocate(ValueType::U32);
                    self.code.push(Instruction::binary(
                        Opcode::CompareEq,
                        ty,
                        equal,
                        discriminator,
                        expected,
                    ));
                    self.push_branch_if(equal, target);
                }
                self.push_branch(targets.otherwise());
            }
            TerminatorKind::Assert { cond, expected, target, .. } => {
                let Some(mut condition) = self.operand_register(cond, span) else { return };
                let Some(ty) = self.place_register_type(condition) else { return };
                if !*expected {
                    let zero = self.allocate(ty);
                    self.code.push(Instruction::constant(ty, zero, 0));
                    let inverted = self.allocate(ValueType::U32);
                    self.code.push(Instruction::binary(
                        Opcode::CompareEq,
                        ty,
                        inverted,
                        condition,
                        zero,
                    ));
                    condition = inverted;
                }
                self.push_branch_if(condition, *target);
                self.code.push(Instruction::trap());
            }
            TerminatorKind::Call { func, args, destination, target, .. } => {
                self.emit_call(func, args, *destination, *target, location, span);
            }
            TerminatorKind::Return => {
                if let Some(ty) = self.result {
                    let place = Place::from(RETURN_PLACE);
                    let Some(source) = self.place_register(place, span) else { return };
                    self.code.push(Instruction::return_(ty, source));
                } else {
                    self.code.push(Instruction::return_void());
                }
            }
            TerminatorKind::Unreachable
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(..) => self.code.push(Instruction::trap()),
            TerminatorKind::CoroutineDrop
            | TerminatorKind::InlineAsm { .. }
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::Yield { .. } => {
                self.error(span, "terminator is not supported by native PolyASM codegen");
                self.code.push(Instruction::trap());
            }
        }
    }

    pub(super) fn emit_call(
        &mut self,
        func: &Operand<'tcx>,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
        target: Option<BasicBlock>,
        location: Location,
        span: Span,
    ) {
        let Some((def_id, generic_args)) = func.const_fn_def() else {
            self.error(span, "PolyASM supports only statically resolved function calls");
            return;
        };
        let generic_args = self.monomorphize(generic_args);
        if self.tcx.is_intrinsic(def_id, sym::cold_path) {
            if !args.is_empty() {
                self.error(span, "PolyASM cold-path hint has an invalid ABI");
                return;
            }
            // `cold_path` only guides the optimizing backend. It carries zero
            // observable operations and zero callable runtime symbols, so close its
            // MIR edge exactly as Cranelift does instead of retaining a call
            // to a function that the interchange backend correctly eliminates.
            if let Some(target) = target {
                self.push_branch(target);
            } else {
                self.code.push(Instruction::trap());
            }
            return;
        }
        if self.emit_compiler_marker(
            def_id,
            generic_args,
            args,
            destination,
            target,
            location,
            span,
        ) {
            return;
        }
        if self.emit_volatile_read(def_id, args, destination, target, span) {
            return;
        }
        let callee = Instance::expect_resolve(
            self.tcx,
            ty::TypingEnv::fully_monomorphized(),
            def_id,
            generic_args,
            span,
        );
        self.source_calls.insert(location.block, callee);
        if self.tcx.polyasm_witness_only_wrapper(callee) {
            if let Some(target) = target {
                self.push_branch(target);
            } else {
                self.code.push(Instruction::trap());
            }
            return;
        }
        if self.emit_endian_call(callee, args, destination, target, span) {
            return;
        }
        let name = self.tcx.symbol_name(callee).name.to_string();
        let Some(&function) = self.indices.get(&name) else {
            self.error(span, "PolyASM call target is outside the emitted graph");
            return;
        };
        let mut values = Vec::with_capacity(args.len());
        for argument in args {
            let Some(value) = self.operand_register(&argument.node, argument.span) else { return };
            let Some(ty) = self.place_register_type(value) else {
                self.error(span, "PolyASM call argument names an undeclared register");
                return;
            };
            values.push((ty, value));
        }
        // The arguments occupy a run of registers declared back to back, so
        // naming the first of them names all of them: how many follow is the
        // callee's declared parameter list, which the call leaves unrestated.
        let mut slots = Vec::with_capacity(values.len());
        for &(ty, _) in &values {
            slots.push(self.allocate(ty));
        }
        let first_arg = slots.first().copied();
        for (&slot, (ty, source)) in slots.iter().zip(values) {
            self.code.push(Instruction::unary(Opcode::Move, ty, slot, source));
        }
        let destination_shape = self.place_shape(destination, span);
        let (ty, destination) = match destination_shape.and_then(|shape| shape.register_type()) {
            Some(ty) => {
                let Some(destination) = self.place_register(destination, span) else { return };
                (ty, Some(destination))
            }
            None => (ValueType::I32, None),
        };
        let call_pc = u32::try_from(self.code.len())
            .unwrap_or_else(|_| self.tcx.dcx().fatal("PolyASM call PC exceeds u32"));
        self.calls.insert(location.block, (callee, call_pc));
        self.code.push(Instruction::call(ty, destination, first_arg, function));
        if let Some(target) = target {
            self.push_branch(target);
        } else {
            self.code.push(Instruction::trap());
        }
    }

    pub(super) fn emit_volatile_read(
        &mut self,
        def_id: rustc_hir::def_id::DefId,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
        target: Option<BasicBlock>,
        span: Span,
    ) -> bool {
        if !self.tcx.is_diagnostic_item(sym::ptr_read_volatile, def_id)
            && !self.tcx.is_intrinsic(def_id, sym::volatile_load)
        {
            return false;
        }
        let [argument] = args else {
            self.error(span, "PolyASM volatile read requires exactly one pointer");
            return true;
        };
        let Some(provenance @ PointerProvenance::Static(_)) =
            self.operand_pointer_provenance(&argument.node, argument.span)
        else {
            self.error(
                argument.span,
                "PolyASM volatile read requires compiler-proven fixed static memory",
            );
            return true;
        };
        let pointer_ty = self.monomorphize(argument.node.ty(&self.body.local_decls, self.tcx));
        let pointee = match *pointer_ty.kind() {
            ty::RawPtr(pointee, _) | ty::Ref(_, pointee, _) => pointee,
            _ => {
                self.error(argument.span, "PolyASM volatile read argument is not a pointer");
                return true;
            }
        };
        let Some(ty) = volatile_value_type(self.tcx, pointee) else {
            self.error(
                argument.span,
                "PolyASM volatile read requires a native-width integer scalar",
            );
            return true;
        };
        let Some(destination) = self.place_register(destination, span) else { return true };
        if !self.require_register_type(destination, ty, span) {
            return true;
        }
        let address = match &argument.node {
            Operand::Constant(_) => {
                let address = self.allocate(pointer_value_type(self.tcx));
                self.emit_pointer_address(address, provenance, argument.span);
                address
            }
            Operand::Copy(place) | Operand::Move(place) => {
                let Some(address) = self.place_register(*place, argument.span) else {
                    return true;
                };
                address
            }
            Operand::RuntimeChecks(_) => {
                self.error(argument.span, "PolyASM volatile read has no fixed pointer operand");
                return true;
            }
        };
        self.capability_ceiling.set(
            self.capability_ceiling.get() & !(CAP_XDP | CAP_LINUX_SAFE | CAP_POINTER_ACCELERATORS),
        );
        self.code.push(Instruction::load_little(ty, destination, address));
        if let Some(target) = target {
            self.push_branch(target);
        } else {
            self.code.push(Instruction::trap());
        }
        true
    }

    pub(super) fn emit_compiler_marker(
        &mut self,
        def_id: rustc_hir::def_id::DefId,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
        target: Option<BasicBlock>,
        location: Location,
        span: Span,
    ) -> bool {
        let Some(marker) = compiler_marker(self.tcx, def_id) else {
            return false;
        };
        if marker == CompilerMarker::RequireStatement {
            self.emit_statement_warrant(generic_args, args, destination, target, location, span);
            return true;
        }
        if marker == CompilerMarker::InvokeStaticFasterExact {
            self.emit_static_invoke_exact(generic_args, args, destination, target, span);
            return true;
        }
        if marker == CompilerMarker::InvokeStaticFasterOn {
            self.emit_static_invoke_on(generic_args, args, destination, target, span);
            return true;
        }
        if marker == CompilerMarker::InvokeStaticFasterProven {
            self.emit_static_invoke_proven(generic_args, args, destination, target, span);
            return true;
        }
        if args.len() != 1 {
            self.error(span, "PolyASM compiler witness call has an invalid ABI");
            return true;
        }
        let callable_index = if marker == CompilerMarker::BindStaticClock { 1 } else { 2 };
        let callable_ty = generic_args.type_at(callable_index);
        let callable = resolve_polyasm_callable(self.tcx, callable_ty, span);
        let Some(callable) = callable else {
            self.tcx.dcx().span_err(
                span,
                "PolyASM compiler witness lost the concrete callable identity before code generation",
            );
            return true;
        };

        match marker {
            // A request carries intent only. Device admission belongs to the
            // host's own check over the decoded callable record.
            CompilerMarker::RequestOffload => {
                if let Some(next) = target {
                    self.offload_markers.push(super::OffloadMarker {
                        callable,
                        next,
                        request: super::super::super::debug::marker_request(
                            self.tcx,
                            generic_args.type_at(0),
                            generic_args.type_at(1),
                        ),
                    });
                }
            }
            // An offload request carries the same property a positive
            // `Always` assertion carries, and it is rechecked from
            // normalized MIR here for the same reason: a request is a
            // claim about a body, and the claim is checked again where
            // the body is lowered. The request answers the callable it
            // named, which is the zero-sized function item or closure
            // already in hand, so the destination takes zero registers and
            // the request closes by branching.
            CompilerMarker::Offload
            | CompilerMarker::RequireAlways
            | CompilerMarker::RequireNotAlways => {
                let property = generic_args.type_at(0);
                let architecture = generic_args.type_at(1);
                let expected = marker != CompilerMarker::RequireNotAlways;
                match normalized_polyasm_property_holds(self.tcx, callable, property, architecture)
                {
                    Some(actual) if actual == expected => {
                        if !expected
                            && let Some(capability) = property_capability(self.tcx, property)
                        {
                            let name = self.tcx.symbol_name(callable).name.to_string();
                            *self.capability_exclusions.entry(name).or_default() |= capability;
                        }
                    }
                    Some(_) => {
                        self.tcx.dcx().span_err(
                            span,
                            "PolyASM compiler witness changed after normalized MIR; the accepted Always assertion is unsound",
                        );
                    }
                    None => {
                        self.tcx.dcx().span_err(
                            span,
                            "PolyASM rechecks this callable from normalized MIR, and the recheck stops here",
                        );
                    }
                }
                if marker == CompilerMarker::Offload {
                    if let Some(next) = target {
                        self.offload_markers.push(super::OffloadMarker {
                            callable,
                            next,
                            request: super::super::super::debug::marker_request(
                                self.tcx,
                                property,
                                architecture,
                            ),
                        });
                    }
                }
            }
            CompilerMarker::BindStaticClock => {
                let Some(requested_cycles) = polyasm_u64_const(self.tcx, generic_args.const_at(2))
                else {
                    self.tcx.dcx().span_err(
                        span,
                        "PolyASM static-clock cycle count was not concrete at code generation",
                    );
                    return true;
                };
                let Some(clock_hz) = polyasm_static_clock_hz(self.tcx, generic_args.type_at(0))
                else {
                    self.tcx.dcx().span_err(
                        span,
                        "PolyASM static-clock binding requires a concrete registered board architecture",
                    );
                    return true;
                };
                let actual =
                    normalized_polyasm_static_schedule(self.tcx, callable, generic_args.type_at(0));
                if actual != Some(requested_cycles) {
                    self.tcx.dcx().span_err(
                        span,
                        format!(
                            "PolyASM normalized MIR accepted source schedule {actual:?}, but the static-clock binding requests {requested_cycles} cycles at {clock_hz} Hz"
                        ),
                    );
                }
                let Some(destination) = self.resolve_place(destination, span) else {
                    return true;
                };
                if destination.dereference || destination.field.is_some() {
                    self.tcx.dcx().span_err(
                        span,
                        "PolyASM static-clock witness must bind an unprojected local",
                    );
                    return true;
                }
                self.static_callables.insert(
                    destination.local,
                    StaticCallableWitness {
                        architecture: generic_args.type_at(0),
                        callable,
                        clock_hz,
                        cycles: requested_cycles,
                    },
                );
            }
            CompilerMarker::InvokeStaticFasterExact
            | CompilerMarker::InvokeStaticFasterOn
            | CompilerMarker::InvokeStaticFasterProven => {
                unreachable!()
            }
            CompilerMarker::RequireStatement => unreachable!(),
        }
        if let Some(target) = target {
            self.push_branch(target);
        } else {
            self.code.push(Instruction::trap());
        }
        true
    }

    pub(super) fn emit_statement_warrant(
        &mut self,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
        target: Option<BasicBlock>,
        location: Location,
        span: Span,
    ) {
        let [argument] = args else {
            self.tcx.dcx().span_err(span, "PolyASM statement warrant marker has an invalid ABI");
            return;
        };
        let property = generic_args.type_at(0);
        let architecture = generic_args.type_at(1);
        let Some(selection) = polyasm_statement_selection(self.tcx, self.instance, location) else {
            self.tcx.dcx().span_err(
                span,
                "PolyASM statement warrant lost its exact normalized-MIR dependency slice",
            );
            return;
        };
        let warrant = normalized_polyasm_statement_property_holds(
            self.tcx,
            self.instance,
            &selection,
            property,
            architecture,
        );
        match (warrant, property_capability(self.tcx, property)) {
            (Some(true), Some(capability)) => {
                self.statement_warrants.push(StatementWarrant { capability, selection });
            }
            (Some(false), _) => {
                self.tcx.dcx().span_err(
                    span,
                    "PolyASM statement witness changed after normalized MIR; the accepted Always assertion is unsound",
                );
            }
            (None, _) | (_, None) => {
                self.tcx.dcx().span_err(
                    span,
                    "PolyASM rechecks this exact statement expression during code generation, and the recheck stops here",
                );
            }
        }
        let evaluation_start = self.code.len();
        let argument_ty = self.monomorphize(argument.node.ty(&self.body.local_decls, self.tcx));
        if !argument_ty.is_unit() {
            self.emit_statement_identity(destination, &argument.node, argument.span);
        }
        self.record_emitted_range(location, evaluation_start);
        if let Some(target) = target {
            self.push_branch(target);
        } else {
            self.code.push(Instruction::trap());
        }
    }

    pub(super) fn emit_static_invoke_proven(
        &mut self,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
        target: Option<BasicBlock>,
        span: Span,
    ) {
        let [lhs, rhs] = args else {
            self.tcx.dcx().span_err(span, "PolyASM static selection has an invalid ABI");
            return;
        };
        let Some(lhs_witness) = self.static_callable_witness(&lhs.node, lhs.span) else {
            self.tcx.dcx().span_err(
                lhs.span,
                "PolyASM static selection lost its left compiler-bound callable witness",
            );
            return;
        };
        let Some(rhs_witness) = self.static_callable_witness(&rhs.node, rhs.span) else {
            self.tcx.dcx().span_err(
                rhs.span,
                "PolyASM static selection lost its right compiler-bound callable witness",
            );
            return;
        };
        let Some(selection) = polyasm_static_callable_selection(self.tcx, generic_args, span)
        else {
            self.tcx
                .dcx()
                .span_err(span, "PolyASM static selection lost its two exact normalized schedules");
            return;
        };
        let schedule = (
            polyasm_u64_const(self.tcx, generic_args.const_at(5)),
            polyasm_static_clock_hz(self.tcx, generic_args.type_at(2)),
            polyasm_u64_const(self.tcx, generic_args.const_at(6)),
            polyasm_static_clock_hz(self.tcx, generic_args.type_at(3)),
        );
        let (Some(lhs_cycles), Some(lhs_clock_hz), Some(rhs_cycles), Some(rhs_clock_hz)) = schedule
        else {
            self.tcx.dcx().span_err(
                span,
                "PolyASM static selection schedules were not concrete at code generation",
            );
            return;
        };
        if lhs_clock_hz == 0 || rhs_clock_hz == 0 {
            self.tcx
                .dcx()
                .span_err(span, "PolyASM static selection requires non-zero clock frequencies");
            return;
        }
        let lhs_architecture = generic_args.type_at(2);
        let rhs_architecture = generic_args.type_at(3);
        if lhs_witness.callable != selection.lhs()
            || lhs_witness.architecture != lhs_architecture
            || lhs_witness.cycles != lhs_cycles
            || lhs_witness.clock_hz != lhs_clock_hz
            || rhs_witness.callable != selection.rhs()
            || rhs_witness.architecture != rhs_architecture
            || rhs_witness.cycles != rhs_cycles
            || rhs_witness.clock_hz != rhs_clock_hz
        {
            self.tcx.dcx().span_err(
                span,
                "PolyASM static selection does not match its compiler-bound callable witness",
            );
            return;
        }
        let selected = if selection.selected_left() { lhs_witness } else { rhs_witness };
        self.emit_selected_static_call(selected.callable, destination, target, span);
    }

    pub(super) fn emit_static_invoke_exact(
        &mut self,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
        target: Option<BasicBlock>,
        span: Span,
    ) {
        let [lhs, rhs] = args else {
            self.tcx.dcx().span_err(span, "PolyASM exact static selection has an invalid ABI");
            return;
        };
        let lhs_ty = self.monomorphize(lhs.node.ty(&self.body.local_decls, self.tcx));
        let rhs_ty = self.monomorphize(rhs.node.ty(&self.body.local_decls, self.tcx));
        if lhs_ty != generic_args.type_at(0) || rhs_ty != generic_args.type_at(1) {
            self.tcx.dcx().span_err(
                span,
                "PolyASM exact static selection operands changed callable identity",
            );
            return;
        }
        let Some(selection) = polyasm_static_callable_selection(self.tcx, generic_args, span)
        else {
            self.tcx.dcx().span_err(
                span,
                "PolyASM exact static selection requires two closed compiler-checked schedules",
            );
            return;
        };
        self.emit_selected_static_call(selection.selected(), destination, target, span);
    }

    pub(super) fn emit_static_invoke_on(
        &mut self,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
        target: Option<BasicBlock>,
        span: Span,
    ) {
        let [lhs, rhs] = args else {
            self.tcx.dcx().span_err(
                span,
                "PolyASM architecture-relative static selection has an invalid ABI",
            );
            return;
        };
        let lhs_ty = self.monomorphize(lhs.node.ty(&self.body.local_decls, self.tcx));
        let rhs_ty = self.monomorphize(rhs.node.ty(&self.body.local_decls, self.tcx));
        if lhs_ty != generic_args.type_at(0) || rhs_ty != generic_args.type_at(1) {
            self.tcx.dcx().span_err(
                span,
                "PolyASM architecture-relative static selection operands changed callable identity",
            );
            return;
        }
        let selection = match polyasm_static_callable_selection_on(self.tcx, generic_args, span) {
            Ok(selection) => selection,
            Err(error) => {
                self.tcx.dcx().span_err(error.span(), error.message().to_owned());
                return;
            }
        };
        self.emit_selected_static_call(selection.selected(), destination, target, span);
    }

    fn emit_selected_static_call(
        &mut self,
        callable: Instance<'tcx>,
        destination: Place<'tcx>,
        target: Option<BasicBlock>,
        span: Span,
    ) {
        let name = self.tcx.symbol_name(callable).name.to_string();
        let Some(&function) = self.indices.get(&name) else {
            self.error(span, "PolyASM selected static call target is outside the emitted graph");
            return;
        };
        let (ty, destination) =
            match self.place_shape(destination, span).and_then(|shape| shape.register_type()) {
                Some(ty) => {
                    let Some(destination) = self.place_register(destination, span) else { return };
                    (ty, Some(destination))
                }
                None => (ValueType::I32, None),
            };
        if self.tcx.def_kind(callable.def_id()) == DefKind::Closure
            && !callable.args.as_closure().tupled_upvars_ty().is_unit()
        {
            self.error(span, "PolyASM selected static closure acquired a capture after check");
            return;
        }
        self.code.push(Instruction::call(ty, destination, None, function));
        if let Some(target) = target {
            self.push_branch(target);
        } else {
            self.code.push(Instruction::trap());
        }
    }

    pub(super) fn static_callable_witness(
        &self,
        operand: &Operand<'tcx>,
        span: Span,
    ) -> Option<StaticCallableWitness<'tcx>> {
        let place = match operand {
            Operand::Copy(place) | Operand::Move(place) => self.resolve_place(*place, span)?,
            Operand::Constant(_) | Operand::RuntimeChecks(_) => return None,
        };
        if place.dereference || place.field.is_some() {
            return None;
        }
        self.static_callables.get(&place.local).copied()
    }

    pub(super) fn emit_endian_call(
        &mut self,
        callee: Instance<'tcx>,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: Place<'tcx>,
        target: Option<BasicBlock>,
        span: Span,
    ) -> bool {
        // Only compiler-identified core/compiler-builtins bodies implement
        // canonical storage. `staged_api` is user-selectable and therefore
        // stays outside the authority boundary.
        if self.tcx.is_polyasm_sysroot_crate(self.instance.def_id().krate) {
            return false;
        }
        if !is_core_endian_helper(self.tcx, callee) {
            return false;
        }
        let Some(name) = self.tcx.opt_item_name(callee.def_id()) else {
            return false;
        };
        let name = name.as_str();
        if args.len() != 1 {
            self.error(span, "explicit byte conversion expects one scalar or byte-array argument");
            return true;
        }
        match name {
            "little_endian_transmute" | "big_endian_transmute" => {
                let Some(source_ty) = self.operand_type(&args[0].node, span) else {
                    return true;
                };
                let Some(source) = self.operand_register(&args[0].node, span) else {
                    return true;
                };
                let Some(destination) = self.place_register(destination, span) else {
                    return true;
                };
                let Some(destination_ty) = self.place_register_type(destination) else {
                    return true;
                };
                if !destination_ty.reinterprets(source_ty) {
                    self.error(span, "ordered PolyASM reinterpret requires F32/U32 or F64/U64");
                    return true;
                }
                self.code.push(Instruction::reinterpret(
                    name == "big_endian_transmute",
                    destination_ty,
                    destination,
                    source,
                ));
            }
            "to_le_bytes" | "to_be_bytes" => {
                let Some(source_ty) = self.operand_type(&args[0].node, span) else {
                    return true;
                };
                let Some(source) = self.operand_register(&args[0].node, span) else {
                    return true;
                };
                let big_endian = name == "to_be_bytes";
                let source = match source_ty {
                    ValueType::U32 => source,
                    ValueType::F32 => {
                        let bits = self.allocate(ValueType::U32);
                        self.code.push(Instruction::reinterpret(
                            big_endian,
                            ValueType::U32,
                            bits,
                            source,
                        ));
                        bits
                    }
                    _ => {
                        self.error(span, "four-byte conversion requires a u32 or f32 scalar");
                        return true;
                    }
                };
                let Some(fields) = self.place_fields(destination, span).map(<[Register]>::to_vec)
                else {
                    return true;
                };
                if fields.len() != 4 {
                    self.error(span, "32-bit byte conversion requires a four-byte result");
                    return true;
                }
                let little = !big_endian;
                for (index, field) in fields.into_iter().enumerate() {
                    if !self.require_register_type(field, ValueType::U32, span) {
                        continue;
                    }
                    let byte = if little { index } else { 3 - index };
                    let shift = self.allocate(ValueType::U32);
                    self.code.push(Instruction::constant(ValueType::U32, shift, (byte * 8) as i64));
                    let shifted = self.allocate(ValueType::U32);
                    self.code.push(Instruction::binary(
                        Opcode::ShiftRight,
                        ValueType::U32,
                        shifted,
                        source,
                        shift,
                    ));
                    let mask = self.allocate(ValueType::U32);
                    self.code.push(Instruction::constant(ValueType::U32, mask, 0xff));
                    self.code.push(Instruction::binary(
                        Opcode::And,
                        ValueType::U32,
                        field,
                        shifted,
                        mask,
                    ));
                }
            }
            "from_le_bytes" | "from_be_bytes" => {
                let Some(place) = args[0].node.place() else {
                    self.error(span, "explicit byte conversion requires a concrete byte array");
                    return true;
                };
                let Some(fields) = self.place_fields(place, span).map(<[Register]>::to_vec) else {
                    return true;
                };
                if fields.len() != 4 {
                    self.error(span, "32-bit byte conversion requires a four-byte argument");
                    return true;
                }
                let Some(output) = self.place_register(destination, span) else {
                    return true;
                };
                let Some(output_ty) = self.place_register_type(output) else {
                    return true;
                };
                let little = name == "from_le_bytes";
                let carrier = match output_ty {
                    ValueType::U32 => output,
                    ValueType::F32 => self.allocate(ValueType::U32),
                    _ => {
                        self.error(span, "four-byte conversion requires a u32 or f32 scalar");
                        return true;
                    }
                };
                self.code.push(Instruction::constant(ValueType::U32, carrier, 0));
                for (index, field) in fields.into_iter().enumerate() {
                    let byte = if little { index } else { 3 - index };
                    let shift = self.allocate(ValueType::U32);
                    self.code.push(Instruction::constant(ValueType::U32, shift, (byte * 8) as i64));
                    let shifted = self.allocate(ValueType::U32);
                    self.code.push(Instruction::binary(
                        Opcode::ShiftLeft,
                        ValueType::U32,
                        shifted,
                        field,
                        shift,
                    ));
                    self.code.push(Instruction::binary(
                        Opcode::Or,
                        ValueType::U32,
                        carrier,
                        carrier,
                        shifted,
                    ));
                }
                if output_ty == ValueType::F32 {
                    self.code.push(Instruction::reinterpret(
                        !little,
                        ValueType::F32,
                        output,
                        carrier,
                    ));
                }
            }
            _ => {
                self.supported.set(false);
                self.tcx
                    .dcx()
                    .span_err(span, "native-endian byte conversion is forbidden for PolyASM");
                return true;
            }
        }
        if let Some(target) = target {
            self.push_branch(target);
        } else {
            self.code.push(Instruction::trap());
        }
        true
    }
}
