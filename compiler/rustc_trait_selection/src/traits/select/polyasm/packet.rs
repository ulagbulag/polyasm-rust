//! Original packet-window identities and branch-dominated fixed prefixes.
//!
//! Calls are interpreted with their actual argument facts. A raw address keeps
//! zero packet authority, whatever its type, its casts or another window's
//! checks. The ordinary MIR checker still checks each callee's non-memory
//! effects.

use std::collections::VecDeque;

use rustc_abi::{TagEncoding, VariantIdx, Variants};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::DefId;
use rustc_middle::mir::{
    AggregateKind, BasicBlock, BinOp, Body, BorrowKind, CastKind, Local, Location, Operand, Place,
    ProjectionElem, Rvalue, START_BLOCK, Statement, StatementKind, TerminatorKind, UnOp,
};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::consts::ConstExt;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_span::sym;

use super::{constant_integer, monomorphize};

pub(super) struct Analysis<'a, 'tcx> {
    pub(super) arguments: Option<&'a Arguments>,
    pub(super) body: &'a Body<'tcx>,
    pub(super) instance: Instance<'tcx>,
    pub(super) tcx: TyCtxt<'tcx>,
}

#[derive(Clone)]
pub(super) struct Arguments {
    values: Vec<Value>,
}

#[derive(Default)]
pub(super) struct Witness {
    assertions: FxHashSet<BasicBlock>,
    arguments: FxHashMap<BasicBlock, Arguments>,
    intrinsics: FxHashSet<BasicBlock>,
    memory: FxHashSet<Location>,
    rejected: FxHashSet<BasicBlock>,
}

impl Witness {
    pub(super) fn assertion(&self, block: BasicBlock) -> bool {
        self.assertions.contains(&block)
    }

    pub(super) fn arguments(&self, block: BasicBlock) -> Option<&Arguments> {
        self.arguments.get(&block)
    }

    pub(super) fn intrinsic(&self, block: BasicBlock) -> bool {
        self.intrinsics.contains(&block)
    }

    pub(super) fn memory(&self, location: Location) -> bool {
        self.memory.contains(&location)
    }

    pub(super) fn rejected(&self, block: BasicBlock) -> bool {
        self.rejected.contains(&block)
    }
}

#[derive(Clone, Eq, PartialEq)]
struct Invocation {
    block: BasicBlock,
    caller: DefId,
}

#[derive(Clone, Eq, PartialEq)]
struct Identity {
    argument: Local,
    calls: Vec<Invocation>,
}

#[derive(Clone, Eq, PartialEq)]
struct Pointer {
    context: Identity,
    producer: Vec<Invocation>,
    readable: u64,
}

#[derive(Clone, Eq, PartialEq)]
struct Prefix {
    length: u64,
    pointer: Pointer,
}

#[derive(Clone, Eq, PartialEq)]
enum Value {
    Array(Prefix),
    Context(Identity),
    End(Identity),
    Integer(u64),
    None,
    Optional(Prefix),
    OptionalDiscriminant { prefix: Prefix, some: u64 },
    Pointer(Pointer),
    Predicate { length: u64, pointer: Pointer, truth: bool },
    RangeOperands(Pointer),
    Some { prefix: Prefix, variant: VariantIdx },
    Slice(Prefix),
    StartOperands(Identity),
    EndOperands(Identity),
    Truth(bool),
    Unknown,
    Window(Pointer),
}

impl Value {
    fn meet(self, other: Self) -> Self {
        if self == other {
            return self;
        }
        match (self, other) {
            (Self::Some { prefix, .. }, Self::None)
            | (Self::None, Self::Some { prefix, .. })
            | (Self::Optional(prefix), Self::None)
            | (Self::None, Self::Optional(prefix)) => Self::Optional(prefix),
            (Self::Optional(prefix), Self::Some { prefix: other, .. })
            | (Self::Some { prefix: other, .. }, Self::Optional(prefix))
                if prefix == other =>
            {
                Self::Optional(prefix)
            }
            (Self::Pointer(mut pointer), Self::Pointer(other))
                if pointer.context == other.context && pointer.producer == other.producer =>
            {
                pointer.readable = pointer.readable.min(other.readable);
                Self::Pointer(pointer)
            }
            _ => Self::Unknown,
        }
    }
}

struct Evaluator<'a, 'tcx> {
    analysis: &'a Analysis<'a, 'tcx>,
    calls: &'a [Invocation],
    values: &'a mut [Value],
}

impl<'tcx> Evaluator<'_, 'tcx> {
    fn operand(&self, operand: &Operand<'tcx>) -> Value {
        if let Some((integer, _, false)) =
            constant_integer(self.analysis.tcx, self.analysis.instance, operand)
        {
            return u64::try_from(integer).map_or(Value::Unknown, Value::Integer);
        }
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.place(*place),
            Operand::Constant(constant) => {
                let tcx = self.analysis.tcx;
                let constant = monomorphize(tcx, self.analysis.instance, constant.const_);
                let ty = constant.ty();
                let ty::Adt(definition, args) = ty.kind() else { return Value::Unknown };
                if !tcx.is_diagnostic_item(sym::Option, definition.did()) {
                    return Value::Unknown;
                }
                let ty::Ref(_, pointee, rustc_hir::Mutability::Not) = *args.type_at(0).kind()
                else {
                    return Value::Unknown;
                };
                if self.array_length(pointee).is_none() {
                    return Value::Unknown;
                }
                let environment = ty::TypingEnv::fully_monomorphized();
                let Ok(layout) = tcx.layout_of(environment.as_query_input(ty)) else {
                    return Value::Unknown;
                };
                let Variants::Multiple {
                    tag,
                    tag_field,
                    tag_encoding: TagEncoding::Niche { niche_variants, niche_start, .. },
                    ..
                } = &layout.variants
                else {
                    return Value::Unknown;
                };
                let Some((none, _)) = definition
                    .variants()
                    .iter_enumerated()
                    .find(|(_, variant)| variant.fields.is_empty())
                else {
                    return Value::Unknown;
                };
                // The sysroot's None is a scalar niche constant at times.
                // Read its evaluated layout in place of assuming a zero tag or
                // a field order under randomized Rust layouts.
                if niche_variants.start == none
                    && niche_variants.last == none
                    && layout.fields.offset(tag_field.as_usize()).bytes() == 0
                    && tag.size(&tcx) == layout.size
                    && constant.try_eval_bits(tcx, environment) == Some(*niche_start)
                {
                    Value::None
                } else {
                    Value::Unknown
                }
            }
            _ => Value::Unknown,
        }
    }

    fn place(&self, place: Place<'tcx>) -> Value {
        let value = self.values[place.local.as_usize()].clone();
        if place.projection.is_empty() {
            return value;
        }
        match (value, place.projection.as_ref()) {
            (
                Value::Some { prefix, variant },
                [ProjectionElem::Downcast(_, selected), ProjectionElem::Field(field, _)],
            ) if variant == *selected && field.as_usize() == 0 => Value::Array(prefix),
            (Value::Window(pointer), [ProjectionElem::Field(field, _)]) => {
                let ty = monomorphize(
                    self.analysis.tcx,
                    self.analysis.instance,
                    self.analysis.body.local_decls[place.local].ty,
                );
                let ty::Adt(definition, _) = ty.kind() else { return Value::Unknown };
                match definition.non_enum_variant().fields[*field].name {
                    sym::start => Value::Pointer(pointer),
                    sym::end => Value::End(pointer.context),
                    _ => Value::Unknown,
                }
            }
            _ => Value::Unknown,
        }
    }

    fn array_length(&self, ty: Ty<'tcx>) -> Option<u64> {
        let ty = monomorphize(self.analysis.tcx, self.analysis.instance, ty);
        let ty::Array(element, length) = *ty.kind() else { return None };
        (element == self.analysis.tcx.types.u8)
            .then(|| length.try_to_target_usize(self.analysis.tcx))
            .flatten()
    }

    fn byte_read(&self, operand: &Operand<'tcx>) -> bool {
        let (Operand::Copy(place) | Operand::Move(place)) = operand else { return false };
        let (Value::Array(prefix) | Value::Slice(prefix)) = &self.values[place.local.as_usize()]
        else {
            return false;
        };
        match place.projection.as_ref() {
            [
                ProjectionElem::Deref,
                ProjectionElem::ConstantIndex { offset, from_end: false, .. },
            ] => *offset < prefix.length,
            [ProjectionElem::Deref, ProjectionElem::Index(index)] => {
                matches!(self.values[index.as_usize()], Value::Integer(offset) if offset < prefix.length)
            }
            _ => false,
        }
    }

    fn evaluate(&self, rvalue: &Rvalue<'tcx>) -> (Value, bool) {
        let tcx = self.analysis.tcx;
        match rvalue {
            Rvalue::Use(operand, _) => {
                let value = self.operand(operand);
                let proven = !matches!(value, Value::Unknown) || self.byte_read(operand);
                (value, proven)
            }
            Rvalue::Cast(CastKind::PtrToPtr, operand, target) => {
                let pointer = match self.operand(operand) {
                    Value::Pointer(pointer) | Value::Array(Prefix { pointer, .. }) => pointer,
                    _ => return (Value::Unknown, false),
                };
                let target = monomorphize(tcx, self.analysis.instance, *target);
                let ty::RawPtr(pointee, rustc_hir::Mutability::Not) = *target.kind() else {
                    return (Value::Unknown, false);
                };
                if pointee != tcx.types.u8 && self.array_length(pointee).is_none() {
                    return (Value::Unknown, false);
                }
                (Value::Pointer(pointer), true)
            }
            Rvalue::Cast(
                CastKind::PointerCoercion(PointerCoercion::Unsize, _),
                operand,
                target,
            ) => {
                let Value::Array(prefix) = self.operand(operand) else {
                    return (Value::Unknown, false);
                };
                let source = monomorphize(
                    tcx,
                    self.analysis.instance,
                    operand.ty(&self.analysis.body.local_decls, tcx),
                );
                let target = monomorphize(tcx, self.analysis.instance, *target);
                let (
                    ty::Ref(_, array, rustc_hir::Mutability::Not),
                    ty::Ref(_, slice, rustc_hir::Mutability::Not),
                ) = (*source.kind(), *target.kind())
                else {
                    return (Value::Unknown, false);
                };
                // Only this built-in coercion supplies the array's exact
                // length as slice metadata. The original packet pointer and
                // its dominating prefix check are already present.
                if self.array_length(array) != Some(prefix.length)
                    || *slice.kind() != ty::Slice(tcx.types.u8)
                    || prefix.length > prefix.pointer.readable
                {
                    return (Value::Unknown, false);
                }
                (Value::Slice(prefix), true)
            }
            Rvalue::Ref(_, BorrowKind::Shared, place) => {
                let [ProjectionElem::Deref] = place.projection.as_ref() else {
                    return (Value::Unknown, false);
                };
                match self.values[place.local.as_usize()].clone() {
                    Value::Pointer(pointer) => {
                        let ty = place.ty(&self.analysis.body.local_decls, tcx).ty;
                        let Some(length) = self.array_length(ty) else {
                            return (Value::Unknown, false);
                        };
                        if length <= pointer.readable {
                            (Value::Array(Prefix { length, pointer }), true)
                        } else {
                            (Value::Unknown, false)
                        }
                    }
                    Value::Array(prefix) => (Value::Array(prefix), true),
                    Value::Slice(prefix) => (Value::Slice(prefix), true),
                    Value::Context(context) => (Value::Context(context), true),
                    _ => (Value::Unknown, false),
                }
            }
            Rvalue::BinaryOp(operation, operands) => {
                let (Value::Integer(lhs), Value::Integer(rhs)) =
                    (self.operand(&operands.0), self.operand(&operands.1))
                else {
                    return (Value::Unknown, false);
                };
                let truth = match operation {
                    BinOp::Eq => lhs == rhs,
                    BinOp::Ne => lhs != rhs,
                    BinOp::Lt => lhs < rhs,
                    BinOp::Le => lhs <= rhs,
                    BinOp::Gt => lhs > rhs,
                    BinOp::Ge => lhs >= rhs,
                    _ => return (Value::Unknown, false),
                };
                (Value::Truth(truth), false)
            }
            Rvalue::UnaryOp(UnOp::Not, operand) => match self.operand(operand) {
                Value::Predicate { length, pointer, truth } => {
                    (Value::Predicate { length, pointer, truth: !truth }, false)
                }
                _ => (Value::Unknown, false),
            },
            Rvalue::Discriminant(place) => {
                let Value::Optional(prefix) = self.place(*place) else {
                    return (Value::Unknown, false);
                };
                (Value::OptionalDiscriminant { prefix, some: 1 }, false)
            }
            Rvalue::Aggregate(kind, operands) => {
                let AggregateKind::Adt(definition, variant, args, _, _) = &**kind else {
                    return (Value::Unknown, false);
                };
                let definition = tcx.adt_def(*definition);
                let field_value = |name: &str| {
                    definition
                        .variant(*variant)
                        .fields
                        .iter_enumerated()
                        .find(|(_, field)| field.name.as_str() == name)
                        .and_then(|(field, _)| operands.raw.get(field.as_usize()))
                        .map_or(Value::Unknown, |operand| self.operand(operand))
                };
                let value = if tcx
                    .is_diagnostic_item(sym::polyasm_packet_data_start_operands, definition.did())
                {
                    match field_value("context") {
                        Value::Context(context) => Value::StartOperands(context),
                        _ => Value::Unknown,
                    }
                } else if tcx
                    .is_diagnostic_item(sym::polyasm_packet_data_end_operands, definition.did())
                {
                    match field_value("context") {
                        Value::Context(context) => Value::EndOperands(context),
                        _ => Value::Unknown,
                    }
                } else if tcx.is_diagnostic_item(sym::polyasm_packet_window, definition.did()) {
                    match (field_value("start"), field_value("end")) {
                        (Value::Pointer(pointer), Value::End(context))
                            if pointer.context == context =>
                        {
                            Value::Window(pointer)
                        }
                        _ => Value::Unknown,
                    }
                } else if tcx
                    .is_diagnostic_item(sym::polyasm_packet_data_range_operands, definition.did())
                {
                    match field_value("window") {
                        Value::Window(pointer) => Value::RangeOperands(pointer),
                        _ => Value::Unknown,
                    }
                } else if tcx.is_diagnostic_item(sym::Option, definition.did()) {
                    match operands.raw.as_slice() {
                        [] => Value::None,
                        [operand] => match self.operand(operand) {
                            Value::Array(prefix) => Value::Some { prefix, variant: *variant },
                            _ => Value::Unknown,
                        },
                        _ => Value::Unknown,
                    }
                } else {
                    let _ = args;
                    Value::Unknown
                };
                let proven = !matches!(value, Value::Unknown);
                (value, proven)
            }
            _ => (Value::Unknown, false),
        }
    }

    fn assign(&mut self, statement: &Statement<'tcx>) -> bool {
        match &statement.kind {
            StatementKind::Assign(assignment) => {
                let (destination, rvalue) = &**assignment;
                if !destination.projection.is_empty() {
                    self.values.fill(Value::Unknown);
                    return false;
                }
                let (value, proven) = self.evaluate(rvalue);
                self.values[destination.local.as_usize()] = value;
                proven
            }
            StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                self.values[local.as_usize()] = Value::Unknown;
                false
            }
            StatementKind::SetDiscriminant { place, .. } => {
                self.values[place.local.as_usize()] = Value::Unknown;
                false
            }
            _ => false,
        }
    }

    fn call(&mut self, block: BasicBlock) -> Call {
        let tcx = self.analysis.tcx;
        let TerminatorKind::Call { func, args, destination, target: Some(_), .. } =
            &self.analysis.body.basic_blocks[block].terminator().kind
        else {
            return Call::default();
        };
        let arguments = Arguments {
            values: args.iter().map(|argument| self.operand(&argument.node)).collect(),
        };
        let Some((definition, generic_args)) = func.const_fn_def() else {
            self.values.fill(Value::Unknown);
            return Call::default();
        };
        let generic_args = monomorphize(tcx, self.analysis.instance, generic_args);
        let mut calls = self.calls.to_vec();
        calls.push(Invocation { block, caller: self.analysis.instance.def_id() });
        let intrinsic = tcx.intrinsic(definition).map(|intrinsic| intrinsic.name);
        let packet_intrinsic = matches!(
            intrinsic,
            Some(sym::packet_data_start | sym::packet_data_end | sym::packet_data_range)
        );
        let value = match (intrinsic, arguments.values.as_slice()) {
            (Some(sym::packet_data_start), [Value::StartOperands(context)]) => {
                Value::Pointer(Pointer { context: context.clone(), producer: calls, readable: 0 })
            }
            (Some(sym::packet_data_end), [Value::EndOperands(context)]) => {
                Value::End(context.clone())
            }
            (Some(sym::packet_data_range), [Value::RangeOperands(pointer)]) => {
                // Linux's kernel checker tracks packet ranges with a 16-bit
                // extent. A larger native instruction is legal PolyASM and
                // stays outside XDP `Always` on this machine.
                match generic_args.const_at(0).try_to_target_usize(tcx) {
                    Some(length @ 0..=65535) => {
                        Value::Predicate { length, pointer: pointer.clone(), truth: true }
                    }
                    _ => Value::Unknown,
                }
            }
            _ if intrinsic.is_none()
                && arguments
                    .values
                    .iter()
                    .any(|value| !matches!(value, Value::Unknown | Value::Integer(_)))
                && calls.len() <= 32
                && !self.calls.iter().any(|call| call.caller == definition) =>
            {
                if let Ok(Some(instance)) = Instance::try_resolve(
                    tcx,
                    ty::TypingEnv::fully_monomorphized(),
                    definition,
                    generic_args,
                ) && !tcx.is_foreign_item(instance.def_id())
                    && matches!(instance.def, ty::InstanceKind::Item(_))
                {
                    let body = tcx.instance_mir(instance.def);
                    evaluate(Analysis { arguments: Some(&arguments), body, instance, tcx }, &calls)
                        .returned
                } else {
                    Value::Unknown
                }
            }
            _ => Value::Unknown,
        };
        let proven = packet_intrinsic && !matches!(value, Value::Unknown);
        // Unknown calls mutate stored pointer identities at times. Closed
        // shared packet calls leave their borrowed context and its storage
        // unchanged.
        if !packet_intrinsic && matches!(value, Value::Unknown) {
            self.values.fill(Value::Unknown);
        }
        if destination.projection.is_empty() {
            self.values[destination.local.as_usize()] = value;
        } else {
            self.values.fill(Value::Unknown);
        }
        Call {
            arguments: Some(arguments),
            intrinsic: proven,
            rejected: packet_intrinsic && !proven,
        }
    }

    fn refine(&mut self, edge: Edge) {
        let Edge { source, destination } = edge;
        let TerminatorKind::SwitchInt { discr, targets } =
            &self.analysis.body.basic_blocks[source].terminator().kind
        else {
            return;
        };
        match self.operand(discr) {
            Value::Predicate { length, pointer, truth } => {
                if targets.target_for_value(u128::from(truth)) != destination
                    || targets.target_for_value(u128::from(!truth)) == destination
                {
                    return;
                }
                for value in self.values.iter_mut() {
                    let candidate = match value {
                        Value::Pointer(candidate) | Value::Window(candidate) => candidate,
                        _ => continue,
                    };
                    if candidate.context == pointer.context
                        && candidate.producer == pointer.producer
                    {
                        candidate.readable = candidate.readable.max(length);
                    }
                }
            }
            Value::OptionalDiscriminant { prefix, some } => {
                if targets.target_for_value(u128::from(some)) != destination
                    || targets.target_for_value(0) == destination
                {
                    return;
                }
                for value in self.values.iter_mut() {
                    if matches!(value, Value::Optional(candidate) if *candidate == prefix) {
                        *value = Value::Some {
                            prefix: prefix.clone(),
                            variant: VariantIdx::from_usize(1),
                        };
                    }
                }
            }
            _ => {}
        }
    }
}

struct Edge {
    destination: BasicBlock,
    source: BasicBlock,
}

#[derive(Default)]
struct Call {
    arguments: Option<Arguments>,
    intrinsic: bool,
    rejected: bool,
}

struct Evaluation {
    witness: Witness,
    returned: Value,
}

fn context_type<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    tcx.get_diagnostic_item(sym::polyasm_packet_context).is_some_and(|definition| {
        tcx.type_of(definition).instantiate_identity().skip_norm_wip() == ty
    })
}

fn initial(analysis: &Analysis<'_, '_>, calls: &[Invocation]) -> Vec<Value> {
    let mut values = vec![Value::Unknown; analysis.body.local_decls.len()];
    for (index, argument) in analysis.body.args_iter().enumerate() {
        if let Some(arguments) = analysis.arguments {
            values[argument.as_usize()] =
                arguments.values.get(index).cloned().unwrap_or(Value::Unknown);
            continue;
        }
        let ty =
            monomorphize(analysis.tcx, analysis.instance, analysis.body.local_decls[argument].ty);
        let identity = Identity { argument, calls: calls.to_vec() };
        if let ty::Ref(_, pointee, rustc_hir::Mutability::Not) = *ty.kind()
            && context_type(analysis.tcx, pointee)
        {
            values[argument.as_usize()] = Value::Context(identity);
        } else if let ty::Adt(definition, _) = ty.kind()
            && analysis.tcx.is_diagnostic_item(sym::polyasm_packet_window, definition.did())
        {
            values[argument.as_usize()] =
                Value::Window(Pointer { context: identity, producer: calls.to_vec(), readable: 0 });
        }
    }
    values
}

fn evaluate(analysis: Analysis<'_, '_>, calls: &[Invocation]) -> Evaluation {
    let mut entries = vec![None; analysis.body.basic_blocks.len()];
    entries[START_BLOCK.as_usize()] = Some(initial(&analysis, calls));
    let mut pending = VecDeque::from([START_BLOCK]);
    while let Some(block) = pending.pop_front() {
        let Some(mut values) = entries[block.as_usize()].clone() else { continue };
        let data = &analysis.body.basic_blocks[block];
        let mut evaluator = Evaluator { analysis: &analysis, calls, values: &mut values };
        for statement in &data.statements {
            evaluator.assign(statement);
        }
        evaluator.call(block);
        if matches!(
            data.terminator().kind,
            TerminatorKind::Drop { .. } | TerminatorKind::InlineAsm { .. }
        ) {
            values.fill(Value::Unknown);
        }
        for successor in data.terminator().successors() {
            let mut incoming = values.clone();
            Evaluator { analysis: &analysis, calls, values: &mut incoming }
                .refine(Edge { source: block, destination: successor });
            let entry = &mut entries[successor.as_usize()];
            let changed = if let Some(entry) = entry {
                let mut changed = false;
                for (current, incoming) in entry.iter_mut().zip(incoming) {
                    let merged = current.clone().meet(incoming);
                    changed |= *current != merged;
                    *current = merged;
                }
                changed
            } else {
                *entry = Some(incoming);
                true
            };
            if changed {
                pending.push_back(successor);
            }
        }
    }
    let mut witness = Witness::default();
    let mut returned: Option<Value> = None;
    for (block, data) in analysis.body.basic_blocks.iter_enumerated() {
        let Some(mut values) = entries[block.as_usize()].clone() else { continue };
        let mut evaluator = Evaluator { analysis: &analysis, calls, values: &mut values };
        for (statement_index, statement) in data.statements.iter().enumerate() {
            if evaluator.assign(statement) {
                witness.memory.insert(Location { block, statement_index });
            }
        }
        if let TerminatorKind::Assert { cond, expected, .. } = &data.terminator().kind
            && evaluator.operand(cond) == Value::Truth(*expected)
        {
            witness.assertions.insert(block);
        }
        let call = evaluator.call(block);
        if let Some(arguments) = call.arguments {
            witness.arguments.insert(block, arguments);
        }
        if call.intrinsic {
            witness.intrinsics.insert(block);
        }
        if call.rejected {
            witness.rejected.insert(block);
        }
        if matches!(data.terminator().kind, TerminatorKind::Return) {
            let result = values[rustc_middle::mir::RETURN_PLACE.as_usize()].clone();
            returned = Some(match returned {
                Some(previous) => previous.meet(result),
                None => result,
            });
        }
    }
    Evaluation { witness, returned: returned.unwrap_or(Value::Unknown) }
}

pub(super) fn analyze(analysis: Analysis<'_, '_>) -> Witness {
    evaluate(analysis, &[]).witness
}
