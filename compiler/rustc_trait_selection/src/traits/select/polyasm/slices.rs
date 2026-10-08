//! Relational bounds facts for byte slices and their checked subregions.
//!
//! A Rust slice parameter establishes a live byte allocation. Its dynamic
//! length stays apart from fixed bounds: only a dominating length check grants
//! the corresponding minimum extent. Joins retain facts common to every
//! predecessor, and raw slice construction fits that extent.

use std::collections::VecDeque;

use rustc_abi::{TagEncoding, VariantIdx, Variants};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::DefId;
use rustc_middle::mir::{
    AggregateKind, BasicBlock, BinOp, Body, CastKind, Local, Location, Operand, Place,
    ProjectionElem, Rvalue, START_BLOCK, Statement, StatementKind, TerminatorKind, UnOp,
};
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
    arguments: FxHashMap<BasicBlock, Arguments>,
    assertions: FxHashSet<BasicBlock>,
    memory: FxHashSet<Location>,
}

impl Witness {
    pub(super) fn arguments(&self, block: BasicBlock) -> Option<&Arguments> {
        self.arguments.get(&block)
    }

    pub(super) fn assertion(&self, block: BasicBlock) -> bool {
        self.assertions.contains(&block)
    }

    pub(super) fn memory(&self, location: Location) -> bool {
        self.memory.contains(&location)
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Representation {
    Pointer,
    Slice,
}

#[derive(Clone, Eq, PartialEq)]
struct Invocation {
    block: BasicBlock,
    caller: DefId,
}

#[derive(Clone, Eq, PartialEq)]
struct Origin {
    argument: Local,
    calls: Vec<Invocation>,
}

#[derive(Clone, Eq, PartialEq)]
struct Region {
    exact_length: Option<u64>,
    minimum_length: u64,
    offset: u64,
    origin: Origin,
    representation: Representation,
    writable: bool,
}

#[derive(Clone, Eq, PartialEq)]
enum Value {
    Bound { minimum_length: u64, origin: Origin },
    Fields(Vec<Value>),
    None,
    OptionalRegion { region: Region, some: VariantIdx },
    OptionalDiscriminant { region: Region, some: VariantIdx },
    SomeInteger { integer: u64, variant: VariantIdx },
    Integer(u64),
    Length(Region),
    Region(Region),
    SomeRegion { region: Region, variant: VariantIdx },
    Truth(bool),
    Unknown,
}

impl Value {
    fn meet(self, incoming: Self) -> Self {
        if self == incoming {
            return self;
        }
        match (self, incoming) {
            (Self::SomeRegion { region, variant: some }, Self::None)
            | (Self::None, Self::SomeRegion { region, variant: some })
            | (Self::OptionalRegion { region, some }, Self::None)
            | (Self::None, Self::OptionalRegion { region, some }) => {
                Self::OptionalRegion { region, some }
            }
            (
                Self::SomeRegion { region, variant: some },
                Self::OptionalRegion { region: other, some: selected },
            )
            | (
                Self::OptionalRegion { region: other, some: selected },
                Self::SomeRegion { region, variant: some },
            ) if region == other && some == selected => Self::OptionalRegion { region, some },
            (Self::Region(current), Self::Region(other))
                if current.origin == other.origin
                    && current.offset == other.offset
                    && current.representation == other.representation
                    && current.writable == other.writable =>
            {
                Self::Region(Region {
                    exact_length: (current.exact_length == other.exact_length)
                        .then_some(current.exact_length)
                        .flatten(),
                    minimum_length: current.minimum_length.min(other.minimum_length),
                    ..current
                })
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
                let ty::Adt(definition, _) = ty.kind() else { return Value::Unknown };
                if !tcx.is_diagnostic_item(sym::Option, definition.did()) {
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
                if niche_variants.start != none || niche_variants.last != none {
                    return Value::Unknown;
                }
                let tag_value = constant.try_eval_bits(tcx, environment).or_else(|| {
                    let rustc_middle::mir::ConstValue::Indirect { alloc_id, offset } =
                        constant.eval(tcx, environment, rustc_span::DUMMY_SP).ok()?
                    else {
                        return None;
                    };
                    let allocation = tcx.global_alloc(alloc_id).unwrap_memory();
                    let bytes = allocation
                        .inner()
                        .get_bytes_strip_provenance(
                            &tcx,
                            rustc_middle::mir::interpret::AllocRange {
                                start: offset + layout.fields.offset(tag_field.as_usize()),
                                size: tag.size(&tcx),
                            },
                        )
                        .ok()?;
                    if bytes.len() > 16 {
                        return None;
                    }
                    let mut value = [0; 16];
                    match tcx.data_layout.endian {
                        rustc_abi::Endian::Little => {
                            value[..bytes.len()].copy_from_slice(bytes);
                            Some(u128::from_le_bytes(value))
                        }
                        rustc_abi::Endian::Big => {
                            value[16 - bytes.len()..].copy_from_slice(bytes);
                            Some(u128::from_be_bytes(value))
                        }
                    }
                });
                if tag_value == Some(*niche_start) { Value::None } else { Value::Unknown }
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
                Value::SomeRegion { region, variant },
                [ProjectionElem::Downcast(_, selected), ProjectionElem::Field(field, _)],
            ) if variant == *selected && field.as_usize() == 0 => Value::Region(region),
            (
                Value::SomeInteger { integer, variant },
                [ProjectionElem::Downcast(_, selected), ProjectionElem::Field(field, _)],
            ) if variant == *selected && field.as_usize() == 0 => Value::Integer(integer),
            (Value::Fields(fields), [ProjectionElem::Field(field, _)]) => {
                fields.get(field.as_usize()).cloned().unwrap_or(Value::Unknown)
            }
            _ => Value::Unknown,
        }
    }

    fn byte_slice(&self, ty: Ty<'tcx>) -> bool {
        let ty = monomorphize(self.analysis.tcx, self.analysis.instance, ty);
        matches!(ty.kind(), ty::Slice(element) if *element == self.analysis.tcx.types.u8)
    }

    fn region_address(&self, place: Place<'tcx>) -> Option<Region> {
        let Value::Region(region) = self.values[place.local.as_usize()].clone() else {
            return None;
        };
        match place.projection.as_ref() {
            [ProjectionElem::Deref] if region.representation == Representation::Slice => {
                Some(region)
            }
            _ => None,
        }
    }

    fn byte_read(&self, operand: &Operand<'tcx>) -> bool {
        let (Operand::Copy(place) | Operand::Move(place)) = operand else {
            return false;
        };
        let Value::Region(region) = self.values[place.local.as_usize()].clone() else {
            return false;
        };
        let offset = match place.projection.as_ref() {
            [
                ProjectionElem::Deref,
                ProjectionElem::ConstantIndex { offset, from_end: false, .. },
            ] => *offset,
            [ProjectionElem::Deref, ProjectionElem::Index(index)] => {
                let Value::Integer(offset) = self.values[index.as_usize()] else { return false };
                offset
            }
            _ => return false,
        };
        region.representation == Representation::Slice && offset < region.minimum_length
    }

    fn evaluate(&self, value: &Rvalue<'tcx>) -> (Value, bool) {
        let tcx = self.analysis.tcx;
        let instance = self.analysis.instance;
        match value {
            Rvalue::Use(operand, _) => {
                let next = self.operand(operand);
                (next.clone(), matches!(next, Value::Region(_)) || self.byte_read(operand))
            }
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => self
                .region_address(*place)
                .map_or((Value::Unknown, false), |region| (Value::Region(region), true)),
            Rvalue::Cast(CastKind::PtrToPtr, operand, target) => {
                let Value::Region(mut region) = self.operand(operand) else {
                    return (Value::Unknown, false);
                };
                let target = monomorphize(tcx, instance, *target);
                let ty::RawPtr(pointee, mutability) = *target.kind() else {
                    return (Value::Unknown, false);
                };
                if pointee != tcx.types.u8
                    || (mutability == rustc_hir::Mutability::Mut && !region.writable)
                {
                    return (Value::Unknown, false);
                }
                region.representation = Representation::Pointer;
                (Value::Region(region), true)
            }
            Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => match self.operand(operand) {
                Value::Region(region) if region.representation == Representation::Slice => {
                    (Value::Length(region), true)
                }
                _ => (Value::Unknown, false),
            },
            Rvalue::Aggregate(kind, operands) => match &**kind {
                AggregateKind::RawPtr(pointee, mutability) if self.byte_slice(*pointee) => {
                    let [pointer, length] = operands.raw.as_slice() else {
                        return (Value::Unknown, false);
                    };
                    let (Value::Region(mut region), Value::Integer(length)) =
                        (self.operand(pointer), self.operand(length))
                    else {
                        return (Value::Unknown, false);
                    };
                    if region.representation != Representation::Pointer
                        || length > region.minimum_length
                        || (*mutability == rustc_hir::Mutability::Mut && !region.writable)
                    {
                        return (Value::Unknown, false);
                    }
                    region.exact_length = Some(length);
                    region.minimum_length = length;
                    region.representation = Representation::Slice;
                    (Value::Region(region), true)
                }
                AggregateKind::Adt(definition, variant, _, _, _)
                    if tcx.is_diagnostic_item(sym::Option, *definition) && operands.len() == 1 =>
                {
                    match self.operand(&operands.raw[0]) {
                        Value::Region(region) => {
                            (Value::SomeRegion { region, variant: *variant }, true)
                        }
                        Value::Integer(integer) => {
                            (Value::SomeInteger { integer, variant: *variant }, false)
                        }
                        _ => (Value::Unknown, false),
                    }
                }
                AggregateKind::Adt(definition, _, _, _, _)
                    if !tcx.adt_def(*definition).is_enum() =>
                {
                    let fields: Vec<_> =
                        operands.iter().map(|operand| self.operand(operand)).collect();
                    (Value::Fields(fields), false)
                }
                _ => (Value::Unknown, false),
            },
            Rvalue::Discriminant(place) => match self.place(*place) {
                Value::OptionalRegion { region, some } => {
                    (Value::OptionalDiscriminant { region, some }, false)
                }
                _ => (Value::Unknown, false),
            },
            Rvalue::BinaryOp(operation, operands) => {
                let lhs = self.operand(&operands.0);
                let rhs = self.operand(&operands.1);
                if let (BinOp::Offset, Value::Region(mut region), Value::Integer(offset)) =
                    (operation, lhs.clone(), rhs.clone())
                {
                    if region.representation != Representation::Pointer
                        || offset > region.minimum_length
                    {
                        return (Value::Unknown, false);
                    }
                    let Some(next_offset) = region.offset.checked_add(offset) else {
                        return (Value::Unknown, false);
                    };
                    region.offset = next_offset;
                    region.minimum_length -= offset;
                    region.exact_length =
                        region.exact_length.and_then(|length| length.checked_sub(offset));
                    return (Value::Region(region), true);
                }
                let integer = |value: &Value| match value {
                    Value::Integer(value) => Some(*value),
                    Value::Length(region) => region.exact_length,
                    _ => None,
                };
                if let (Some(lhs), Some(rhs)) = (integer(&lhs), integer(&rhs)) {
                    if matches!(operation, BinOp::Sub | BinOp::SubUnchecked) {
                        return (
                            lhs.checked_sub(rhs).map_or(Value::Unknown, Value::Integer),
                            false,
                        );
                    }
                    let truth = match operation {
                        BinOp::Eq => lhs == rhs,
                        BinOp::Ne => lhs != rhs,
                        BinOp::Lt => lhs < rhs,
                        BinOp::Le => lhs <= rhs,
                        BinOp::Gt => lhs > rhs,
                        BinOp::Ge => lhs >= rhs,
                        _ => return (Value::Unknown, false),
                    };
                    return (Value::Truth(truth), false);
                }
                let bound = match (operation, lhs, rhs) {
                    (BinOp::Le, Value::Integer(minimum_length), Value::Length(region))
                    | (BinOp::Ge, Value::Length(region), Value::Integer(minimum_length)) => {
                        Some((minimum_length, region))
                    }
                    (BinOp::Lt, Value::Integer(lower), Value::Length(region))
                    | (BinOp::Gt, Value::Length(region), Value::Integer(lower)) => {
                        lower.checked_add(1).map(|minimum| (minimum, region))
                    }
                    _ => None,
                };
                let next = bound.map_or(Value::Unknown, |(minimum_length, region)| {
                    // A bound on a truncated subslice leaves every other slice
                    // as it is. The whole argument allocation is the only origin
                    // whose open length grants additional readable bytes.
                    if region.offset == 0 && region.exact_length.is_none() {
                        Value::Bound { minimum_length, origin: region.origin }
                    } else {
                        Value::Unknown
                    }
                });
                (next, false)
            }
            _ => (Value::Unknown, false),
        }
    }

    fn assign(&mut self, statement: &Statement<'tcx>) -> bool {
        match &statement.kind {
            StatementKind::Assign(assignment) => {
                let (destination, rvalue) = &**assignment;
                if !destination.projection.is_empty() {
                    // Aliasing writes overwrite a reference or a stored length
                    // at times, so this drops every slice fact across such a store.
                    self.values.fill(Value::Unknown);
                    return false;
                }
                let (next, checked) = self.evaluate(rvalue);
                self.values[destination.local.as_usize()] = next;
                checked
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

    fn call(&mut self, block: BasicBlock) -> Option<Arguments> {
        let tcx = self.analysis.tcx;
        let TerminatorKind::Call { func, args, destination, target: Some(_), .. } =
            &self.analysis.body.basic_blocks[block].terminator().kind
        else {
            return None;
        };
        let arguments = Arguments {
            values: args.iter().map(|argument| self.operand(&argument.node)).collect(),
        };
        let mut calls = self.calls.to_vec();
        calls.push(Invocation { block, caller: self.analysis.instance.def_id() });
        let value = if let Some((definition, generic_args)) = func.const_fn_def()
            && arguments.values.iter().any(|value| {
                matches!(
                    value,
                    Value::Region(_)
                        | Value::Fields(_)
                        | Value::SomeRegion { .. }
                        | Value::OptionalRegion { .. }
                )
            })
            && calls.len() <= 32
            && !self.calls.iter().any(|call| call.caller == definition)
            && tcx.intrinsic(definition).is_none()
            && let Ok(Some(instance)) = Instance::try_resolve(
                tcx,
                ty::TypingEnv::fully_monomorphized(),
                definition,
                monomorphize(tcx, self.analysis.instance, generic_args),
            )
            && !tcx.is_foreign_item(instance.def_id())
            && matches!(instance.def, ty::InstanceKind::Item(_))
        {
            evaluate(
                Analysis {
                    arguments: Some(&arguments),
                    body: tcx.instance_mir(instance.def),
                    instance,
                    tcx,
                },
                &calls,
            )
            .returned
        } else {
            Value::Unknown
        };
        if matches!(value, Value::Unknown) {
            self.values.fill(Value::Unknown);
        }
        if destination.projection.is_empty() {
            self.values[destination.local.as_usize()] = value;
        } else {
            self.values.fill(Value::Unknown);
        }
        Some(arguments)
    }

    fn refine(&mut self, edge: Edge) {
        let TerminatorKind::SwitchInt { discr, targets } =
            &self.analysis.body.basic_blocks[edge.source].terminator().kind
        else {
            return;
        };
        let (minimum_length, origin) = match self.operand(discr) {
            Value::Bound { minimum_length, origin } => (minimum_length, origin),
            Value::OptionalDiscriminant { region, some } => {
                if targets.target_for_value(some.as_u32() as u128) != edge.destination
                    || targets.target_for_value(0) == edge.destination
                {
                    return;
                }
                for value in self.values.iter_mut() {
                    if matches!(value, Value::OptionalRegion { region: candidate, some: variant } if *candidate == region && *variant == some)
                    {
                        *value = Value::SomeRegion { region: region.clone(), variant: some };
                    }
                }
                return;
            }
            _ => return,
        };
        if targets.target_for_value(1) != edge.destination
            || targets.target_for_value(0) == edge.destination
        {
            return;
        }
        for value in self.values.iter_mut() {
            if let Value::Region(region) = value
                && region.origin == origin
                && region.offset == 0
                && region.exact_length.is_none()
            {
                region.minimum_length = region.minimum_length.max(minimum_length);
            }
        }
    }
}

struct Edge {
    destination: BasicBlock,
    source: BasicBlock,
}

struct Evaluation {
    witness: Witness,
    returned: Value,
}

fn evaluate(analysis: Analysis<'_, '_>, calls: &[Invocation]) -> Evaluation {
    let mut initial = vec![Value::Unknown; analysis.body.local_decls.len()];
    for (index, argument) in analysis.body.args_iter().enumerate() {
        if let Some(arguments) = analysis.arguments
            && let Some(value) = arguments.values.get(index)
            && !matches!(value, Value::Unknown)
        {
            initial[argument.as_usize()] = value.clone();
            continue;
        }
        let ty =
            monomorphize(analysis.tcx, analysis.instance, analysis.body.local_decls[argument].ty);
        if let ty::Ref(_, pointee, mutability) = *ty.kind()
            && let ty::Slice(element) = *pointee.kind()
            && element == analysis.tcx.types.u8
        {
            initial[argument.as_usize()] = Value::Region(Region {
                exact_length: None,
                minimum_length: 0,
                offset: 0,
                origin: Origin { argument, calls: calls.to_vec() },
                representation: Representation::Slice,
                writable: mutability == rustc_hir::Mutability::Mut,
            });
        }
    }
    let mut entries = vec![None; analysis.body.basic_blocks.len()];
    entries[START_BLOCK.as_usize()] = Some(initial);
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
                .refine(Edge { destination: successor, source: block });
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
        if let Some(arguments) = evaluator.call(block) {
            witness.arguments.insert(block, arguments);
        }
        if matches!(data.terminator().kind, TerminatorKind::Return) {
            let value = values[rustc_middle::mir::RETURN_PLACE.as_usize()].clone();
            returned = Some(match returned {
                Some(previous) => previous.meet(value),
                None => value,
            });
        }
    }
    Evaluation { witness, returned: returned.unwrap_or(Value::Unknown) }
}

pub(super) fn analyze(analysis: Analysis<'_, '_>) -> Witness {
    evaluate(analysis, &[]).witness
}
