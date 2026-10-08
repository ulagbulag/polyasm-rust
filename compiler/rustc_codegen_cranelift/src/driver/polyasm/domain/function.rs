//! Per-function MIR lowering orchestration and shared lowering state.

use std::cell::Cell;
use std::collections::BTreeMap;

use polyasm_format::ir::Request;
use rustc_codegen_ssa::polyasm::PolyasmStatementSelection;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::{
    BasicBlock, Body, Location, ProjectionElem, RETURN_PLACE, StatementKind, TerminatorKind,
    VarDebugInfoContents,
};
use rustc_middle::ty::consts::ConstExt;
use rustc_middle::ty::{self, EarlyBinder, Instance, Ty, TyCtxt};
use rustc_span::{Span, StableSourceFileId, sym};

use super::super::interchange::{CallableKind, FunctionSignature, PortableImage, callable_kind};
use super::capability::{CAP_POINTER_ACCELERATORS, CAPABILITY_MASK, derive_always_scopes};
use super::model::{
    AlwaysScope, AlwaysScopeKind, CallRequest, ClosureEnvironment, DebugLocation, DebugProjection,
    DebugRange, DebugScope, DebugVariable, Function, Hint, HintKind, Instruction,
    OFFLOAD_CALL_HINT_TAG, Register, ValueType,
};
use super::signature::{closure_environment, closure_environment_parameter};
use super::source::debug_position;

mod control;
mod expression;
mod place;

pub(super) struct FunctionEmitter<'a, 'tcx> {
    body: &'tcx Body<'tcx>,
    calls: FxHashMap<BasicBlock, (Instance<'tcx>, u32)>,
    capability_exclusions: &'a mut BTreeMap<String, u32>,
    code: Vec<Instruction>,
    indices: &'a FxHashMap<String, u32>,
    image: &'a PortableImage,
    instance: Instance<'tcx>,
    labels: Vec<Option<u32>>,
    locals: Vec<Option<LocalRegisters>>,
    mir_pcs: Vec<Vec<u32>>,
    offload_markers: Vec<OffloadMarker<'tcx>>,
    params: Vec<ValueType>,
    patches: Vec<(usize, BasicBlock)>,
    registers: Vec<ValueType>,
    result: Option<ValueType>,
    source: Option<u32>,
    source_indices: &'a BTreeMap<StableSourceFileId, u32>,
    source_ranges: Vec<EmittedSourceRange>,
    source_calls: FxHashMap<BasicBlock, Instance<'tcx>>,
    static_callables: FxHashMap<usize, StaticCallableWitness<'tcx>>,
    statement_warrants: Vec<StatementWarrant>,
    capability_ceiling: Cell<u32>,
    callable_kind: CallableKind,
    supported: Cell<bool>,
    tcx: TyCtxt<'tcx>,
    closure_environment: ClosureEnvironment,
}

#[derive(Clone)]
enum LocalShape {
    Opaque,
    Pointer { address: ValueType, pointee: ValueType },
    Scalar(ValueType),
    Tuple(Vec<ValueType>),
    Unit,
}

#[derive(Clone)]
struct LocalRegisters {
    alias: Option<PointerProvenance>,
    fields: Vec<Register>,
    shape: LocalShape,
}

#[derive(Copy, Clone)]
struct SimplePlace {
    dereference: bool,
    field: Option<usize>,
    local: usize,
}

#[derive(Copy, Clone)]
struct StaticAddress {
    offset: u32,
    symbol: u32,
}

#[derive(Copy, Clone)]
enum PointerProvenance {
    Place(SimplePlace),
    Static(StaticAddress),
}

#[derive(Copy, Clone)]
struct StaticCallableWitness<'tcx> {
    architecture: Ty<'tcx>,
    callable: Instance<'tcx>,
    clock_hz: u64,
    cycles: u64,
}

#[derive(Copy, Clone)]
struct EmittedSourceRange {
    end: u32,
    location: Location,
    start: u32,
}

struct StatementWarrant {
    capability: u32,
    selection: PolyasmStatementSelection,
}

#[derive(Copy, Clone)]
struct OffloadMarker<'tcx> {
    callable: Instance<'tcx>,
    next: BasicBlock,
    request: Request,
}

impl<'a, 'tcx> FunctionEmitter<'a, 'tcx> {
    pub(super) fn new(
        tcx: TyCtxt<'tcx>,
        instance: Instance<'tcx>,
        indices: &'a FxHashMap<String, u32>,
        image: &'a PortableImage,
        capability_exclusions: &'a mut BTreeMap<String, u32>,
        source_indices: &'a BTreeMap<StableSourceFileId, u32>,
        interface: &FunctionSignature,
    ) -> Self {
        let body = tcx.instance_mir(instance.def);
        let callable_kind = callable_kind(tcx.def_kind(instance.def_id()));
        if callable_kind == CallableKind::Unknown {
            tcx.dcx().fatal(format!(
                "semantic MonoItem `{}` has a non-callable definition kind",
                interface.name
            ));
        }
        if interface.callable_kind != CallableKind::Unknown
            && interface.callable_kind != callable_kind
        {
            tcx.dcx().fatal(format!(
                "semantic callable kind for `{}` disagrees with its interchange interface",
                interface.name
            ));
        }
        let source = debug_position(tcx, source_indices, body.span, body.span).map(|value| value.0);
        let mut emitter = Self {
            body,
            calls: FxHashMap::default(),
            capability_exclusions,
            code: Vec::new(),
            indices,
            image,
            instance,
            labels: vec![None; body.basic_blocks.len()],
            locals: (0..body.local_decls.len()).map(|_| None).collect(),
            mir_pcs: body
                .basic_blocks
                .iter()
                .map(|block| vec![0; block.statements.len() + 1])
                .collect(),
            offload_markers: Vec::new(),
            params: Vec::new(),
            patches: Vec::new(),
            registers: Vec::new(),
            result: None,
            source,
            source_indices,
            source_ranges: Vec::new(),
            source_calls: FxHashMap::default(),
            static_callables: FxHashMap::default(),
            statement_warrants: Vec::new(),
            capability_ceiling: Cell::new(CAPABILITY_MASK),
            callable_kind,
            supported: Cell::new(true),
            tcx,
            closure_environment: ClosureEnvironment::None,
        };
        let requested_closure_environment = closure_environment(interface);
        let unit_environment_closure = callable_kind == CallableKind::Closure
            && instance.args.as_closure().tupled_upvars_ty().is_unit();
        if unit_environment_closure && body.arg_count == 1 {
            match requested_closure_environment {
                ClosureEnvironment::Elided => {
                    emitter.closure_environment = ClosureEnvironment::Elided;
                }
                ClosureEnvironment::Retained => {
                    if let Some(environment_type) = closure_environment_parameter(interface) {
                        emitter.params.push(environment_type);
                        let environment = emitter.allocate(environment_type);
                        debug_assert_eq!(environment.number(), 0);
                        emitter.closure_environment = ClosureEnvironment::Retained;
                    } else {
                        emitter.error(
                            body.span,
                            "PolyASM retained unit environment has no target-width scalar ABI",
                        );
                    }
                }
                ClosureEnvironment::None => emitter.error(
                    body.span,
                    "PolyASM zero-argument closure has an unsupported interchange environment ABI",
                ),
            }
        }
        for (argument, local) in body.args_iter().enumerate() {
            if unit_environment_closure && argument == 0 {
                // The MIR environment is fieldless. Its interchange ABI is chosen
                // above from the actual interface: zero parameters, or one
                // reserved target-width register outside this local's reach.
                emitter.assign_local(local.as_usize(), LocalShape::Unit);
                continue;
            }
            let shape = emitter
                .local_shape(body.local_decls[local].ty, body.local_decls[local].source_info.span);
            let ty = match shape {
                LocalShape::Scalar(ty) | LocalShape::Pointer { address: ty, .. } => ty,
                _ => {
                    emitter.error(
                        body.local_decls[local].source_info.span,
                        "PolyASM function parameters must be scalar or thin pointers",
                    );
                    continue;
                }
            };
            emitter.params.push(ty);
            emitter.assign_local(local.as_usize(), shape);
        }
        let return_span = body.local_decls[RETURN_PLACE].source_info.span;
        let return_shape = emitter.local_shape(body.return_ty(), return_span);
        emitter.result = match return_shape {
            LocalShape::Scalar(ty) => Some(ty),
            LocalShape::Unit => None,
            _ => {
                emitter.error(return_span, "PolyASM function results must be scalar or unit");
                None
            }
        };
        emitter.assign_local(0, return_shape);
        for local in body.vars_and_temps_iter() {
            let declaration = &body.local_decls[local];
            let shape = emitter.local_shape(declaration.ty, declaration.source_info.span);
            emitter.assign_local(local.as_usize(), shape);
        }
        emitter
    }

    pub(super) fn emit(mut self) -> (Option<Function>, Vec<CallRequest>) {
        for (block, data) in self.body.basic_blocks.iter_enumerated() {
            self.labels[block.as_usize()] = Some(self.code.len() as u32);
            if data.is_cleanup {
                self.code.push(Instruction::trap());
                continue;
            }
            for (statement_index, statement) in data.statements.iter().enumerate() {
                let location = Location { block, statement_index };
                let start = self.code.len();
                self.mir_pcs[block.as_usize()][statement_index] = start as u32;
                self.emit_statement(&statement.kind, statement.source_info.span);
                self.record_emitted_range(location, start);
            }
            let location = Location { block, statement_index: data.statements.len() };
            let start = self.code.len();
            self.mir_pcs[block.as_usize()][data.statements.len()] = start as u32;
            self.emit_terminator(
                &data.terminator().kind,
                location,
                data.terminator().source_info.span,
            );
            if !is_warrant_only_compiler_marker_terminator(self.tcx, &data.terminator().kind) {
                self.record_emitted_range(location, start);
            }
        }
        for (instruction, target) in &self.patches {
            let Some(target) = self.labels[target.as_usize()] else {
                self.error(self.body.span, "PolyASM branch target is missing");
                continue;
            };
            self.code[*instruction].immediate = i64::from(target);
        }
        let mut always_scopes =
            derive_always_scopes(&self.code, self.callable_kind == CallableKind::Closure);
        if let Some(root) = always_scopes.first_mut() {
            root.capabilities &= self.capability_ceiling.get();
        }
        always_scopes.extend(self.statement_scopes());
        let locations = self.debug_locations();
        let marked = self.marked_calls();
        let hints = self.offload_hints(&marked);
        let call_requests = self.call_requests(&marked);
        let debug_scopes = self.debug_scopes();
        let debug_variables = self.debug_variables();
        let source = self.source.or_else(|| locations.first().map(|location| location.source));
        let root_kind = match self.callable_kind {
            CallableKind::Closure => AlwaysScopeKind::Closure,
            CallableKind::Function => AlwaysScopeKind::Function,
            CallableKind::Unknown => unreachable!("MonoItem callable kind is always concrete"),
        };
        let function = Function {
            always_scopes,
            code: self.code,
            hints,
            debug_scopes,
            debug_variables,
            locations,
            name: self.tcx.symbol_name(self.instance).name.to_string(),
            params: self.params,
            registers: self.registers,
            result: self.result,
            source,
            closure_environment: self.closure_environment,
        };
        if self.supported.get() {
            return (Some(function), call_requests);
        }
        // A body outside this scalar dialect still names an exact scalar
        // signature, and an interchange interface reads a signed full-width
        // integer and an unsigned one alike. Recording the fallback here keeps the
        // advertised parameter types and callable kind the ones the source
        // spelled.
        let Some(&index) = self.indices.get(&function.name) else {
            return (None, call_requests);
        };
        (
            Some(Function {
                always_scopes: vec![AlwaysScope {
                    block: None,
                    capabilities: 0,
                    instruction: None,
                    kind: root_kind,
                }],
                code: vec![Instruction::portable_fallback(index)],
                hints: Vec::new(),
                debug_scopes: Vec::new(),
                debug_variables: Vec::new(),
                locations: Vec::new(),
                registers: function.params.clone(),
                closure_environment: ClosureEnvironment::None,
                ..function
            }),
            call_requests,
        )
    }

    fn marked_calls(&self) -> BTreeMap<BasicBlock, Request> {
        let mut marked = BTreeMap::<BasicBlock, Request>::new();
        for marker in &self.offload_markers {
            let block = marker.next;
            if self.source_calls.get(&block) != Some(&marker.callable) {
                self.tcx.dcx().fatal(
                    "PolyASM offload marker requires one immediately invoked direct call: `marker::<P, Arch, _>(function)(args)`",
                );
            }
            if let Some(previous) = marked.insert(block, marker.request)
                && previous != marker.request
            {
                self.tcx
                    .dcx()
                    .fatal("one PolyASM source call receives conflicting offload requests");
            }
        }
        marked
    }

    fn call_requests(&self, marked: &BTreeMap<BasicBlock, Request>) -> Vec<CallRequest> {
        let caller = self.tcx.symbol_name(self.instance).name.to_string();
        marked
            .iter()
            .map(|(&block, &request)| {
                let callable = self.source_calls[&block];
                let callee = self.tcx.symbol_name(callable).name.to_string();
                let source_block = u32::try_from(block.as_usize())
                    .expect("MIR call block fits u32");
                let offset = self.image.direct_call_offset(&caller, &callee, source_block)
                    .unwrap_or_else(|| self.tcx.dcx().fatal(format!(
                        "PolyASM offload marker in `{caller}` has no unique native direct call to `{callee}`",
                    )));
                CallRequest {
                    caller: caller.clone(),
                    callee,
                    offset,
                    request,
                }
            })
            .collect()
    }

    fn offload_hints(&self, marked: &BTreeMap<BasicBlock, Request>) -> Vec<Hint> {
        let mut calls = BTreeMap::<u32, Request>::new();
        for (&block, &request) in marked {
            if let Some(&(_, pc)) = self.calls.get(&block) {
                if let Some(previous) = calls.insert(pc, request)
                    && previous != request
                {
                    self.tcx
                        .dcx()
                        .fatal("one PolyASM call site receives conflicting offload requests");
                }
            }
        }
        calls
            .into_iter()
            .map(|(start, request)| {
                let [property, architecture] = request.to_requested_bytes();
                Hint {
                    start,
                    end: start.checked_add(1).expect("semantic call PC fits u32"),
                    kind: HintKind::Compiler,
                    value: OFFLOAD_CALL_HINT_TAG
                        | u64::from(property)
                        | (u64::from(architecture) << 8),
                }
            })
            .collect()
    }

    fn record_emitted_range(&mut self, location: Location, start: usize) {
        if start == self.code.len() {
            return;
        }
        let Some(start) = u32::try_from(start).ok() else {
            self.tcx.dcx().fatal("PolyASM function exceeds the 32-bit instruction address space");
        };
        let Some(end) = u32::try_from(self.code.len()).ok() else {
            self.tcx.dcx().fatal("PolyASM function exceeds the 32-bit instruction address space");
        };
        self.source_ranges.push(EmittedSourceRange { end, location, start });
    }

    fn debug_locations(&self) -> Vec<DebugLocation> {
        let mut locations = BTreeMap::new();
        for range in &self.source_ranges {
            let span = self.body.source_info(range.location).span;
            let Some((source, line, column)) =
                debug_position(self.tcx, self.source_indices, self.body.span, span)
            else {
                continue;
            };
            for pc in range.start..range.end {
                locations.entry(pc).or_insert(DebugLocation { column, line, pc, source });
            }
        }
        locations.into_values().collect()
    }

    fn debug_scopes(&self) -> Vec<DebugScope> {
        self.body
            .source_scopes
            .iter()
            .map(|scope| DebugScope { parent: scope.parent_scope.map(|parent| parent.as_u32()) })
            .collect()
    }

    fn debug_variables(&self) -> Vec<DebugVariable> {
        let mut variables = Vec::new();
        for variable in &self.body.var_debug_info {
            let VarDebugInfoContents::Place(place) = &variable.value else {
                // Constants live apart from MIR places, projections, storage
                // lifetimes and virtual registers, outside this typed place slice.
                continue;
            };
            let local = place.local.as_u32();
            let register = self.debug_register(*place);
            let mut projections = self.debug_projections(place.projection);
            if let Some(composite) = &variable.composite {
                projections.extend(self.debug_projections(&composite.projection));
            }
            let ranges = self.debug_ranges(place.local.as_usize());
            if ranges.is_empty() {
                continue;
            }
            variables.push(DebugVariable {
                name: variable.name.to_string(),
                scope: variable.source_info.scope.as_u32(),
                local,
                register,
                projections,
                ranges,
            });
        }
        variables
    }

    fn debug_projections(
        &self,
        projections: &[rustc_middle::mir::PlaceElem<'tcx>],
    ) -> Vec<DebugProjection> {
        projections
            .iter()
            .map(|projection| match projection {
                ProjectionElem::Deref => DebugProjection::Deref,
                ProjectionElem::Field(field, _) => DebugProjection::Field(field.as_u32()),
                ProjectionElem::Index(local) => DebugProjection::Index(local.as_u32()),
                ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                    DebugProjection::ConstantIndex {
                        offset: u32::try_from(*offset).unwrap_or_else(|_| {
                            self.tcx.dcx().fatal("PolyASM debug projection offset exceeds u32")
                        }),
                        min_length: u32::try_from(*min_length).unwrap_or_else(|_| {
                            self.tcx.dcx().fatal("PolyASM debug projection length exceeds u32")
                        }),
                        from_end: *from_end,
                    }
                }
                ProjectionElem::Subslice { from, to, from_end } => DebugProjection::Subslice {
                    from: u32::try_from(*from).unwrap_or_else(|_| {
                        self.tcx.dcx().fatal("PolyASM debug subslice start exceeds u32")
                    }),
                    to: u32::try_from(*to).unwrap_or_else(|_| {
                        self.tcx.dcx().fatal("PolyASM debug subslice end exceeds u32")
                    }),
                    from_end: *from_end,
                },
                ProjectionElem::Downcast(_, variant) => DebugProjection::Downcast(variant.as_u32()),
                ProjectionElem::OpaqueCast(_) => DebugProjection::OpaqueCast,
                ProjectionElem::UnwrapUnsafeBinder(_) => DebugProjection::UnwrapUnsafeBinder,
                ProjectionElem::PhantomDeref => {
                    rustc_span::bug!("encountered PhantomDeref in codegen")
                }
            })
            .collect()
    }

    fn debug_register(&self, place: rustc_middle::mir::Place<'tcx>) -> Option<Register> {
        let mut local = place.local.as_usize();
        let mut field = None;
        for projection in place.projection {
            match projection {
                ProjectionElem::Deref if field.is_none() => {
                    let target = match self.locals.get(local)?.as_ref()?.alias? {
                        PointerProvenance::Place(target) => target,
                        PointerProvenance::Static(_) => return None,
                    };
                    local = target.local;
                    field = target.field;
                }
                ProjectionElem::Field(index, _) if field.is_none() => {
                    field = Some(index.as_usize())
                }
                ProjectionElem::OpaqueCast(_) | ProjectionElem::UnwrapUnsafeBinder(_) => {}
                _ => return None,
            }
        }
        let registers = &self.locals.get(local)?.as_ref()?.fields;
        match field {
            Some(field) => registers.get(field).copied(),
            None if registers.len() == 1 => registers.first().copied(),
            None => None,
        }
    }

    fn debug_ranges(&self, local: usize) -> Vec<DebugRange> {
        let has_storage_live = self.body.basic_blocks.iter().any(|block| {
            block.statements.iter().any(|statement| {
                matches!(&statement.kind, StatementKind::StorageLive(value) if value.as_usize() == local)
            })
        });
        let mut start = (!has_storage_live).then_some(0_u32);
        let mut ranges = Vec::new();
        for (block_index, block) in self.body.basic_blocks.iter().enumerate() {
            for (statement_index, statement) in block.statements.iter().enumerate() {
                let pc = self.mir_pcs[block_index][statement_index];
                match &statement.kind {
                    StatementKind::StorageLive(value) if value.as_usize() == local => {
                        if let Some(previous) = start.replace(pc)
                            && previous < pc
                        {
                            ranges.push(DebugRange { start: previous, end: pc });
                        }
                    }
                    StatementKind::StorageDead(value) if value.as_usize() == local => {
                        if let Some(previous) = start.take()
                            && previous < pc
                        {
                            ranges.push(DebugRange { start: previous, end: pc });
                        }
                    }
                    _ => {}
                }
            }
        }
        let end = self.code.len() as u32;
        if let Some(previous) = start
            && previous < end
        {
            ranges.push(DebugRange { start: previous, end });
        }
        ranges.sort_unstable_by_key(|range| (range.start, range.end));
        ranges.dedup();
        ranges
    }

    fn statement_scopes(&self) -> Vec<AlwaysScope> {
        let mut scopes = Vec::new();
        for warrant in &self.statement_warrants {
            let mut ranges = self
                .source_ranges
                .iter()
                .filter(|range| warrant.selection.contains(range.location))
                .map(|range| range.start..range.end)
                .collect::<Vec<_>>();
            ranges.sort_unstable_by_key(|range| (range.start, range.end));
            let mut merged = Vec::<core::ops::Range<u32>>::with_capacity(ranges.len());
            for range in ranges {
                if let Some(previous) = merged.last_mut()
                    && range.start <= previous.end
                {
                    previous.end = previous.end.max(range.end);
                } else {
                    merged.push(range);
                }
            }
            if merged.is_empty() {
                self.tcx.dcx().span_err(
                    warrant.selection.argument_span(),
                    "PolyASM statement warrant lowered to no executable instructions",
                );
                self.supported.set(false);
                continue;
            }
            scopes.extend(merged.into_iter().map(|range| AlwaysScope {
                block: Some(range.start),
                capabilities: warrant.capability,
                instruction: Some(range.end),
                kind: AlwaysScopeKind::StatementRange,
            }));
        }
        scopes.sort_unstable_by_key(|scope| (scope.block, scope.instruction));
        scopes
    }

    fn assign_local(&mut self, local: usize, shape: LocalShape) {
        if self.locals[local].is_some() {
            return;
        }
        let mut fields = Vec::new();
        match &shape {
            LocalShape::Opaque => {}
            LocalShape::Pointer { address, .. } => fields.push(self.allocate(*address)),
            LocalShape::Scalar(ty) => fields.push(self.allocate(*ty)),
            LocalShape::Tuple(types) => {
                for &ty in types {
                    fields.push(self.allocate(ty));
                }
            }
            LocalShape::Unit => {}
        }
        self.locals[local] = Some(LocalRegisters { alias: None, fields, shape });
    }

    /// Declares one more register of this function's own register table and
    /// names it.
    ///
    /// The table is appended to, so registers allocated in a row are numbered
    /// in a row, which is what lets a call name only its first argument. A
    /// table that reached the one number the format reserves has spent every
    /// name: the body leaves this scalar dialect like every other body
    /// outside it (a 64 KiB byte array local alone fills the table), so the
    /// emitter keeps it on the canonical interchange section and the
    /// placeholder it answers stays out of every emitted function.
    fn allocate(&mut self, ty: ValueType) -> Register {
        let Some(register) = u16::try_from(self.registers.len()).ok().and_then(Register::named)
        else {
            self.supported.set(false);
            return Register::named(0).expect("register zero carries a name");
        };
        self.registers.push(ty);
        register
    }

    fn error(&self, _span: Span, _message: &str) {
        // The placement graph is an optional specialization sidecar. A MIR
        // body outside this scalar dialect remains executable through the
        // canonical interchange section and stays unadvertised here.
        self.supported.set(false);
    }

    fn local_shape(&self, ty: Ty<'tcx>, span: Span) -> LocalShape {
        let ty = self.monomorphize(ty);
        if ty.is_unit() || ty.is_never() {
            return LocalShape::Unit;
        }
        if let Some(scalar) = value_type(self.tcx, ty) {
            return LocalShape::Scalar(scalar);
        }
        match *ty.kind() {
            ty::FnDef(..) | ty::Closure(..) => LocalShape::Opaque,
            ty::Adt(definition, _)
                if self.tcx.is_diagnostic_item(sym::polyasm_static_callable, definition.did()) =>
            {
                LocalShape::Opaque
            }
            ty::Ref(_, pointee, _) | ty::RawPtr(pointee, _) => {
                self.capability_ceiling
                    .set(self.capability_ceiling.get() & !CAP_POINTER_ACCELERATORS);
                let pointee = value_type(self.tcx, pointee).unwrap_or(ValueType::U32);
                LocalShape::Pointer { address: pointer_value_type(self.tcx), pointee }
            }
            ty::Tuple(types) => {
                let mut output = Vec::new();
                for field in types {
                    let Some(ty) = value_type(self.tcx, field) else {
                        self.error(span, "PolyASM tuple fields must be scalar");
                        return LocalShape::Unit;
                    };
                    output.push(ty);
                }
                LocalShape::Tuple(output)
            }
            ty::Array(element, length) if matches!(element.kind(), ty::Uint(ty::UintTy::U8)) => {
                let Some(length) = length.try_to_target_usize(self.tcx) else {
                    self.error(span, "PolyASM byte arrays require a fixed length");
                    return LocalShape::Unit;
                };
                LocalShape::Tuple(vec![ValueType::U32; length as usize])
            }
            _ => {
                self.error(span, "type is not supported by native PolyASM codegen");
                LocalShape::Unit
            }
        }
    }

    fn monomorphize<T>(&self, value: T) -> T
    where
        T: ty::TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            ty::TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(self.tcx, value),
        )
    }

    fn emit_statement(&mut self, statement: &StatementKind<'tcx>, span: Span) {
        match statement {
            StatementKind::Assign(assignment) => {
                let (destination, value) = &**assignment;
                self.emit_assignment(*destination, value, span);
            }
            StatementKind::AscribeUserType(..)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::ConstEvalCounter
            | StatementKind::Coverage(..)
            | StatementKind::FakeRead(..)
            | StatementKind::Nop
            | StatementKind::PlaceMention(..)
            | StatementKind::StorageDead(..)
            | StatementKind::StorageLive(..) => {}
            StatementKind::SetDiscriminant { .. } => {
                self.error(span, "PolyASM does not support enum discriminants");
            }
            StatementKind::Intrinsic(..) => {
                self.error(span, "PolyASM does not support this non-diverging intrinsic");
            }
        }
    }
}

impl LocalShape {
    fn register_type(&self) -> Option<ValueType> {
        match self {
            Self::Pointer { address, .. } => Some(*address),
            Self::Scalar(ty) => Some(*ty),
            Self::Opaque | Self::Tuple(_) | Self::Unit => None,
        }
    }
}

fn address_token(place: SimplePlace) -> Option<i64> {
    let local = u64::try_from(place.local).ok()?;
    let field = u64::try_from(place.field.unwrap_or(0)).ok()?;
    let token = local.checked_add(1)?.checked_mul(256)?.checked_add(field.checked_mul(8)?)?;
    u32::try_from(token).ok().map(i64::from)
}

fn volatile_value_type<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<ValueType> {
    Some(match *ty.kind() {
        ty::Uint(ty::UintTy::U32) => ValueType::U32,
        ty::Uint(ty::UintTy::U64) => ValueType::U64,
        ty::Uint(ty::UintTy::Usize) => {
            if tcx.data_layout.pointer_size().bits() == 32 {
                ValueType::U32
            } else {
                ValueType::U64
            }
        }
        ty::Int(ty::IntTy::I32) => ValueType::I32,
        ty::Int(ty::IntTy::I64) => ValueType::I64,
        ty::Int(ty::IntTy::Isize) => {
            if tcx.data_layout.pointer_size().bits() == 32 {
                ValueType::I32
            } else {
                ValueType::I64
            }
        }
        _ => return None,
    })
}

fn value_type<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<ValueType> {
    Some(match *ty.kind() {
        ty::Bool | ty::Char | ty::Uint(ty::UintTy::U8 | ty::UintTy::U16 | ty::UintTy::U32) => {
            ValueType::U32
        }
        ty::Uint(ty::UintTy::U64) => ValueType::U64,
        ty::Uint(ty::UintTy::Usize) => {
            if tcx.data_layout.pointer_size().bits() == 32 {
                ValueType::U32
            } else {
                ValueType::U64
            }
        }
        ty::Int(ty::IntTy::I8 | ty::IntTy::I16 | ty::IntTy::I32) => ValueType::I32,
        ty::Int(ty::IntTy::I64) => ValueType::I64,
        ty::Int(ty::IntTy::Isize) => {
            if tcx.data_layout.pointer_size().bits() == 32 {
                ValueType::I32
            } else {
                ValueType::I64
            }
        }
        ty::Float(ty::FloatTy::F32) => ValueType::F32,
        ty::Float(ty::FloatTy::F64) => ValueType::F64,
        // A `repr(simd)` array names one of PolyASM's own vector widths. The
        // bank holds a 64-byte slot per register, so every width a guest
        // spells lands in one register rather than a pair, and the width is the
        // whole of what the wire tag has to carry.
        ty::Adt(definition, arguments) if definition.repr().simd() => {
            let layout = tcx
                .layout_of(
                    ::rustc_middle::ty::TypingEnv::fully_monomorphized().as_query_input(
                        ::rustc_middle::ty::Ty::new_adt(tcx, definition, arguments),
                    ),
                )
                .ok()?;
            match layout.size.bits() {
                128 => ValueType::V128,
                256 => ValueType::V256,
                512 => ValueType::V512,
                _ => return None,
            }
        }
        _ => return None,
    })
}

fn encoded_type<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<ValueType> {
    match ty.kind() {
        ty::RawPtr(..) | ty::Ref(..) => Some(pointer_value_type(tcx)),
        _ => value_type(tcx, ty),
    }
}

fn pointer_value_type(tcx: TyCtxt<'_>) -> ValueType {
    match tcx.data_layout.pointer_size().bits() {
        32 => ValueType::U32,
        64 => ValueType::U64,
        width => {
            tcx.dcx().fatal(format!("PolyASM target has unsupported {width}-bit pointer width"))
        }
    }
}

fn polyasm_u64_const<'tcx>(tcx: TyCtxt<'tcx>, constant: ty::Const<'tcx>) -> Option<u64> {
    let value = constant.try_to_value()?;
    if value.ty != tcx.types.u64 {
        return None;
    }
    value.try_to_bits(tcx, ty::TypingEnv::fully_monomorphized())?.try_into().ok()
}

pub(super) fn is_core_endian_helper(tcx: TyCtxt<'_>, instance: Instance<'_>) -> bool {
    tcx.crate_name(instance.def_id().krate).as_str() == "core"
        && tcx.opt_item_name(instance.def_id()).is_some_and(|name| {
            matches!(
                name.as_str(),
                "from_be_bytes"
                    | "from_le_bytes"
                    | "from_ne_bytes"
                    | "to_be_bytes"
                    | "to_le_bytes"
                    | "to_ne_bytes"
                    | "little_endian_transmute"
                    | "big_endian_transmute"
            )
        })
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum CompilerMarker {
    BindStaticClock,
    InvokeStaticFasterExact,
    InvokeStaticFasterOn,
    InvokeStaticFasterProven,
    Offload,
    RequestOffload,
    RequireAlways,
    RequireNotAlways,
    RequireStatement,
}

fn compiler_marker(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> Option<CompilerMarker> {
    if tcx.is_diagnostic_item(sym::polyasm_bind_static_clock, def_id) {
        Some(CompilerMarker::BindStaticClock)
    } else if tcx.is_diagnostic_item(sym::polyasm_invoke_static_faster, def_id) {
        Some(CompilerMarker::InvokeStaticFasterProven)
    } else if tcx.is_diagnostic_item(sym::polyasm_invoke_static_faster_exact, def_id) {
        Some(CompilerMarker::InvokeStaticFasterExact)
    } else if tcx.is_diagnostic_item(sym::polyasm_invoke_static_faster_on, def_id) {
        Some(CompilerMarker::InvokeStaticFasterOn)
    } else if tcx.is_diagnostic_item(sym::polyasm_offload, def_id) {
        Some(CompilerMarker::Offload)
    } else if tcx.is_diagnostic_item(sym::polyasm_request_offload, def_id) {
        Some(CompilerMarker::RequestOffload)
    } else if tcx.is_diagnostic_item(sym::polyasm_require_always, def_id) {
        Some(CompilerMarker::RequireAlways)
    } else if tcx.is_diagnostic_item(sym::polyasm_require_not_always, def_id) {
        Some(CompilerMarker::RequireNotAlways)
    } else if tcx.is_diagnostic_item(sym::polyasm_require_statement, def_id) {
        Some(CompilerMarker::RequireStatement)
    } else {
        None
    }
}

pub(super) fn is_compiler_marker(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> bool {
    compiler_marker(tcx, def_id).is_some()
}

fn is_warrant_only_compiler_marker_terminator(
    tcx: TyCtxt<'_>,
    terminator: &TerminatorKind<'_>,
) -> bool {
    let TerminatorKind::Call { func, .. } = terminator else { return false };
    func.const_fn_def().is_some_and(|(def_id, _)| {
        matches!(
            compiler_marker(tcx, def_id),
            Some(
                CompilerMarker::BindStaticClock
                    | CompilerMarker::Offload
                    | CompilerMarker::RequestOffload
                    | CompilerMarker::RequireAlways
                    | CompilerMarker::RequireNotAlways
                    | CompilerMarker::RequireStatement
            )
        )
    })
}
