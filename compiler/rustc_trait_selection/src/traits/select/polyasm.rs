//! Built-in PolyASM markers for concrete function items and closures.
//!
//! Lang items alone identify the marker traits. Diagnostic items from the
//! same target crate identify the property and architecture arguments; crate
//! names, path strings and source-level marker names carry zero authority
//! here.

mod aggregate;
mod packet;
mod slices;

use std::collections::VecDeque;

use aggregate::{PointerLayout, TypeInspection};
use polyasm_format::board::{
    FPGA_BOARDS, FpgaBoard, INSTRUCTION_CLASS_COUNT, InstructionClass as TimingClass,
};
use rustc_ast::ast;
use rustc_attr_ir::find_attr;
use rustc_attr_ir::lang_items::LangItem;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::mir::interpret::{GlobalAlloc, Scalar};
use rustc_middle::mir::visit::{PlaceContext, Visitor as MirVisitor};
use rustc_middle::mir::{
    AssertKind, BasicBlock, BinOp, Body, BorrowKind, CastKind, Local, Location, Operand, Place,
    PlaceTy, ProjectionElem, RawPtrKind, Rvalue, START_BLOCK, Statement, StatementKind, Terminator,
    TerminatorKind, UnOp,
};
use rustc_middle::mono::{polyasm_witness_callables, resolve_polyasm_callable};
use rustc_middle::ty::consts::ConstExt;
use rustc_middle::ty::fast_reject::{TreatParams, simplify_type};
use rustc_middle::ty::{self, EarlyBinder, Instance, Ty, TyCtxt, TypeFoldable, TypeVisitableExt};
use rustc_span::{Span, Symbol, sym};

const WASM: u32 = 1 << 0;
const RUST: u32 = 1 << 1;
const EBPF: u32 = 1 << 2;
const XDP: u32 = 1 << 3;
const VERILOG: u32 = 1 << 4;
const STATIC_MEMORY: u32 = 1 << 5;
const STATIC_MEMORY_UPPER: u32 = 1 << 6;
const LINUX_SAFE: u32 = 1 << 7;
const STATIC_CLOCK: u32 = 1 << 8;
const ACCELERATOR: u32 = 1 << 9;
const NATIVE: u32 = 1 << 10;
const CUDA: u32 = 1 << 11;
const DPA: u32 = 1 << 12;
const PTX: u32 = 1 << 13;
const P4: u32 = 1 << 14;
const LLVM_BRIDGE: u32 = 1 << 15;
const ALL: u32 = (1 << 16) - 1;
const CAPABILITY_COUNT: usize = 16;
const POINTER_ACCELERATORS: u32 = ACCELERATOR | CUDA | DPA | PTX | P4;

#[derive(Copy, Clone)]
pub(crate) enum WitnessDecision {
    OtherTrait,
    Ambiguous,
    Proven,
    Rejected,
}

#[derive(Copy, Clone)]
enum Property {
    Capability(u32),
    StaticClock,
}

#[derive(Copy, Clone)]
enum Architecture {
    Aarch64,
    Cxl,
    Dpu,
    Ebpf,
    Fpga,
    Gpu,
    Host,
    Npu,
    Polyasm,
    Riscv64,
    Wasm32,
    X86_64,
    Xdp,
    FpgaBoard(&'static FpgaBoard),
}

#[derive(Copy, Clone)]
struct TimingCounts([u64; INSTRUCTION_CLASS_COUNT]);

#[derive(Copy, Clone)]
struct Report {
    capabilities: u32,
    disproven: u32,
    cycles: Option<u64>,
    timing: TimingCounts,
    failures: [Option<Failure>; CAPABILITY_COUNT],
}

#[derive(Copy, Clone)]
struct Failure {
    reason: &'static str,
    span: Span,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Warrant {
    Deferred,
    Disproven,
    Proven,
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct MemoryLayout {
    align: u64,
    bytes: u64,
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct MemoryAccess {
    address_only: bool,
    allocation: MemoryLayout,
    offset: u64,
    pointee: MemoryLayout,
    writable: bool,
}

impl MemoryAccess {
    fn permits(self, pointee: MemoryLayout) -> bool {
        !self.address_only
            && self.pointee == pointee
            && self.offset.checked_add(pointee.bytes).is_some_and(|end| {
                end <= self.allocation.bytes && effective_alignment(self) >= pointee.align
            })
    }

    fn retarget(self, pointee: MemoryLayout) -> Self {
        if self.pointee == pointee { self } else { Self { address_only: true, pointee, ..self } }
    }

    fn permits_write(self, pointee: MemoryLayout) -> bool {
        self.writable && self.permits(pointee)
    }

    fn project(self, offset: u64, pointee: MemoryLayout) -> Option<Self> {
        let projected = Self { offset: self.offset.checked_add(offset)?, pointee, ..self };
        projected.permits(pointee).then_some(projected)
    }

    fn restrict_write(self, writable: bool) -> Self {
        Self { writable: self.writable && writable, ..self }
    }

    fn without_dereference_authority(self) -> Self {
        Self { address_only: true, ..self }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum HirMemoryProvenance {
    FixedStatic(MemoryAccess),
    InBounds(MemoryAccess),
    Unknown,
}

impl HirMemoryProvenance {
    fn access(self) -> Option<MemoryAccess> {
        match self {
            Self::FixedStatic(access) | Self::InBounds(access) => Some(access),
            Self::Unknown => None,
        }
    }

    fn is_fixed_static(self) -> bool {
        matches!(self, Self::FixedStatic(_))
    }

    fn project(self, offset: u64, pointee: MemoryLayout) -> Self {
        match self {
            Self::FixedStatic(access) => {
                access.project(offset, pointee).map_or(Self::Unknown, Self::FixedStatic)
            }
            Self::InBounds(access) => {
                access.project(offset, pointee).map_or(Self::Unknown, Self::InBounds)
            }
            Self::Unknown => Self::Unknown,
        }
    }

    fn retarget(self, pointee: MemoryLayout) -> Self {
        match self {
            Self::FixedStatic(access) => Self::FixedStatic(access.retarget(pointee)),
            Self::InBounds(access) => Self::InBounds(access.retarget(pointee)),
            Self::Unknown => Self::Unknown,
        }
    }

    fn restrict_write(self, writable: bool) -> Self {
        match self {
            Self::FixedStatic(access) => Self::FixedStatic(access.restrict_write(writable)),
            Self::InBounds(access) => Self::InBounds(access.restrict_write(writable)),
            Self::Unknown => Self::Unknown,
        }
    }

    fn without_dereference_authority(self) -> Self {
        match self {
            Self::FixedStatic(access) => Self::FixedStatic(access.without_dereference_authority()),
            Self::InBounds(access) => Self::InBounds(access.without_dereference_authority()),
            Self::Unknown => Self::Unknown,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum MemoryOrigin {
    Argument(Local),
    BranchMerged,
    ClosureEnvironment(DefId),
    FixedStatic(DefId),
    Local(Local),
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum MemoryProvenance {
    Proven { access: MemoryAccess, origin: MemoryOrigin },
    Unknown,
}

impl MemoryProvenance {
    fn access(self) -> Option<MemoryAccess> {
        match self {
            Self::Proven { access, .. } => Some(access),
            Self::Unknown => None,
        }
    }

    fn is_fixed_static(self) -> bool {
        matches!(self, Self::Proven { origin: MemoryOrigin::FixedStatic(_), .. })
    }

    fn has_nonzero_address(self) -> bool {
        matches!(self, Self::Proven { .. })
    }

    fn retarget(self, pointee: MemoryLayout) -> Self {
        match self {
            Self::Proven { access, origin } => {
                Self::Proven { access: access.retarget(pointee), origin }
            }
            Self::Unknown => Self::Unknown,
        }
    }

    fn restrict_write(self, writable: bool) -> Self {
        match self {
            Self::Proven { access, origin } => {
                Self::Proven { access: access.restrict_write(writable), origin }
            }
            Self::Unknown => Self::Unknown,
        }
    }

    fn without_dereference_authority(self) -> Self {
        match self {
            Self::Proven { access, origin } => {
                Self::Proven { access: access.without_dereference_authority(), origin }
            }
            Self::Unknown => Self::Unknown,
        }
    }
}

pub(crate) struct WitnessFailure {
    pub(crate) message: String,
    pub(crate) span: Span,
}

/// A source-located reason an architecture-relative static choice stayed open
/// through monomorphization.
#[derive(Clone)]
pub struct PolyasmStaticCallableSelectionError {
    message: String,
    span: Span,
}

impl PolyasmStaticCallableSelectionError {
    /// Returns the fail-closed selection diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the callable or marker span responsible for the decision.
    pub fn span(&self) -> Span {
        self.span
    }
}

/// Exact normalized-MIR dependency slice for one source-statement marker.
///
/// The marker's MIR [`Location`] is its identity. Membership decisions leave
/// source spans out on purpose: macro expansions and optimized MIR assign one
/// coarse source span to several unrelated expressions at times.
#[derive(Clone)]
pub struct PolyasmStatementSelection {
    argument_span: Span,
    locations: FxHashSet<Location>,
    marker: Location,
}

/// One compiler-checked choice between two exact static-clock callables.
#[derive(Copy, Clone)]
pub struct PolyasmStaticCallableSelection<'tcx> {
    lhs: Instance<'tcx>,
    lhs_ty: Ty<'tcx>,
    rhs: Instance<'tcx>,
    rhs_ty: Ty<'tcx>,
    selected_left: bool,
}

impl<'tcx> PolyasmStaticCallableSelection<'tcx> {
    /// Returns the normalized left callable instance.
    pub fn lhs(&self) -> Instance<'tcx> {
        self.lhs
    }

    /// Returns the normalized right callable instance.
    pub fn rhs(&self) -> Instance<'tcx> {
        self.rhs
    }

    /// Returns the only callable selected for executable lowering.
    pub fn selected(&self) -> Instance<'tcx> {
        if self.selected_left { self.lhs } else { self.rhs }
    }

    /// Returns the selected callable's concrete source type.
    pub fn selected_ty(&self) -> Ty<'tcx> {
        if self.selected_left { self.lhs_ty } else { self.rhs_ty }
    }

    /// Returns whether equal-or-faster elapsed time selected the left side.
    pub fn selected_left(&self) -> bool {
        self.selected_left
    }
}

impl PolyasmStatementSelection {
    /// Returns the marker argument span for source diagnostics only.
    pub fn argument_span(&self) -> Span {
        self.argument_span
    }

    /// Returns whether this exact MIR operation contributes to the argument.
    pub fn contains(&self, location: Location) -> bool {
        self.locations.contains(&location)
    }

    /// Returns whether the marker operand lacks a preceding MIR definition.
    ///
    /// Constants and direct function arguments are evaluated at the marker
    /// location itself by the semantic emitter.
    pub fn is_direct(&self) -> bool {
        self.locations.len() == 1
    }

    /// Returns the unique MIR location of the compiler statement marker.
    pub fn marker(&self) -> Location {
        self.marker
    }
}

pub(crate) fn decide<'tcx>(tcx: TyCtxt<'tcx>, trait_ref: ty::TraitRef<'tcx>) -> WitnessDecision {
    if !tcx.sess.is_polyasm_target() {
        return WitnessDecision::OtherTrait;
    }
    let Some(item) = tcx.as_lang_item(trait_ref.def_id) else {
        return WitnessDecision::OtherTrait;
    };
    let (negative, static_schedule) = match item {
        LangItem::PolyasmAlways => (false, false),
        LangItem::PolyasmCompilerCertificate => (false, false),
        LangItem::PolyasmCompilerCounterexample => (true, false),
        LangItem::PolyasmCompilerStaticClockCertificate => (false, true),
        _ => return WitnessDecision::OtherTrait,
    };
    if trait_ref.args.has_infer() || trait_ref.args.has_non_region_param() {
        return WitnessDecision::Ambiguous;
    }
    let self_ty = trait_ref.args.type_at(0);
    if item == LangItem::PolyasmCompilerCertificate
        && !matches!(self_ty.kind(), ty::FnDef(..) | ty::Closure(..))
    {
        // The witness trait also carries the sealed checker tokens of its
        // defining crate. Coherence confines their unsafe impls to the
        // trait's defining crate, and they continue through ordinary impl
        // selection. Compiler derivation covers concrete callable bodies alone.
        return WitnessDecision::OtherTrait;
    }
    let report = match *self_ty.kind() {
        ty::FnDef(def_id, args) => {
            let Some(args) = args.no_bound_vars() else {
                return WitnessDecision::Ambiguous;
            };
            analyze_fn_hir(tcx, def_id, args, &mut FxHashSet::default())
        }
        ty::Closure(def_id, args) => {
            analyze_closure_hir(tcx, def_id, args, &mut FxHashSet::default())
        }
        _ if self_ty.has_non_region_param() => return WitnessDecision::Ambiguous,
        _ => return WitnessDecision::Rejected,
    };
    let Some(report) = report else {
        return WitnessDecision::Ambiguous;
    };

    decide_report(tcx, trait_ref, negative, static_schedule, report)
}

/// Rechecks a property marker against fully normalized, monomorphized MIR.
///
/// Closure obligations use a conservative HIR reading while their enclosing
/// body is under type checking, which avoids a query cycle. Code generation
/// calls this function before erasing the compiler marker from executable MIR.
/// `Some(false)` stands for an invariant counterexample alone; `None` means
/// the analysis is unavailable or the requested capability remains deferred.
pub fn normalized_polyasm_property_holds<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    property_ty: Ty<'tcx>,
    architecture_ty: Ty<'tcx>,
) -> Option<bool> {
    if !tcx.sess.is_polyasm_target() {
        return None;
    }
    let property = property(tcx, property_ty)?;
    let architecture = architecture(tcx, architecture_ty)?;
    let capability = match property {
        Property::Capability(capability) => capability,
        Property::StaticClock => STATIC_CLOCK,
    };
    if !accepts(architecture, capability) {
        return Some(false);
    }
    match analyze(tcx, instance, &mut FxHashSet::default())?.warrant(capability) {
        Warrant::Proven => Some(true),
        Warrant::Disproven => Some(false),
        Warrant::Deferred => None,
    }
}

/// Returns the exact normalized-MIR reason for a declined callable property.
///
/// This is diagnostic-only. [`normalized_polyasm_property_holds`] remains the
/// admission authority and both functions independently recompute the report
/// from the same concrete instance.
pub fn explain_normalized_polyasm_rejection<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    property_ty: Ty<'tcx>,
    architecture_ty: Ty<'tcx>,
) -> Option<(Span, &'static str)> {
    if !tcx.sess.is_polyasm_target() {
        return None;
    }
    let property = property(tcx, property_ty)?;
    let architecture = architecture(tcx, architecture_ty)?;
    let capability = match property {
        Property::Capability(capability) => capability,
        Property::StaticClock => STATIC_CLOCK,
    };
    if !accepts(architecture, capability) {
        return Some((
            tcx.def_span(instance.def_id()),
            "the requested property is not supported by this architecture",
        ));
    }
    let failure = analyze(tcx, instance, &mut FxHashSet::default())?.failure(capability)?;
    Some((failure.span, failure.reason))
}

/// Resolves one compiler-owned statement marker to its exact normalized-MIR
/// dependency slice.
///
/// The backward walk stops at the nearest reaching definition of each local
/// and follows exactly the values and control edges the marker argument
/// requires. Another expression with the same value type or source span
/// therefore stays outside this marker's slice.
pub fn polyasm_statement_selection<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    marker: Location,
) -> Option<PolyasmStatementSelection> {
    if !tcx.sess.is_polyasm_target() {
        return None;
    }
    let body = tcx.instance_mir(instance.def);
    let data = body.basic_blocks.get(marker.block)?;
    if marker.statement_index != data.statements.len() {
        return None;
    }
    let terminator = data.terminator();
    let TerminatorKind::Call { func, args, .. } = &terminator.kind else {
        return None;
    };
    let (def_id, _) = func.const_fn_def()?;
    if !tcx.is_diagnostic_item(sym::polyasm_require_statement, def_id) {
        return None;
    }
    let [argument] = &args[..] else { return None };
    if matches!(argument.node, Operand::RuntimeChecks(_)) || argument.span.is_dummy() {
        return None;
    }

    let mut locations = FxHashSet::default();
    locations.insert(marker);
    let mut pending = VecDeque::new();
    let mut seeds = LocalUses::default();
    seeds.visit_operand(&argument.node, marker);
    for local in sorted_locals(&seeds.locals) {
        pending.push_back(DependencyPoint {
            before: marker.statement_index,
            block: marker.block,
            local,
        });
    }
    let mut visited = FxHashSet::default();
    while let Some(point) = pending.pop_front() {
        if !visited.insert(point) {
            continue;
        }
        let data = &body.basic_blocks[point.block];
        let mut found = false;
        for statement_index in (0..point.before.min(data.statements.len())).rev() {
            let statement = &data.statements[statement_index];
            let location = Location { block: point.block, statement_index };
            if !defines_local(statement, location, point.local) {
                continue;
            }
            locations.insert(location);
            enqueue_statement_dependencies(statement, location, &mut pending);
            if statement_partially_defines(statement, point.local) {
                pending.push_back(DependencyPoint {
                    before: statement_index,
                    block: point.block,
                    local: point.local,
                });
            }
            found = true;
            break;
        }
        if found {
            continue;
        }

        for &predecessor in &body.basic_blocks.predecessors()[point.block] {
            let predecessor_data = &body.basic_blocks[predecessor];
            let predecessor_location =
                Location { block: predecessor, statement_index: predecessor_data.statements.len() };
            let predecessor_terminator = predecessor_data.terminator();
            if terminator_defines_local(predecessor_terminator, predecessor_location, point.local) {
                locations.insert(predecessor_location);
                enqueue_terminator_dependencies(
                    predecessor_terminator,
                    predecessor_location,
                    &mut pending,
                );
                continue;
            }
            if matches!(
                predecessor_terminator.kind,
                TerminatorKind::SwitchInt { .. } | TerminatorKind::Assert { .. }
            ) {
                locations.insert(predecessor_location);
                enqueue_terminator_dependencies(
                    predecessor_terminator,
                    predecessor_location,
                    &mut pending,
                );
            }
            pending.push_back(DependencyPoint {
                before: predecessor_data.statements.len(),
                block: predecessor,
                local: point.local,
            });
        }
    }

    Some(PolyasmStatementSelection { argument_span: argument.span, locations, marker })
}

/// Rechecks one exact source expression carried by the compiler-owned
/// statement marker against fully monomorphized MIR.
pub fn normalized_polyasm_statement_property_holds<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    selection: &PolyasmStatementSelection,
    property_ty: Ty<'tcx>,
    architecture_ty: Ty<'tcx>,
) -> Option<bool> {
    if !tcx.sess.is_polyasm_target() {
        return None;
    }
    let property = property(tcx, property_ty)?;
    let architecture = architecture(tcx, architecture_ty)?;
    let capability = match property {
        Property::Capability(capability) => capability,
        Property::StaticClock => STATIC_CLOCK,
    };
    if !accepts(architecture, capability) {
        return Some(false);
    }
    match analyze_statement(tcx, instance, selection, &mut FxHashSet::default())?
        .warrant(capability)
    {
        Warrant::Proven => Some(true),
        Warrant::Disproven => Some(false),
        Warrant::Deferred => None,
    }
}

/// Returns the precise MIR decision associated with a failed statement
/// marker. The marker call remains the primary diagnostic span; this location
/// labels the operation that ended the requested property.
pub fn explain_normalized_polyasm_statement_rejection<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    selection: &PolyasmStatementSelection,
    property_ty: Ty<'tcx>,
    architecture_ty: Ty<'tcx>,
) -> Option<(Span, &'static str)> {
    let property = property(tcx, property_ty)?;
    let architecture = architecture(tcx, architecture_ty)?;
    let capability = match property {
        Property::Capability(capability) => capability,
        Property::StaticClock => STATIC_CLOCK,
    };
    if !accepts(architecture, capability) {
        return Some((
            selection.argument_span(),
            "the requested property is not supported by this architecture",
        ));
    }
    let report = analyze_statement(tcx, instance, selection, &mut FxHashSet::default())?;
    report.failure(capability).map(|failure| (failure.span, failure.reason))
}

/// Times the complete normalized graph with the exact board profile.
///
/// Owner-HIR inspection is provisional because trait selection stays outside
/// its own normalized MIR. This function is the authoritative static schedule
/// used after monomorphization and by code generation. `None` means the graph
/// is dynamic or unsupported, or the architecture lies outside the concrete
/// registered boards; every answer is a static schedule, distinct from a
/// runtime estimate.
pub fn normalized_polyasm_static_schedule<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    architecture_ty: Ty<'tcx>,
) -> Option<u64> {
    if !tcx.sess.is_polyasm_target() {
        return None;
    }
    let architecture = architecture(tcx, architecture_ty)?;
    let board = static_clock_board(architecture)?;
    let normalized = analyze(tcx, instance, &mut FxHashSet::default())?;
    if normalized.warrant(STATIC_CLOCK) != Warrant::Proven {
        return None;
    }
    normalized.timing.schedule(board)
}

/// Returns the immutable clock registered for one exact PolyASM board type.
pub fn polyasm_static_clock_hz(tcx: TyCtxt<'_>, architecture_ty: Ty<'_>) -> Option<u64> {
    static_clock_board(architecture(tcx, architecture_ty)?).map(FpgaBoard::clock_hz)
}

/// Selects one fully monomorphized callable after rechecking both exact
/// schedules against normalized MIR.
pub fn polyasm_static_callable_selection<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: ty::GenericArgsRef<'tcx>,
    span: Span,
) -> Option<PolyasmStaticCallableSelection<'tcx>> {
    polyasm_static_callable_selection_with_visiting(tcx, args, span, &mut FxHashSet::default())
}

/// Selects one callable for a single architecture from measured timing data,
/// leaving virtual targets to their own owners.
pub fn polyasm_static_callable_selection_on<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: ty::GenericArgsRef<'tcx>,
    span: Span,
) -> Result<PolyasmStaticCallableSelection<'tcx>, PolyasmStaticCallableSelectionError> {
    polyasm_static_callable_selection_on_with_visiting(tcx, args, span, &mut FxHashSet::default())
}

/// Returns the callables one PolyASM marker hands to code generation.
///
/// [`polyasm_witness_callables`] names every subject a marker establishes,
/// which is what the analyzer reads. Establishing a property and running a
/// body are separate: a static-clock comparison runs exactly one side, and a
/// clock binding only measures the body it describes. The callables returned
/// here alone become monomorphization and offload roots, so the declined side
/// of a comparison is established, reported, and then left unlowered (LANG
/// rule 17).
pub fn polyasm_offload_roots<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: ty::GenericArgsRef<'tcx>,
    span: Span,
) -> [Option<Ty<'tcx>>; 2] {
    let subjects = polyasm_witness_callables(tcx, def_id, args);
    if subjects.iter().all(Option::is_none) {
        return subjects;
    }
    // A clock binding measures the body it describes and leaves it unrun.
    if tcx.is_diagnostic_item(sym::polyasm_bind_static_clock, def_id) {
        return [None, None];
    }
    if !is_polyasm_static_invoke_marker(tcx, def_id) {
        return subjects;
    }
    let selection = if tcx.is_diagnostic_item(sym::polyasm_invoke_static_faster_on, def_id) {
        polyasm_static_callable_selection_on(tcx, args, span).ok()
    } else {
        polyasm_static_callable_selection(tcx, args, span)
    };
    // A comparison the compiler leaves undecided stops where it is
    // lowered. Both subjects stay roots until that diagnostic is emitted, so
    // every callable identity stays in an artifact that is still being
    // reported on.
    match selection {
        Some(selection) => [Some(selection.selected_ty()), None],
        None => subjects,
    }
}

fn polyasm_static_callable_selection_on_with_visiting<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: ty::GenericArgsRef<'tcx>,
    span: Span,
    visiting: &mut FxHashSet<DefId>,
) -> Result<PolyasmStaticCallableSelection<'tcx>, PolyasmStaticCallableSelectionError> {
    if !tcx.sess.is_polyasm_target() {
        return Err(static_selection_error(
            span,
            "invoke_static_faster_on requires the polyasm-unknown-unknown compiler target",
        ));
    }
    if args.has_infer() || args.has_non_region_param() {
        return Err(static_selection_error(
            span,
            "invoke_static_faster_on lost its fully monomorphized callable or architecture type",
        ));
    }
    let Some(lhs_ty) = args.first().and_then(|argument| argument.as_type()) else {
        return Err(static_selection_error(
            span,
            "invoke_static_faster_on has no concrete left callable type",
        ));
    };
    let Some(rhs_ty) = args.get(1).and_then(|argument| argument.as_type()) else {
        return Err(static_selection_error(
            span,
            "invoke_static_faster_on has no concrete right callable type",
        ));
    };
    let Some(architecture_ty) = args.get(2).and_then(|argument| argument.as_type()) else {
        return Err(static_selection_error(
            span,
            "invoke_static_faster_on has no concrete architecture type",
        ));
    };
    let Some(architecture) = architecture(tcx, architecture_ty) else {
        return Err(static_selection_error(
            span,
            "invoke_static_faster_on requires a compiler-known architecture with an immutable timing model",
        ));
    };
    let lhs = closed_static_selection_callable(tcx, lhs_ty, "left", span)?;
    let rhs = closed_static_selection_callable(tcx, rhs_ty, "right", span)?;
    let lhs_report = static_selection_report(tcx, lhs, "left", span, visiting)?;
    let rhs_report = static_selection_report(tcx, rhs, "right", span, visiting)?;
    let selected_left = if let Some(board) = static_clock_board(architecture) {
        let Some(lhs_cycles) = lhs_report.timing.schedule(board) else {
            return Err(static_selection_error(
                tcx.def_span(lhs.def_id()),
                "the left callable's registered-board cycle total exceeds u64",
            ));
        };
        let Some(rhs_cycles) = rhs_report.timing.schedule(board) else {
            return Err(static_selection_error(
                tcx.def_span(rhs.def_id()),
                "the right callable's registered-board cycle total exceeds u64",
            ));
        };
        lhs_cycles <= rhs_cycles
    } else if lhs_report.timing.dominates(rhs_report.timing) {
        true
    } else if rhs_report.timing.dominates(lhs_report.timing) {
        false
    } else {
        return Err(static_selection_error(
            span,
            "neither timing profile dominates on arch::Current; ordering is device-dependent—select a registered architecture or use invoke_runtime_faster",
        ));
    };
    Ok(PolyasmStaticCallableSelection { lhs, lhs_ty, rhs, rhs_ty, selected_left })
}

fn closed_static_selection_callable<'tcx>(
    tcx: TyCtxt<'tcx>,
    callable_ty: Ty<'tcx>,
    side: &'static str,
    span: Span,
) -> Result<Instance<'tcx>, PolyasmStaticCallableSelectionError> {
    let definition_span = match *callable_ty.kind() {
        ty::FnDef(def_id, _) => tcx.def_span(def_id),
        ty::Closure(def_id, args) if args.as_closure().tupled_upvars_ty().is_unit() => {
            tcx.def_span(def_id)
        }
        ty::Closure(def_id, _) => {
            return Err(static_selection_error(
                tcx.def_span(def_id),
                format!(
                    "the {side} callable captures runtime state; invoke_static_faster_on accepts only function items or unit-environment closures"
                ),
            ));
        }
        ty::FnPtr(..) => {
            return Err(static_selection_error(
                span,
                format!(
                    "the {side} callable is a function pointer with no body identity; pass the concrete function item directly"
                ),
            ));
        }
        _ => {
            return Err(static_selection_error(
                span,
                format!("the {side} operand is not a concrete function item or closure"),
            ));
        }
    };
    resolve_polyasm_callable(tcx, callable_ty, span).ok_or_else(|| {
        static_selection_error(
            definition_span,
            format!("the {side} callable identity could not be resolved after monomorphization"),
        )
    })
}

fn static_selection_report<'tcx>(
    tcx: TyCtxt<'tcx>,
    callable: Instance<'tcx>,
    side: &'static str,
    span: Span,
    visiting: &mut FxHashSet<DefId>,
) -> Result<Report, PolyasmStaticCallableSelectionError> {
    let Some(report) = analyze(tcx, callable, visiting) else {
        return Err(static_selection_error(
            tcx.def_span(callable.def_id()),
            format!("the {side} callable has no complete normalized StaticClock schedule"),
        ));
    };
    if report.warrant(STATIC_CLOCK) == Warrant::Proven {
        return Ok(report);
    }
    let (reason, failure_span) = report
        .failure(STATIC_CLOCK)
        .map_or(("its normalized timing graph is incomplete", span), |failure| {
            (failure.reason, failure.span)
        });
    Err(static_selection_error(
        failure_span,
        format!("the {side} callable has no complete normalized StaticClock schedule: {reason}"),
    ))
}

fn static_selection_error(
    span: Span,
    message: impl Into<String>,
) -> PolyasmStaticCallableSelectionError {
    PolyasmStaticCallableSelectionError { message: message.into(), span }
}

fn polyasm_static_callable_selection_with_visiting<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: ty::GenericArgsRef<'tcx>,
    span: Span,
    visiting: &mut FxHashSet<DefId>,
) -> Option<PolyasmStaticCallableSelection<'tcx>> {
    if !tcx.sess.is_polyasm_target() || args.has_infer() || args.has_non_region_param() {
        return None;
    }
    let lhs_ty = args.first()?.as_type()?;
    let rhs_ty = args.get(1)?.as_type()?;
    let lhs_architecture = args.get(2)?.as_type()?;
    let rhs_architecture = args.get(3)?.as_type()?;
    let lhs_cycles = polyasm_u64_const(tcx, args.get(5)?.as_const()?)?;
    let rhs_cycles = polyasm_u64_const(tcx, args.get(6)?.as_const()?)?;
    if !is_closed_static_callable(lhs_ty) || !is_closed_static_callable(rhs_ty) {
        return None;
    }
    let lhs = resolve_polyasm_callable(tcx, lhs_ty, span)?;
    let rhs = resolve_polyasm_callable(tcx, rhs_ty, span)?;
    let lhs_board = static_clock_board(architecture(tcx, lhs_architecture)?)?;
    let rhs_board = static_clock_board(architecture(tcx, rhs_architecture)?)?;
    if lhs_board.clock_hz() == 0 || rhs_board.clock_hz() == 0 {
        return None;
    }
    let lhs_report = analyze(tcx, lhs, visiting)?;
    if lhs_report.warrant(STATIC_CLOCK) != Warrant::Proven
        || lhs_report.timing.schedule(lhs_board) != Some(lhs_cycles)
    {
        return None;
    }
    let rhs_report = analyze(tcx, rhs, visiting)?;
    if rhs_report.warrant(STATIC_CLOCK) != Warrant::Proven
        || rhs_report.timing.schedule(rhs_board) != Some(rhs_cycles)
    {
        return None;
    }
    let lhs_elapsed = u128::from(lhs_cycles) * u128::from(rhs_board.clock_hz());
    let rhs_elapsed = u128::from(rhs_cycles) * u128::from(lhs_board.clock_hz());
    Some(PolyasmStaticCallableSelection {
        lhs,
        lhs_ty,
        rhs,
        rhs_ty,
        selected_left: lhs_elapsed <= rhs_elapsed,
    })
}

fn is_closed_static_callable(callable: Ty<'_>) -> bool {
    match *callable.kind() {
        ty::FnDef(..) => true,
        ty::Closure(_, args) => args.as_closure().tupled_upvars_ty().is_unit(),
        _ => false,
    }
}

fn decide_report<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    negative: bool,
    static_schedule: bool,
    report: Report,
) -> WitnessDecision {
    if static_schedule {
        let Some(architecture) = architecture(tcx, trait_ref.args.type_at(1)) else {
            return WitnessDecision::Rejected;
        };
        let Some(requested_cycles) = polyasm_u64_const(tcx, trait_ref.args.const_at(2)) else {
            return WitnessDecision::Ambiguous;
        };
        if static_clock_board(architecture).is_none() {
            return WitnessDecision::Rejected;
        }
        return if report.warrant(STATIC_CLOCK) == Warrant::Deferred {
            // Unresolved generic schedules returned above. A concrete
            // callable that lacks an exact owner-HIR schedule receives the
            // static-clock trait's dedicated fail-closed diagnostic.
            WitnessDecision::Rejected
        } else if report.warrant(STATIC_CLOCK) == Warrant::Proven {
            // This owner-HIR reading is provisional. Querying normalized MIR
            // here would recursively request the owner's typeck result.
            // Monomorphization and code generation both compare
            // `requested_cycles` with the board-timed normalized graph.
            let _ = requested_cycles;
            WitnessDecision::Proven
        } else {
            WitnessDecision::Rejected
        };
    }

    let Some(property) = property(tcx, trait_ref.args.type_at(1)) else {
        return WitnessDecision::Rejected;
    };
    let Some(architecture) = architecture(tcx, trait_ref.args.type_at(2)) else {
        return WitnessDecision::Rejected;
    };
    let capability = match property {
        Property::Capability(capability) => capability,
        Property::StaticClock => STATIC_CLOCK,
    };
    let warrant = if accepts(architecture, capability) {
        report.warrant(capability)
    } else {
        Warrant::Disproven
    };
    match (negative, warrant) {
        // Monomorphization checks concrete callable properties again.
        // Method resolution and iterator desugaring are unavailable
        // while this owner's typeck query is running, so their HIR report is
        // provisional. `rustc_monomorphize::mono_checks::polyasm` rechecks
        // every concrete marker obligation against normalized MIR, including
        // obligations in wrappers whose marker call is optimized away. A
        // negative marker still requires a concrete counterexample.
        (false, Warrant::Deferred)
            if matches!(capability, WASM | VERILOG | EBPF | XDP | LINUX_SAFE) =>
        {
            WitnessDecision::Proven
        }
        // Generic obligations have already returned `Ambiguous` above. A
        // concrete callable with an inconclusive owner-HIR report fails
        // closed here so `diagnostic::on_unimplemented` explains the
        // missing property in place of an unrelated inference error.
        (_, Warrant::Deferred) => WitnessDecision::Rejected,
        (false, Warrant::Proven) | (true, Warrant::Disproven) => WitnessDecision::Proven,
        (false, Warrant::Disproven) | (true, Warrant::Proven) => WitnessDecision::Rejected,
    }
}

/// Returns the MIR source location and concrete reason behind a declined
/// compiler marker. This serves diagnostics alone: [`decide`] remains the sole
/// trait-solver authority.
pub(crate) fn explain_rejection<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    obligation_span: Span,
) -> Option<WitnessFailure> {
    if !tcx.sess.is_polyasm_target() {
        return None;
    }
    let item = tcx.as_lang_item(trait_ref.def_id)?;
    let (negative, static_schedule) = match item {
        LangItem::PolyasmAlways => (false, false),
        LangItem::PolyasmCompilerCertificate => (false, false),
        LangItem::PolyasmCompilerCounterexample => (true, false),
        LangItem::PolyasmCompilerStaticClockCertificate => (false, true),
        _ => return None,
    };
    if trait_ref.args.has_infer() || trait_ref.args.has_non_region_param() {
        return None;
    }
    let self_ty = trait_ref.args.type_at(0);
    if item == LangItem::PolyasmCompilerCertificate
        && !matches!(self_ty.kind(), ty::FnDef(..) | ty::Closure(..))
    {
        return None;
    }
    if !matches!(self_ty.kind(), ty::FnDef(..) | ty::Closure(..)) {
        return explain_non_callable(tcx, self_ty, obligation_span);
    }
    let report = match *self_ty.kind() {
        ty::FnDef(def_id, args) => {
            analyze_fn_hir(tcx, def_id, args.no_bound_vars()?, &mut FxHashSet::default())?
        }
        ty::Closure(def_id, args) => {
            let report = analyze_closure_hir(tcx, def_id, args, &mut FxHashSet::default())?;
            return explain_report_rejection(
                tcx,
                trait_ref,
                negative,
                static_schedule,
                report,
                tcx.def_span(def_id),
            );
        }
        _ => unreachable!(),
    };
    let ty::FnDef(def_id, _) = *self_ty.kind() else { unreachable!() };
    let definition_span = tcx.def_span(def_id);

    explain_report_rejection(tcx, trait_ref, negative, static_schedule, report, definition_span)
}

fn explain_non_callable(
    tcx: TyCtxt<'_>,
    ty: Ty<'_>,
    obligation_span: Span,
) -> Option<WitnessFailure> {
    Some(match ty.kind() {
        ty::FnPtr(..) => WitnessFailure {
            message: "a function pointer has no body identity; pass the concrete function item or closure directly".to_owned(),
            span: obligation_span,
        },
        ty::Adt(definition, _) => WitnessFailure {
            message: "this type is not a callable function item or closure".to_owned(),
            span: tcx.def_span(definition.did()),
        },
        _ => WitnessFailure {
            message: "this value is not a concrete function item or closure".to_owned(),
            span: obligation_span,
        },
    })
}

fn explain_report_rejection<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    negative: bool,
    static_schedule: bool,
    report: Report,
    definition_span: Span,
) -> Option<WitnessFailure> {
    if static_schedule {
        let architecture = architecture(tcx, trait_ref.args.type_at(1))?;
        let requested_cycles = polyasm_u64_const(tcx, trait_ref.args.const_at(2))?;
        let Some(board) = static_clock_board(architecture) else {
            return Some(WitnessFailure {
                message: "exact static-clock witness requires a concrete registered FPGA board architecture; generic `Fpga` has no timing table".to_owned(),
                span: definition_span,
            });
        };
        if let Some(failure) = report.failure(STATIC_CLOCK) {
            return Some(WitnessFailure { message: failure.reason.to_owned(), span: failure.span });
        }
        return Some(WitnessFailure {
            message: format!(
                "the requested {requested_cycles}-cycle schedule is provisionally valid at {} Hz; its exact board-timed normalized MIR is checked during monomorphization",
                board.clock_hz()
            ),
            span: definition_span,
        });
    }

    let property = property(tcx, trait_ref.args.type_at(1))?;
    let architecture = architecture(tcx, trait_ref.args.type_at(2))?;
    let capability = match property {
        Property::Capability(capability) => capability,
        Property::StaticClock => STATIC_CLOCK,
    };
    let warrant = if accepts(architecture, capability) {
        report.warrant(capability)
    } else {
        Warrant::Disproven
    };
    if negative && warrant == Warrant::Proven {
        return Some(WitnessFailure {
            message: "the normalized callable body satisfies the requested property, so negative witness would be unsound".to_owned(),
            span: definition_span,
        });
    }
    if !accepts(architecture, capability) {
        return Some(WitnessFailure {
            message: "the requested property is not supported by this architecture".to_owned(),
            span: definition_span,
        });
    }
    report
        .failure(capability)
        .map(|failure| WitnessFailure { message: failure.reason.to_owned(), span: failure.span })
}

/// Trait selection runs while the callable's owner is under type checking.
/// Querying `optimized_mir` or resolving an `Instance` here would recursively
/// request that same typeck result. Function items and closures therefore use
/// this conservative HIR pre-check. Code generation resolves the exact
/// callable instance and rechecks its normalized MIR before erasing a
/// marker.
fn analyze_fn_hir<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: ty::GenericArgsRef<'tcx>,
    visiting: &mut FxHashSet<DefId>,
) -> Option<Report> {
    if tcx.is_intrinsic(def_id, sym::bswap) {
        return Some(bswap_report(tcx, def_id, args));
    }
    if tcx.intrinsic(def_id).is_some() {
        return Some(bodyless_callee_report(
            tcx.def_span(def_id),
            "this compiler intrinsic has no closed PolyASM lowering",
        ));
    }
    if tcx.is_foreign_item(def_id) {
        return Some(bodyless_callee_report(
            tcx.def_span(def_id),
            "this foreign function has no closed PolyASM lowering",
        ));
    }
    let local_def_id = def_id.as_local()?;
    if !visiting.insert(def_id) {
        return None;
    }
    let mut report = Report::new(1);
    let signature = tcx.fn_sig(def_id).instantiate(tcx, args).skip_norm_wip();
    let inputs_and_output = signature.inputs_and_output().skip_binder();
    for ty in inputs_and_output.iter() {
        reject_float_type(tcx, &mut report, tcx.def_span(def_id), ty);
        reject_pointer_accelerators_type(tcx, &mut report, tcx.def_span(def_id), ty);
        if !ty.has_non_region_param() && !(TypeInspection { tcx, ty }).is_drop_free() {
            defer_unknown_effects(
                &mut report,
                tcx.def_span(def_id),
                "a concrete parameter or return type may require unmodelled drop glue",
            );
        }
    }
    let body = tcx.hir_body_owned_by(local_def_id);
    let mut types = FxHashMap::default();
    for (parameter, ty) in body.params.iter().zip(signature.inputs().skip_binder()) {
        if !record_hir_pattern_type(parameter.pat, *ty, &mut types) && ty.has_non_region_param() {
            defer_unknown_effects(
                &mut report,
                parameter.span,
                "a destructured parametric argument has no owner-local binding type warrant",
            );
        }
    }
    HirAnalyzer { tcx, visiting, report: &mut report, provenance: FxHashMap::default(), types }
        .visit_expr(body.value);
    visiting.remove(&def_id);
    Some(report)
}

fn analyze_closure_hir<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: ty::GenericArgsRef<'tcx>,
    visiting: &mut FxHashSet<DefId>,
) -> Option<Report> {
    let local_def_id = def_id.as_local()?;
    if !visiting.insert(def_id) {
        return None;
    }
    // Normalized closure MIR always has a terminal control transfer in
    // addition to the operations represented by its HIR expression.
    let mut report = Report::new(1);
    let mut types = FxHashMap::default();
    if let Some((binding, ty)) = proven_local_closure_capture(tcx, def_id, args) {
        types.insert(binding, ty);
        disprove(
            &mut report,
            EBPF | XDP | LINUX_SAFE | STATIC_CLOCK | POINTER_ACCELERATORS,
            tcx.def_span(def_id),
            "the current closed device executor has no binder for a runtime closure environment",
        );
    } else if !args.as_closure().tupled_upvars_ty().is_unit() {
        defer_unknown_effects(
            &mut report,
            tcx.def_span(def_id),
            "a capturing closure contains an unresolved, borrowed, or non-primitive environment",
        );
    }
    for ty in args.as_closure().sig().inputs_and_output().skip_binder().iter() {
        reject_float_type(tcx, &mut report, tcx.def_span(def_id), ty);
        reject_pointer_accelerators_type(tcx, &mut report, tcx.def_span(def_id), ty);
        if !ty.has_non_region_param() && !(TypeInspection { tcx, ty }).is_drop_free() {
            defer_unknown_effects(
                &mut report,
                tcx.def_span(def_id),
                "a concrete closure input or return type may require unmodelled drop glue",
            );
        }
    }
    HirAnalyzer { tcx, visiting, report: &mut report, provenance: FxHashMap::default(), types }
        .visit_expr(tcx.hir_body_owned_by(local_def_id).value);
    visiting.remove(&def_id);
    Some(report)
}

fn proven_local_closure_capture<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: ty::GenericArgsRef<'tcx>,
) -> Option<(hir::HirId, Ty<'tcx>)> {
    let tupled_upvars = args.as_closure().tupled_upvars_ty();
    let ty::Tuple(captures) = tupled_upvars.kind() else {
        return None;
    };
    if captures.len() != 1 {
        return None;
    }
    let ty = captures[0];
    if !known_primitive_value(ty) || !(TypeInspection { tcx, ty }).is_drop_free() {
        return None;
    }
    let mentioned = tcx.upvars_mentioned(def_id)?;
    if mentioned.len() != 1 {
        return None;
    }
    mentioned.keys().next().copied().map(|binding| (binding, ty))
}

struct HirAnalyzer<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    visiting: &'a mut FxHashSet<DefId>,
    report: &'a mut Report,
    provenance: FxHashMap<hir::HirId, HirMemoryProvenance>,
    types: FxHashMap<hir::HirId, Ty<'tcx>>,
}

impl<'tcx> HirAnalyzer<'_, 'tcx> {
    fn expression_provenance(&self, expression: &hir::Expr<'_>) -> HirMemoryProvenance {
        match expression.kind {
            hir::ExprKind::AddrOf(_, mutability, place) => {
                self.place_provenance(place, mutability == hir::Mutability::Mut)
            }
            hir::ExprKind::Cast(inner, target) => {
                let source = self.expression_provenance(inner);
                let Some(target_pointee) = hir_pointer_pointee_type(self.tcx, target) else {
                    return HirMemoryProvenance::Unknown;
                };
                let Some(target_layout) = memory_layout(self.tcx, target_pointee) else {
                    return HirMemoryProvenance::Unknown;
                };
                let source_pointee = self
                    .expression_type(inner)
                    .and_then(|ty| ty.builtin_deref(true))
                    .or_else(|| self.addressed_type(inner));
                let retargeted = source.retarget(target_layout);
                if source_pointee == Some(target_pointee) {
                    retargeted
                } else {
                    retargeted.without_dereference_authority()
                }
            }
            hir::ExprKind::Type(inner, _) | hir::ExprKind::Use(inner, _) => {
                self.expression_provenance(inner)
            }
            hir::ExprKind::Block(block, _) if block.stmts.is_empty() => block
                .expr
                .map_or(HirMemoryProvenance::Unknown, |tail| self.expression_provenance(tail)),
            hir::ExprKind::If(_, then, Some(otherwise)) => {
                let then = self.expression_provenance(then);
                if then == self.expression_provenance(otherwise) {
                    then
                } else {
                    HirMemoryProvenance::Unknown
                }
            }
            hir::ExprKind::Path(hir::QPath::Resolved(_, path)) => match path.res {
                hir::def::Res::Local(binding) => {
                    self.provenance.get(&binding).copied().unwrap_or(HirMemoryProvenance::Unknown)
                }
                _ => HirMemoryProvenance::Unknown,
            },
            _ => HirMemoryProvenance::Unknown,
        }
    }

    fn place_provenance(&self, place: &hir::Expr<'_>, writable: bool) -> HirMemoryProvenance {
        match place.kind {
            hir::ExprKind::Path(hir::QPath::Resolved(_, path)) => match path.res {
                hir::def::Res::Local(_) => self
                    .expression_type(place)
                    .and_then(|ty| memory_layout(self.tcx, ty))
                    .map(|layout| {
                        HirMemoryProvenance::InBounds(MemoryAccess {
                            address_only: false,
                            allocation: layout,
                            offset: 0,
                            pointee: layout,
                            writable,
                        })
                    })
                    .unwrap_or(HirMemoryProvenance::Unknown),
                hir::def::Res::Def(DefKind::Static { .. }, def_id)
                    if !self.tcx.is_thread_local_static(def_id)
                        && !self.tcx.is_foreign_item(def_id) =>
                {
                    self.expression_type(place)
                        .and_then(|ty| memory_layout(self.tcx, ty))
                        .map(|layout| {
                            HirMemoryProvenance::FixedStatic(MemoryAccess {
                                address_only: false,
                                allocation: layout,
                                offset: 0,
                                pointee: layout,
                                writable,
                            })
                        })
                        .unwrap_or(HirMemoryProvenance::Unknown)
                }
                _ => HirMemoryProvenance::Unknown,
            },
            hir::ExprKind::Unary(hir::UnOp::Deref, pointer) => self
                .expression_type(place)
                .and_then(|ty| memory_layout(self.tcx, ty))
                .map(|pointee| {
                    self.expression_provenance(pointer).restrict_write(writable).retarget(pointee)
                })
                .unwrap_or(HirMemoryProvenance::Unknown),
            hir::ExprKind::Field(base, field) => self
                .expression_type(base)
                .and_then(|ty| field_projection(self.tcx, ty, field.name))
                .map(|(offset, _, pointee)| {
                    self.place_provenance(base, writable).project(offset, pointee)
                })
                .unwrap_or(HirMemoryProvenance::Unknown),
            _ => HirMemoryProvenance::Unknown,
        }
    }

    fn addressed_type(&self, expression: &hir::Expr<'_>) -> Option<Ty<'tcx>> {
        match expression.kind {
            hir::ExprKind::AddrOf(_, _, place) => self.expression_type(place),
            hir::ExprKind::Cast(inner, _)
            | hir::ExprKind::Type(inner, _)
            | hir::ExprKind::Use(inner, _) => self.addressed_type(inner),
            _ => None,
        }
    }

    fn is_explicit_endian_method(
        &self,
        segment: &hir::PathSegment<'_>,
        receiver: &hir::Expr<'_>,
    ) -> bool {
        if !self.tcx.is_polyasm_explicit_endian_name(segment.ident.name) {
            return false;
        }
        if let hir::def::Res::Def(DefKind::AssocFn, def_id) = segment.res
            && self.tcx.is_polyasm_explicit_endian_helper(def_id)
        {
            return true;
        }
        self.expression_type(receiver).is_some_and(|ty| {
            is_trusted_explicit_endian_primitive(self.tcx, ty, segment.ident.name)
        })
    }

    fn expression_type(&self, expression: &hir::Expr<'_>) -> Option<Ty<'tcx>> {
        match expression.kind {
            hir::ExprKind::Lit(literal) => match literal.node {
                ast::LitKind::Int(_, ast::LitIntType::Signed(integer)) => {
                    Some(Ty::new_int(self.tcx, integer))
                }
                ast::LitKind::Int(_, ast::LitIntType::Unsigned(integer)) => {
                    Some(Ty::new_uint(self.tcx, integer))
                }
                ast::LitKind::Int(_, ast::LitIntType::Unsuffixed) => Some(self.tcx.types.i32),
                ast::LitKind::Float(_, ast::LitFloatType::Suffixed(float)) => {
                    Some(Ty::new_float(self.tcx, float))
                }
                ast::LitKind::Float(_, ast::LitFloatType::Unsuffixed) => Some(self.tcx.types.f64),
                ast::LitKind::Bool(_) => Some(self.tcx.types.bool),
                ast::LitKind::Char(_) => Some(self.tcx.types.char),
                ast::LitKind::Byte(_) => Some(self.tcx.types.u8),
                _ => None,
            },
            hir::ExprKind::Path(hir::QPath::Resolved(_, path)) => match path.res {
                hir::def::Res::Local(binding) => self.types.get(&binding).copied(),
                hir::def::Res::Def(
                    DefKind::Const | DefKind::AssocConst | DefKind::Static { .. },
                    def_id,
                ) => Some(self.tcx.type_of(def_id).instantiate_identity().skip_norm_wip()),
                _ => None,
            },
            hir::ExprKind::Call(callee, arguments) => {
                let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = callee.kind else {
                    return None;
                };
                let hir::def::Res::Def(DefKind::Fn | DefKind::AssocFn, def_id) = path.res else {
                    return None;
                };
                if self.tcx.is_diagnostic_item(sym::polyasm_require_statement, def_id) {
                    let [argument] = arguments else { return None };
                    return self.expression_type(argument);
                }
                Some(
                    self.tcx
                        .fn_sig(def_id)
                        .instantiate_identity()
                        .skip_norm_wip()
                        .output()
                        .skip_binder(),
                )
            }
            hir::ExprKind::Binary(_, lhs, rhs) => {
                let lhs = self.expression_type(lhs)?;
                let rhs = self.expression_type(rhs)?;
                (known_primitive_value(lhs) && known_primitive_value(rhs)).then_some(lhs)
            }
            hir::ExprKind::Tup(fields) => {
                let fields = fields
                    .iter()
                    .map(|field| self.expression_type(field))
                    .collect::<Option<Vec<_>>>()?;
                Some(Ty::new_tup(self.tcx, &fields))
            }
            hir::ExprKind::Field(base, field) => self
                .expression_type(base)
                .and_then(|ty| field_projection(self.tcx, ty, field.name))
                .map(|(_, ty, _)| ty),
            hir::ExprKind::Struct(path, ..) => hir_struct_type(self.tcx, path),
            hir::ExprKind::AddrOf(_, mutability, place) => {
                Some(Ty::new_ptr(self.tcx, self.expression_type(place)?, mutability))
            }
            hir::ExprKind::Block(block, _) => {
                block.expr.and_then(|tail| self.expression_type(tail))
            }
            hir::ExprKind::If(_, then, Some(otherwise)) => {
                let then = self.expression_type(then)?;
                (self.expression_type(otherwise) == Some(then)).then_some(then)
            }
            hir::ExprKind::Unary(hir::UnOp::Deref, inner) => {
                self.expression_type(inner).and_then(|ty| ty.builtin_deref(true))
            }
            hir::ExprKind::Unary(_, inner) | hir::ExprKind::Use(inner, _) => {
                self.expression_type(inner)
            }
            hir::ExprKind::Cast(_, target) | hir::ExprKind::Type(_, target) => {
                hir_known_type(self.tcx, target)
            }
            _ => None,
        }
    }

    fn has_float_value(&self, expression: &hir::Expr<'_>) -> bool {
        if self.expression_type(expression).is_some_and(|ty| contains_float(self.tcx, ty)) {
            return true;
        }
        match expression.kind {
            hir::ExprKind::Lit(literal) => matches!(literal.node, ast::LitKind::Float(..)),
            hir::ExprKind::Binary(_, lhs, rhs) | hir::ExprKind::AssignOp(_, lhs, rhs) => {
                self.has_float_value(lhs) || self.has_float_value(rhs)
            }
            hir::ExprKind::Unary(_, inner)
            | hir::ExprKind::Cast(inner, _)
            | hir::ExprKind::Type(inner, _)
            | hir::ExprKind::Use(inner, _) => self.has_float_value(inner),
            _ => false,
        }
    }

    fn has_parametric_value(&self, expression: &hir::Expr<'_>) -> bool {
        if self.expression_type(expression).is_some_and(|ty| ty.has_non_region_param()) {
            return true;
        }
        match expression.kind {
            hir::ExprKind::Binary(_, lhs, rhs) => {
                self.has_parametric_value(lhs) || self.has_parametric_value(rhs)
            }
            hir::ExprKind::Unary(_, inner)
            | hir::ExprKind::Cast(inner, _)
            | hir::ExprKind::Type(inner, _)
            | hir::ExprKind::Use(inner, _) => self.has_parametric_value(inner),
            _ => false,
        }
    }

    fn write_place_is_proven(&self, place: &hir::Expr<'_>) -> bool {
        if !hir_place_has_deref(place) {
            return true;
        }
        let Some(pointee) = self.expression_type(place).and_then(|ty| memory_layout(self.tcx, ty))
        else {
            return false;
        };
        self.place_provenance(place, true)
            .access()
            .is_some_and(|access| access.permits_write(pointee))
    }

    fn analyze_direct_call(
        &mut self,
        callee: &hir::Expr<'_>,
        arguments: &[hir::Expr<'_>],
        span: Span,
    ) -> bool {
        if is_explicit_endian_primitive_path(self.tcx, callee) {
            disprove_schedule(
                self.report,
                span,
                "a sysroot byte-order conversion has no HIR-level exact clock model",
            );
            return false;
        }
        let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = callee.kind else {
            defer_unknown_effects(
                self.report,
                span,
                "an indirect call has no statically identifiable callee body",
            );
            return false;
        };
        let hir::def::Res::Def(kind, def_id) = path.res else {
            defer_unknown_effects(
                self.report,
                span,
                "an associated, constructed, or otherwise unresolved call has no owner-local warrant",
            );
            return false;
        };
        if is_polyasm_static_invoke_marker(self.tcx, def_id) {
            // Static selection is a provisional HIR identity. Its exact
            // callables, schedules, and selected effects are rechecked from
            // monomorphized MIR before code generation. Visiting both source
            // operands here would merge the deliberately unreachable slow
            // callable into the enclosing reading.
            return true;
        }
        if is_polyasm_compiler_marker(self.tcx, def_id) {
            // Compiler markers are semantic identities. Their arguments are
            // still visited, while the marker body stays outside the user's
            // function or statement reading.
            return false;
        }
        if self.tcx.is_diagnostic_item(sym::ptr_read_volatile, def_id)
            || self.tcx.is_intrinsic(def_id, sym::volatile_load)
        {
            let fixed_static_address =
                arguments.len() == 1 && self.expression_provenance(&arguments[0]).is_fixed_static();
            let supported_scalar = arguments.len() == 1
                && self.addressed_type(&arguments[0]).is_some_and(volatile_ebpf_type);
            if fixed_static_address && supported_scalar {
                disprove(
                    self.report,
                    XDP | LINUX_SAFE | POINTER_ACCELERATORS,
                    span,
                    "a fixed guest-memory volatile read is supported only by user-space eBPF",
                );
                return true;
            }
            if fixed_static_address {
                disprove(
                    self.report,
                    EBPF | XDP | LINUX_SAFE | POINTER_ACCELERATORS,
                    span,
                    "a volatile read requires a native-width 32-bit or 64-bit integer scalar",
                );
                return false;
            }
            defer_unknown_effects(
                self.report,
                span,
                "a volatile read requires one compiler-proven fixed static address",
            );
            return false;
        }
        if kind == DefKind::AssocFn && self.tcx.is_polyasm_explicit_endian_helper(def_id) {
            disprove_schedule(
                self.report,
                span,
                "a sysroot byte-order conversion has no HIR-level exact clock model",
            );
            return false;
        }
        if kind != DefKind::Fn {
            defer_unknown_effects(
                self.report,
                span,
                "an associated or constructed call has no owner-local warrant",
            );
            return false;
        }
        if def_id.as_local().is_none() {
            defer_unknown_effects(
                self.report,
                span,
                "only an owner-local free function can be resolved before owner type checking",
            );
            return false;
        }
        // A generic helper is inspected parametrically here. Any type-driven
        // operation remains fail-closed in HIR, while the concrete outer
        // callable signature is checked with its instantiated arguments and
        // normalized MIR is rechecked before code generation.
        let Some(callee_report) = analyze_fn_hir(
            self.tcx,
            def_id,
            ty::GenericArgs::identity_for_item(self.tcx, def_id),
            self.visiting,
        ) else {
            defer_unknown_effects(
                self.report,
                span,
                "a recursive or bodyless local call has no finite owner-local warrant",
            );
            return false;
        };
        merge_callee_report(self.report, callee_report, span);
        false
    }
}

impl<'v> Visitor<'v> for HirAnalyzer<'_, '_> {
    fn visit_stmt(&mut self, statement: &'v hir::Stmt<'v>) {
        match statement.kind {
            hir::StmtKind::Let(local) => {
                if local.ty.is_some_and(hir_type_contains_pointer) {
                    disprove(
                        self.report,
                        POINTER_ACCELERATORS,
                        local.span,
                        "pointer values are not admitted by pointer-accelerator lowering",
                    );
                }
                if let Some(initializer) = local.init {
                    self.visit_expr(initializer);
                    if let hir::PatKind::Binding(_, binding, _, None) = local.pat.kind {
                        let mut provenance = self.expression_provenance(initializer);
                        if local.ty.is_some_and(|ty| {
                            matches!(ty.kind, hir::TyKind::Ref(_, _, hir::Mutability::Not))
                        }) {
                            provenance = provenance.restrict_write(false);
                        }
                        self.provenance.insert(binding, provenance);
                        if let Some(ty) = self.expression_type(initializer) {
                            self.types.insert(binding, ty);
                        }
                    } else if self.has_parametric_value(initializer) {
                        defer_unknown_effects(
                            self.report,
                            local.pat.span,
                            "a destructured parametric local has no owner-local binding type warrant",
                        );
                    }
                }
                if let Some(else_block) = local.els {
                    self.visit_block(else_block);
                }
            }
            hir::StmtKind::Item(_) => {}
            hir::StmtKind::Expr(expression) | hir::StmtKind::Semi(expression) => {
                self.visit_expr(expression);
            }
        }
    }

    fn visit_expr(&mut self, expression: &'v hir::Expr<'v>) {
        if !matches!(expression.kind, hir::ExprKind::Block(..)) {
            add_cycles(self.report, 1);
        }
        if let hir::ExprKind::Assign(place, ..) | hir::ExprKind::AssignOp(_, place, _) =
            expression.kind
            && !self.write_place_is_proven(place)
        {
            disprove_invalid_pointer_write(self.report, expression.span);
        }
        if let hir::ExprKind::Assign(place, ..) = expression.kind
            && hir_place_has_deref(place)
            && self.expression_type(place).is_some_and(|ty| {
                TypeInspection { tcx: self.tcx, ty }.pointer_layout() != PointerLayout::Addressless
            })
        {
            // An indirect assignment overwrites a local pointer value through
            // an alias at times. HIR lacks an allocation identity to update
            // just that binding, so this drops the provisional facts. The
            // normalized-MIR pass below performs the corresponding precise
            // local-value clearing.
            self.provenance.clear();
        }
        let children_previsited = match expression.kind {
            hir::ExprKind::Binary(_, lhs, rhs) | hir::ExprKind::AssignOp(_, lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
                true
            }
            _ => false,
        };
        match expression.kind {
            hir::ExprKind::Call(callee, arguments) => {
                let skip_arguments = self.analyze_direct_call(callee, arguments, expression.span);
                self.visit_expr(callee);
                if !skip_arguments {
                    for argument in arguments {
                        self.visit_expr(argument);
                    }
                }
                return;
            }
            hir::ExprKind::MethodCall(segment, receiver, ..)
                if self.is_explicit_endian_method(segment, receiver) =>
            {
                disprove_schedule(
                    self.report,
                    expression.span,
                    "a primitive byte-order conversion has no HIR-level exact clock model",
                );
            }
            hir::ExprKind::Lit(literal) if matches!(literal.node, ast::LitKind::Float(..)) => {
                disprove(
                    self.report,
                    EBPF | XDP | VERILOG | LINUX_SAFE | P4,
                    expression.span,
                    "floating-point arithmetic is outside the eBPF/XDP/P4 integer subset",
                );
            }
            hir::ExprKind::AddrOf(..) => disprove(
                self.report,
                POINTER_ACCELERATORS,
                expression.span,
                "pointer values are not admitted by pointer-accelerator lowering",
            ),
            hir::ExprKind::Cast(source, target) => {
                if hir_type_contains_float(target) || self.has_float_value(source) {
                    disprove(
                        self.report,
                        EBPF | XDP | VERILOG | LINUX_SAFE | P4,
                        expression.span,
                        "a floating-point cast is outside the eBPF/XDP/P4 integer subset",
                    );
                } else if matches!(target.kind, hir::TyKind::Ptr(..)) {
                    disprove(
                        self.report,
                        POINTER_ACCELERATORS,
                        expression.span,
                        "pointer values are not admitted by pointer-accelerator lowering",
                    );
                    if self.expression_provenance(source) == HirMemoryProvenance::Unknown {
                        defer_unknown_effects(
                            self.report,
                            expression.span,
                            "this pointer cast has no compiler-proven local provenance",
                        );
                    }
                } else if !hir_type_is_integer(target) {
                    defer_unknown_effects(
                        self.report,
                        expression.span,
                        "this cast target has no primitive integer warrant before owner type checking",
                    );
                } else if self.has_parametric_value(source) {
                    defer_unknown_effects(
                        self.report,
                        expression.span,
                        "a cast from a parametric value has no owner-local concrete type warrant",
                    );
                }
            }
            hir::ExprKind::Binary(_, lhs, rhs)
                if self.has_float_value(lhs) || self.has_float_value(rhs) =>
            {
                disprove(
                    self.report,
                    EBPF | XDP | VERILOG | LINUX_SAFE | P4,
                    expression.span,
                    "floating-point arithmetic is outside the eBPF/XDP/P4 integer subset",
                );
            }
            hir::ExprKind::Binary(_, lhs, rhs)
                if self.has_parametric_value(lhs) || self.has_parametric_value(rhs) =>
            {
                defer_unknown_effects(
                    self.report,
                    expression.span,
                    "arithmetic on a parametric value has no owner-local concrete type warrant",
                );
            }
            hir::ExprKind::Binary(_, lhs, rhs)
                if !self.expression_type(lhs).is_some_and(known_primitive_value)
                    || !self.expression_type(rhs).is_some_and(known_primitive_value) =>
            {
                defer_unknown_effects(
                    self.report,
                    expression.span,
                    "an overloaded or unresolved binary operation has no primitive owner-local type warrant",
                );
            }
            hir::ExprKind::Binary(operation, ..)
                if matches!(operation.node, hir::BinOpKind::Div | hir::BinOpKind::Rem) =>
            {
                defer(
                    self.report,
                    STATIC_MEMORY | STATIC_MEMORY_UPPER,
                    expression.span,
                    "division or remainder may enter an unmodelled panic path before MIR normalization",
                );
                disprove(
                    self.report,
                    EBPF | XDP | LINUX_SAFE | P4,
                    expression.span,
                    "division and remainder are not admitted by the portable eBPF/XDP/P4 warrant",
                );
            }
            hir::ExprKind::Unary(_, operand) if self.has_float_value(operand) => {
                disprove(
                    self.report,
                    EBPF | XDP | VERILOG | LINUX_SAFE | P4,
                    expression.span,
                    "floating-point arithmetic is outside the eBPF/XDP/P4 integer subset",
                );
            }
            hir::ExprKind::Unary(hir::UnOp::Deref, pointer) => {
                disprove(
                    self.report,
                    POINTER_ACCELERATORS,
                    expression.span,
                    "pointer values are not admitted by pointer-accelerator lowering",
                );
                let provenance = self.expression_provenance(pointer);
                let pointee = self
                    .expression_type(pointer)
                    .and_then(|ty| ty.builtin_deref(true))
                    .and_then(|ty| memory_layout(self.tcx, ty));
                if provenance == HirMemoryProvenance::Unknown || pointee.is_none() {
                    defer_unknown_effects(
                        self.report,
                        expression.span,
                        "this dereference has no compiler-proven local provenance",
                    );
                } else if !provenance
                    .access()
                    .zip(pointee)
                    .is_some_and(|(access, pointee)| access.permits(pointee))
                {
                    disprove_invalid_pointer_layout(self.report, expression.span);
                } else if provenance.is_fixed_static() {
                    disprove(
                        self.report,
                        XDP | LINUX_SAFE,
                        expression.span,
                        "fixed guest memory is outside the Linux checker-owned eBPF stack",
                    );
                }
            }
            hir::ExprKind::Unary(_, operand) if self.has_parametric_value(operand) => {
                defer_unknown_effects(
                    self.report,
                    expression.span,
                    "a unary operation on a parametric value has no owner-local concrete type warrant",
                );
            }
            hir::ExprKind::Unary(_, operand)
                if !self.expression_type(operand).is_some_and(known_primitive_value) =>
            {
                defer_unknown_effects(
                    self.report,
                    expression.span,
                    "an overloaded or unresolved unary operation has no primitive owner-local type warrant",
                );
            }
            hir::ExprKind::AssignOp(_, lhs, rhs)
                if self.has_float_value(lhs) || self.has_float_value(rhs) =>
            {
                disprove(
                    self.report,
                    EBPF | XDP | VERILOG | LINUX_SAFE | P4,
                    expression.span,
                    "floating-point arithmetic is outside the eBPF/XDP/P4 integer subset",
                );
            }
            hir::ExprKind::AssignOp(_, lhs, rhs)
                if self.has_parametric_value(lhs) || self.has_parametric_value(rhs) =>
            {
                defer_unknown_effects(
                    self.report,
                    expression.span,
                    "compound assignment on a parametric value has no owner-local concrete type warrant",
                );
            }
            hir::ExprKind::Assign(lhs, _, _)
                if let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = lhs.kind
                    && let hir::def::Res::Local(binding) = path.res =>
            {
                self.provenance.insert(binding, HirMemoryProvenance::Unknown);
            }
            hir::ExprKind::Loop(..) | hir::ExprKind::Continue(..) => {
                disprove(
                    self.report,
                    EBPF | XDP | LINUX_SAFE,
                    expression.span,
                    "a source loop has no compiler-checked finite iteration bound for eBPF/XDP",
                );
                disprove_schedule(
                    self.report,
                    expression.span,
                    "a source loop prevents one exact static schedule without a compiler-checked iteration bound",
                );
            }
            hir::ExprKind::If(..) | hir::ExprKind::Match(..) | hir::ExprKind::Break(..) => {
                defer_schedule(
                    self.report,
                    expression.span,
                    "runtime control flow prevents one exact static schedule",
                )
            }
            hir::ExprKind::MethodCall(..)
            | hir::ExprKind::Become(..)
            | hir::ExprKind::InlineAsm(..)
            | hir::ExprKind::Yield(..)
            | hir::ExprKind::Err(..)
            | hir::ExprKind::DropTemps(..) => defer_unknown_effects(
                self.report,
                expression.span,
                "this expression has effects that cannot be proven before MIR normalization",
            ),
            hir::ExprKind::Index(..) => defer_unknown_effects(
                self.report,
                expression.span,
                "indexed memory has no pre-typeck in-bounds warrant",
            ),
            hir::ExprKind::Path(hir::QPath::Resolved(_, path))
                if let hir::def::Res::Def(DefKind::Static { .. }, def_id) = path.res =>
            {
                if self.tcx.is_thread_local_static(def_id) || self.tcx.is_foreign_item(def_id) {
                    defer_unknown_effects(
                        self.report,
                        expression.span,
                        "thread-local or foreign static memory has no closed device provenance",
                    );
                } else {
                    disprove(
                        self.report,
                        XDP | LINUX_SAFE,
                        expression.span,
                        "fixed guest memory is outside the Linux checker-owned eBPF stack",
                    );
                }
            }
            hir::ExprKind::Closure(..) => defer_unknown_effects(
                self.report,
                expression.span,
                "a nested closure has no independent pre-typeck hardware warrant",
            ),
            hir::ExprKind::Struct(..) => {
                if let Some(ty) = self.expression_type(expression) {
                    reject_float_type(self.tcx, self.report, expression.span, ty);
                    reject_pointer_accelerators_type(self.tcx, self.report, expression.span, ty);
                    if !(TypeInspection { tcx: self.tcx, ty }).is_drop_free() {
                        defer_unknown_effects(
                            self.report,
                            expression.span,
                            "a struct literal has unresolved fields, union storage, or unmodelled drop glue",
                        );
                    }
                } else {
                    defer_unknown_effects(
                        self.report,
                        expression.span,
                        "a struct literal has no concrete field layout before type checking",
                    );
                }
            }
            _ => {}
        }
        if !children_previsited {
            intravisit::walk_expr(self, expression);
        }
    }
}

fn record_hir_pattern_type<'tcx>(
    pattern: &hir::Pat<'_>,
    ty: Ty<'tcx>,
    types: &mut FxHashMap<hir::HirId, Ty<'tcx>>,
) -> bool {
    if let hir::PatKind::Binding(_, binding, _, None) = pattern.kind {
        types.insert(binding, ty);
        true
    } else {
        false
    }
}

fn is_explicit_endian_primitive_path(tcx: TyCtxt<'_>, expression: &hir::Expr<'_>) -> bool {
    let hir::ExprKind::Path(hir::QPath::TypeRelative(ty, segment)) = expression.kind else {
        return false;
    };
    let Some(ty) = hir_primitive_type(tcx, ty) else {
        return false;
    };
    is_trusted_explicit_endian_primitive(tcx, ty, segment.ident.name)
}

fn is_trusted_explicit_endian_primitive<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    name: Symbol,
) -> bool {
    if !is_endian_scalar(ty) || !tcx.is_polyasm_explicit_endian_name(name) {
        return false;
    }
    let Some(core) = tcx.lang_items().sized_trait().map(|item| item.krate) else {
        return false;
    };
    if !tcx.is_polyasm_sysroot_crate(core) {
        return false;
    }
    let Some(simplified) = simplify_type(tcx, ty, TreatParams::AsRigid) else {
        return false;
    };
    tcx.incoherent_impls(simplified).iter().any(|impl_def_id| {
        impl_def_id.krate == core
            && tcx
                .associated_items(*impl_def_id)
                .in_definition_order()
                .any(|item| matches!(item.kind, ty::AssocKind::Fn { .. }) && item.name() == name)
    })
}

fn is_endian_scalar(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), ty::Int(_) | ty::Uint(_) | ty::Float(_))
}

fn hir_type_contains_float(ty: &hir::Ty<'_>) -> bool {
    match ty.kind {
        hir::TyKind::Path(hir::QPath::Resolved(_, path)) => {
            matches!(path.res, hir::def::Res::PrimTy(hir::PrimTy::Float(_)))
        }
        hir::TyKind::Slice(element)
        | hir::TyKind::Array(element, _)
        | hir::TyKind::Ptr(element, _)
        | hir::TyKind::Ref(_, element, _)
        | hir::TyKind::Pat(element, _)
        | hir::TyKind::FieldOf(element, _) => hir_type_contains_float(element),
        hir::TyKind::Tup(fields) => fields.iter().any(hir_type_contains_float),
        _ => false,
    }
}

fn hir_type_contains_pointer(ty: &hir::Ty<'_>) -> bool {
    match ty.kind {
        hir::TyKind::Ptr(..) | hir::TyKind::Ref(..) => true,
        hir::TyKind::Slice(element)
        | hir::TyKind::Array(element, _)
        | hir::TyKind::Pat(element, _)
        | hir::TyKind::FieldOf(element, _) => hir_type_contains_pointer(element),
        hir::TyKind::Tup(fields) => fields.iter().any(hir_type_contains_pointer),
        _ => false,
    }
}

fn hir_place_has_deref(place: &hir::Expr<'_>) -> bool {
    match place.kind {
        hir::ExprKind::Unary(hir::UnOp::Deref, _) => true,
        hir::ExprKind::Field(base, _)
        | hir::ExprKind::Index(base, _, _)
        | hir::ExprKind::Type(base, _)
        | hir::ExprKind::Use(base, _) => hir_place_has_deref(base),
        _ => false,
    }
}

fn hir_type_is_integer(ty: &hir::Ty<'_>) -> bool {
    let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = ty.kind else {
        return false;
    };
    matches!(path.res, hir::def::Res::PrimTy(hir::PrimTy::Int(_) | hir::PrimTy::Uint(_)))
}

fn hir_primitive_type<'tcx>(tcx: TyCtxt<'tcx>, ty: &hir::Ty<'_>) -> Option<Ty<'tcx>> {
    let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = ty.kind else {
        return None;
    };
    match path.res {
        hir::def::Res::PrimTy(hir::PrimTy::Bool) => Some(tcx.types.bool),
        hir::def::Res::PrimTy(hir::PrimTy::Char) => Some(tcx.types.char),
        hir::def::Res::PrimTy(hir::PrimTy::Int(integer)) => Some(Ty::new_int(tcx, integer)),
        hir::def::Res::PrimTy(hir::PrimTy::Uint(integer)) => Some(Ty::new_uint(tcx, integer)),
        hir::def::Res::PrimTy(hir::PrimTy::Float(float)) => Some(Ty::new_float(tcx, float)),
        _ => None,
    }
}

fn hir_known_type<'tcx>(tcx: TyCtxt<'tcx>, ty: &hir::Ty<'_>) -> Option<Ty<'tcx>> {
    match ty.kind {
        hir::TyKind::Ptr(pointee, mutbl) => {
            Some(Ty::new_ptr(tcx, hir_known_type(tcx, pointee)?, mutbl))
        }
        _ => hir_primitive_type(tcx, ty),
    }
}

fn hir_pointer_pointee_type<'tcx>(tcx: TyCtxt<'tcx>, ty: &hir::Ty<'_>) -> Option<Ty<'tcx>> {
    match ty.kind {
        hir::TyKind::Ptr(pointee, _) | hir::TyKind::Ref(_, pointee, _) => {
            hir_known_type(tcx, pointee)
        }
        _ => None,
    }
}

fn memory_layout<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<MemoryLayout> {
    let layout = tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty)).ok()?;
    Some(MemoryLayout { align: layout.align.abi.bytes(), bytes: layout.size.bytes() })
}

fn field_projection<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    field: Symbol,
) -> Option<(u64, Ty<'tcx>, MemoryLayout)> {
    let field = match ty.kind() {
        ty::Tuple(_) => field.as_str().parse::<usize>().ok()?,
        ty::Adt(definition, _) if definition.is_struct() => definition
            .non_enum_variant()
            .fields
            .iter()
            .position(|definition| definition.name == field)?,
        _ => return None,
    };
    field_projection_index(tcx, ty, field)
}

fn field_projection_index<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    field: usize,
) -> Option<(u64, Ty<'tcx>, MemoryLayout)> {
    let field_ty = match ty.kind() {
        ty::Tuple(fields) => *fields.get(field)?,
        ty::Adt(definition, args) if definition.is_struct() => {
            definition.non_enum_variant().fields.iter().nth(field)?.ty(tcx, args).skip_norm_wip()
        }
        ty::Closure(_, args) => *args.as_closure().upvar_tys().get(field)?,
        _ => return None,
    };
    let layout = tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty)).ok()?;
    Some((layout.fields.offset(field).bytes(), field_ty, memory_layout(tcx, field_ty)?))
}

fn hir_struct_type<'tcx>(tcx: TyCtxt<'tcx>, path: &hir::QPath<'_>) -> Option<Ty<'tcx>> {
    let hir::QPath::Resolved(_, path) = path else {
        return None;
    };
    let hir::def::Res::Def(DefKind::Struct, def_id) = path.res else {
        return None;
    };
    let ty = tcx.type_of(def_id).instantiate_identity().skip_norm_wip();
    (!ty.has_non_region_param()).then_some(ty)
}

fn effective_alignment(access: MemoryAccess) -> u64 {
    if access.offset == 0 {
        access.allocation.align
    } else {
        access.allocation.align.min(1_u64 << access.offset.trailing_zeros())
    }
}

fn disprove_invalid_pointer_layout(report: &mut Report, span: Span) {
    disprove(
        report,
        EBPF | XDP
            | VERILOG
            | LINUX_SAFE
            | STATIC_MEMORY
            | STATIC_MEMORY_UPPER
            | STATIC_CLOCK
            | ACCELERATOR
            | CUDA
            | DPA
            | PTX
            | P4,
        span,
        "this dereference exceeds its allocation extent or violates its pointee alignment",
    );
    report.cycles = None;
}

fn disprove_invalid_pointer_write(report: &mut Report, span: Span) {
    disprove(
        report,
        EBPF | XDP
            | VERILOG
            | LINUX_SAFE
            | STATIC_MEMORY
            | STATIC_MEMORY_UPPER
            | STATIC_CLOCK
            | ACCELERATOR
            | CUDA
            | DPA
            | PTX
            | P4,
        span,
        "this write does not originate from compiler-proven mutable memory",
    );
    report.cycles = None;
}

fn known_primitive_value(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_))
}

fn reject_float_type<'tcx>(tcx: TyCtxt<'tcx>, report: &mut Report, span: Span, ty: Ty<'tcx>) {
    if contains_float(tcx, ty) {
        disprove(
            report,
            EBPF | XDP | VERILOG | LINUX_SAFE | P4,
            span,
            "floating-point values are not admitted by the eBPF/XDP/P4 integer subset",
        );
    }
}

fn reject_pointer_accelerators_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    report: &mut Report,
    span: Span,
    ty: Ty<'tcx>,
) {
    match (TypeInspection { tcx, ty }).pointer_layout() {
        PointerLayout::Addressless => {}
        PointerLayout::ContainsPointer => disprove(
            report,
            POINTER_ACCELERATORS,
            span,
            "pointer values are not admitted by pointer-accelerator lowering",
        ),
        PointerLayout::Unresolved => defer(
            report,
            POINTER_ACCELERATORS,
            span,
            "this value has no concrete field-by-field accelerator layout",
        ),
    }
}

fn analyze<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    visiting: &mut FxHashSet<DefId>,
) -> Option<Report> {
    analyze_mir(MirAnalysis {
        instance,
        packet_arguments: None,
        selection: None,
        slice_arguments: None,
        tcx,
        visiting,
    })
}

/// Reachability after substituting this instance's concrete type arguments.
/// Generic sysroot MIR still contains branches such as `T::IS_ZST` and
/// session-dependent UB checks. Their unreachable arms stay outside the
/// generated callable. Facts deliberately remain block-local: a value from
/// one predecessor stays silent about a condition reached through another
/// predecessor.
fn instantiated_reachable_blocks<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
) -> FxHashSet<BasicBlock> {
    let mut reachable = FxHashSet::default();
    let mut pending = vec![rustc_middle::mir::START_BLOCK];
    while let Some(block) = pending.pop() {
        if !reachable.insert(block) {
            continue;
        }
        let data = &body.basic_blocks[block];
        let mut known_boolean = vec![None; body.local_decls.len()];
        let mut known_nonzero = vec![None; body.local_decls.len()];
        let unknown_provenance = vec![MemoryProvenance::Unknown; body.local_decls.len()];
        for statement in &data.statements {
            match &statement.kind {
                StatementKind::Assign(assignment) => {
                    let (destination, value) = &**assignment;
                    if rvalue_moves_value(value) || !destination.projection.is_empty() {
                        known_boolean.fill(None);
                        known_nonzero.fill(None);
                    } else {
                        known_boolean[destination.local.as_usize()] = known_boolean_rvalue(
                            tcx,
                            instance,
                            value,
                            &unknown_provenance,
                            &known_boolean,
                            &known_nonzero,
                        );
                        known_nonzero[destination.local.as_usize()] =
                            known_nonzero_rvalue(tcx, instance, value, &known_nonzero);
                    }
                }
                StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                    known_boolean[local.as_usize()] = None;
                    known_nonzero[local.as_usize()] = None;
                }
                _ => {}
            }
        }
        if let TerminatorKind::SwitchInt { discr, targets } = &data.terminator().kind {
            let value = known_bool_operand(tcx, instance, discr, &known_boolean, &[])
                .map(u128::from)
                .or_else(|| constant_integer(tcx, instance, discr).map(|(value, _, _)| value));
            if let Some(value) = value {
                pending.push(targets.target_for_value(value));
                continue;
            }
        }
        pending.extend(data.terminator().successors());
    }
    reachable
}

#[derive(Copy, Clone, Eq, Hash, PartialEq)]
struct DependencyPoint {
    before: usize,
    block: BasicBlock,
    local: Local,
}

#[derive(Default)]
struct LocalDefinitions {
    locals: FxHashSet<Local>,
}

impl<'tcx> MirVisitor<'tcx> for LocalDefinitions {
    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        if context.is_place_assignment() {
            self.locals.insert(local);
        }
    }
}

#[derive(Default)]
struct LocalUses {
    locals: FxHashSet<Local>,
}

impl<'tcx> MirVisitor<'tcx> for LocalUses {
    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        if context.is_use() && !context.is_place_assignment() {
            self.locals.insert(local);
        }
    }
}

fn defines_local(statement: &Statement<'_>, location: Location, local: Local) -> bool {
    let mut definitions = LocalDefinitions::default();
    definitions.visit_statement(statement, location);
    definitions.locals.contains(&local)
}

fn terminator_defines_local(terminator: &Terminator<'_>, location: Location, local: Local) -> bool {
    let mut definitions = LocalDefinitions::default();
    definitions.visit_terminator(terminator, location);
    definitions.locals.contains(&local)
}

fn statement_partially_defines(statement: &Statement<'_>, local: Local) -> bool {
    match &statement.kind {
        StatementKind::Assign(assignment) => {
            assignment.0.local == local && !assignment.0.projection.is_empty()
        }
        StatementKind::SetDiscriminant { place, .. } => place.local == local,
        _ => false,
    }
}

fn enqueue_statement_dependencies(
    statement: &Statement<'_>,
    location: Location,
    pending: &mut VecDeque<DependencyPoint>,
) {
    let mut uses = LocalUses::default();
    uses.visit_statement(statement, location);
    enqueue_dependencies(uses, location, pending);
}

fn enqueue_terminator_dependencies(
    terminator: &Terminator<'_>,
    location: Location,
    pending: &mut VecDeque<DependencyPoint>,
) {
    let mut uses = LocalUses::default();
    uses.visit_terminator(terminator, location);
    enqueue_dependencies(uses, location, pending);
}

fn enqueue_dependencies(
    uses: LocalUses,
    location: Location,
    pending: &mut VecDeque<DependencyPoint>,
) {
    for local in sorted_locals(&uses.locals) {
        pending.push_back(DependencyPoint {
            before: location.statement_index,
            block: location.block,
            local,
        });
    }
}

#[allow(rustc::potential_query_instability)]
fn sorted_locals(locals: &FxHashSet<Local>) -> Vec<Local> {
    // Iteration order of the unordered set leaves the query result unchanged,
    // because every element is sorted by its stable MIR local index before the
    // worklist observes it.
    let mut locals = locals.iter().copied().collect::<Vec<_>>();
    locals.sort_unstable_by_key(|local| local.as_usize());
    locals
}

fn analyze_statement<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    selection: &PolyasmStatementSelection,
    visiting: &mut FxHashSet<DefId>,
) -> Option<Report> {
    analyze_mir(MirAnalysis {
        instance,
        packet_arguments: None,
        selection: Some(selection),
        slice_arguments: None,
        tcx,
        visiting,
    })
}

fn memory_provenance_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
) -> Vec<Option<Vec<MemoryProvenance>>> {
    let mut entries = vec![None; body.basic_blocks.len()];
    let mut initial = vec![MemoryProvenance::Unknown; body.local_decls.len()];
    for argument in body.args_iter() {
        let pointer = monomorphize(tcx, instance, body.local_decls[argument].ty);
        let ty::Ref(_, pointee, mutability) = *pointer.kind() else {
            continue;
        };
        let Some(layout) = memory_layout(tcx, pointee) else {
            continue;
        };
        // A Rust reference admits exactly one live, aligned pointee. Raw
        // pointers and dynamically sized references stay open-ended
        // here. Shared atomics permit writes to their UnsafeCell;
        // ordinary shared references remain read-only.
        let atomic = matches!(pointee.kind(), ty::Adt(definition, _)
            if tcx.is_diagnostic_item(sym::Atomic, definition.did()));
        initial[argument.as_usize()] = MemoryProvenance::Proven {
            access: MemoryAccess {
                address_only: false,
                allocation: layout,
                offset: 0,
                pointee: layout,
                writable: mutability == hir::Mutability::Mut || atomic,
            },
            origin: MemoryOrigin::Argument(argument),
        };
    }
    if let Some(environment) = proven_closure_environment(tcx, instance, body) {
        let pointer = monomorphize(tcx, instance, body.local_decls[environment].ty);
        if let Some(pointee) = pointer.builtin_deref(true).and_then(|ty| memory_layout(tcx, ty)) {
            initial[environment.as_usize()] = MemoryProvenance::Proven {
                access: MemoryAccess {
                    address_only: false,
                    allocation: pointee,
                    offset: 0,
                    pointee,
                    writable: pointer.is_mutable_ptr(),
                },
                origin: MemoryOrigin::ClosureEnvironment(instance.def_id()),
            };
        }
    }
    entries[START_BLOCK.as_usize()] = Some(initial);
    let mut pending = VecDeque::from([START_BLOCK]);
    while let Some(block) = pending.pop_front() {
        let Some(mut state) = entries[block.as_usize()].clone() else {
            continue;
        };
        let data = &body.basic_blocks[block];
        for statement in &data.statements {
            transfer_memory_statement(tcx, instance, body, statement, &mut state);
        }
        transfer_memory_terminator(&data.terminator().kind, &mut state);
        for successor in data.terminator().successors() {
            let entry = &mut entries[successor.as_usize()];
            let changed = if let Some(entry) = entry {
                meet_memory_provenance(entry, &state)
            } else {
                *entry = Some(state.clone());
                true
            };
            if changed {
                pending.push_back(successor);
            }
        }
    }
    entries
}

fn proven_closure_environment<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
) -> Option<Local> {
    if tcx.def_kind(instance.def_id()) != DefKind::Closure {
        return None;
    }
    let tupled_upvars = instance.args.as_closure().tupled_upvars_ty();
    let ty::Tuple(captures) = tupled_upvars.kind() else {
        return None;
    };
    if captures.len() != 1
        || !known_primitive_value(captures[0])
        || !(TypeInspection { tcx, ty: captures[0] }).is_drop_free()
    {
        return None;
    }
    body.args_iter().next()
}

fn transfer_memory_statement<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
    statement: &Statement<'tcx>,
    provenance: &mut [MemoryProvenance],
) {
    match &statement.kind {
        StatementKind::Assign(assignment) => {
            clear_indirect_local_value(assignment.0, provenance);
            if !assignment.0.projection.is_empty() {
                return;
            }
            let next = assigned_memory_provenance(
                tcx,
                instance,
                body,
                assignment.0,
                &assignment.1,
                provenance,
            );
            if rvalue_moves_value(&assignment.1) {
                provenance.fill(MemoryProvenance::Unknown);
            }
            provenance[assignment.0.local.as_usize()] = next;
        }
        StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
            clear_local_allocation(*local, provenance);
        }
        _ => {}
    }
}

fn transfer_memory_terminator(
    terminator: &TerminatorKind<'_>,
    provenance: &mut [MemoryProvenance],
) {
    match terminator {
        TerminatorKind::Call { args, destination, .. } => {
            if args.iter().any(|argument| operand_moves_value(&argument.node)) {
                provenance.fill(MemoryProvenance::Unknown);
            }
            clear_indirect_local_value(*destination, provenance);
            if destination.projection.is_empty() {
                provenance[destination.local.as_usize()] = MemoryProvenance::Unknown;
            }
        }
        TerminatorKind::Drop { place, .. } => {
            clear_local_allocation(place.local, provenance);
        }
        TerminatorKind::InlineAsm { .. } => provenance.fill(MemoryProvenance::Unknown),
        _ => {}
    }
}

fn rvalue_moves_value(value: &Rvalue<'_>) -> bool {
    match value {
        Rvalue::Use(operand, _)
        | Rvalue::Cast(_, operand, _)
        | Rvalue::WrapUnsafeBinder(operand, _)
        | Rvalue::UnaryOp(_, operand)
        | Rvalue::Repeat(operand, _) => operand_moves_value(operand),
        Rvalue::BinaryOp(_, operands) => {
            operand_moves_value(&operands.0) || operand_moves_value(&operands.1)
        }
        Rvalue::Aggregate(_, operands) => operands.iter().any(operand_moves_value),
        Rvalue::Ref(..)
        | Rvalue::RawPtr(..)
        | Rvalue::ThreadLocalRef(..)
        | Rvalue::Reborrow(..)
        | Rvalue::CopyForDeref(..)
        | Rvalue::Discriminant(..) => false,
    }
}

fn operand_moves_value(operand: &Operand<'_>) -> bool {
    matches!(operand, Operand::Move(_))
}

fn meet_memory_provenance(current: &mut [MemoryProvenance], incoming: &[MemoryProvenance]) -> bool {
    let mut changed = false;
    for (current, incoming) in current.iter_mut().zip(incoming) {
        if *current == *incoming || *current == MemoryProvenance::Unknown {
            continue;
        }
        let merged = match (*current, *incoming) {
            (
                MemoryProvenance::Proven { access, origin },
                MemoryProvenance::Proven { access: incoming_access, origin: incoming_origin },
            ) if access == incoming_access
                && matches!(origin, MemoryOrigin::Local(_) | MemoryOrigin::BranchMerged)
                && matches!(
                    incoming_origin,
                    MemoryOrigin::Local(_) | MemoryOrigin::BranchMerged
                ) =>
            {
                MemoryProvenance::Proven { access, origin: MemoryOrigin::BranchMerged }
            }
            _ => MemoryProvenance::Unknown,
        };
        if *current != merged {
            *current = merged;
            changed = true;
        }
    }
    changed
}

fn clear_local_allocation(local: Local, provenance: &mut [MemoryProvenance]) {
    for fact in provenance.iter_mut() {
        if matches!(
            *fact,
            MemoryProvenance::Proven { origin: MemoryOrigin::Local(origin), .. } if origin == local
        ) || matches!(*fact, MemoryProvenance::Proven { origin: MemoryOrigin::BranchMerged, .. })
        {
            *fact = MemoryProvenance::Unknown;
        }
    }
    provenance[local.as_usize()] = MemoryProvenance::Unknown;
}

fn clear_indirect_local_value(place: Place<'_>, provenance: &mut [MemoryProvenance]) {
    let mut projections = place.projection.iter();
    if !projections.next().is_some_and(|projection| matches!(projection, ProjectionElem::Deref))
        || !projections.all(|projection| {
            matches!(
                projection,
                ProjectionElem::OpaqueCast(_) | ProjectionElem::UnwrapUnsafeBinder(_)
            )
        })
    {
        return;
    }
    match local_memory_provenance(place, provenance) {
        MemoryProvenance::Proven { origin: MemoryOrigin::Local(local), .. } => {
            provenance[local.as_usize()] = MemoryProvenance::Unknown;
        }
        MemoryProvenance::Proven { origin: MemoryOrigin::BranchMerged, .. } => {
            provenance.fill(MemoryProvenance::Unknown);
        }
        MemoryProvenance::Proven { .. } | MemoryProvenance::Unknown => {}
    }
}

fn rvalue_memory_provenance<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
    value: &Rvalue<'tcx>,
    provenance: &[MemoryProvenance],
) -> MemoryProvenance {
    match value {
        Rvalue::Ref(_, kind, place) => place_address_provenance(
            tcx,
            instance,
            body,
            *place,
            provenance,
            matches!(kind, BorrowKind::Mut { .. }),
        ),
        Rvalue::RawPtr(kind, place) => place_address_provenance(
            tcx,
            instance,
            body,
            *place,
            provenance,
            *kind == RawPtrKind::Mut,
        ),
        Rvalue::Reborrow(_, mutability, place) => place_address_provenance(
            tcx,
            instance,
            body,
            *place,
            provenance,
            *mutability == hir::Mutability::Mut,
        ),
        Rvalue::Use(operand, _) => operand_memory_provenance(tcx, instance, operand, provenance),
        Rvalue::Cast(kind, operand, target) => {
            let source_provenance = operand_memory_provenance(tcx, instance, operand, provenance);
            let source_ty = monomorphize(tcx, instance, operand.ty(&body.local_decls, tcx));
            let target = monomorphize(tcx, instance, *target);
            let retarget_source = |target_pointee: Ty<'tcx>| {
                let mut pointee = source_ty.builtin_deref(true);
                while let Some(current) = pointee {
                    if current == target_pointee {
                        return match source_provenance {
                            MemoryProvenance::Proven { access, origin } => {
                                memory_layout(tcx, target_pointee)
                                    .and_then(|layout| access.project(0, layout))
                                    .map_or(MemoryProvenance::Unknown, |access| {
                                        MemoryProvenance::Proven { access, origin }
                                    })
                            }
                            MemoryProvenance::Unknown => MemoryProvenance::Unknown,
                        };
                    }
                    // A pointer to a struct and a pointer to its actual
                    // zero-offset first field have the same allocation and
                    // address. In particular this preserves the provenance
                    // of Atomic<T>'s UnsafeCell and aligned scalar storage.
                    pointee = field_projection_index(tcx, current, 0)
                        .filter(|(offset, _, _)| *offset == 0)
                        .map(|(_, field, _)| field);
                }
                source_provenance.without_dereference_authority()
            };
            let preserves_nonzero_address = || {
                memory_layout(tcx, source_ty)
                    .zip(memory_layout(tcx, target))
                    .is_some_and(|(source, target)| source.bytes <= target.bytes)
            };
            match kind {
                CastKind::PtrToPtr | CastKind::PointerCoercion(..) | CastKind::Subtype => target
                    .builtin_deref(true)
                    .and_then(|ty| memory_layout(tcx, ty).map(|layout| (ty, layout)))
                    .map_or(MemoryProvenance::Unknown, |(pointee, layout)| {
                        retarget_source(pointee).retarget(layout)
                    }),
                CastKind::Transmute | CastKind::BoxDerefTransmute
                    if target.builtin_deref(true).is_some() =>
                {
                    target
                        .builtin_deref(true)
                        .and_then(|ty| memory_layout(tcx, ty).map(|layout| (ty, layout)))
                        .map_or(MemoryProvenance::Unknown, |(pointee, layout)| {
                            retarget_source(pointee).retarget(layout)
                        })
                }
                CastKind::PointerExposeProvenance
                | CastKind::Transmute
                | CastKind::BoxDerefTransmute => {
                    if preserves_nonzero_address() {
                        source_provenance.without_dereference_authority()
                    } else {
                        MemoryProvenance::Unknown
                    }
                }
                CastKind::IntToInt if preserves_nonzero_address() => source_provenance,
                CastKind::IntToInt => MemoryProvenance::Unknown,
                CastKind::PointerWithExposedProvenance
                | CastKind::FloatToInt
                | CastKind::FloatToFloat
                | CastKind::IntToFloat
                | CastKind::FnPtrToPtr => MemoryProvenance::Unknown,
            }
        }
        Rvalue::WrapUnsafeBinder(..)
        | Rvalue::Repeat(..)
        | Rvalue::ThreadLocalRef(..)
        | Rvalue::BinaryOp(..)
        | Rvalue::UnaryOp(..)
        | Rvalue::Aggregate(..)
        | Rvalue::CopyForDeref(..)
        | Rvalue::Discriminant(..) => MemoryProvenance::Unknown,
    }
}

fn assigned_memory_provenance<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
    destination: Place<'tcx>,
    value: &Rvalue<'tcx>,
    provenance: &[MemoryProvenance],
) -> MemoryProvenance {
    let next = rvalue_memory_provenance(tcx, instance, body, value, provenance);
    let destination_ty = monomorphize(tcx, instance, destination.ty(&body.local_decls, tcx).ty);
    if destination_ty.ref_mutability() == Some(hir::Mutability::Not) {
        next.restrict_write(false)
    } else {
        next
    }
}

struct MirAnalysis<'a, 'tcx> {
    instance: Instance<'tcx>,
    packet_arguments: Option<&'a packet::Arguments>,
    selection: Option<&'a PolyasmStatementSelection>,
    slice_arguments: Option<&'a slices::Arguments>,
    tcx: TyCtxt<'tcx>,
    visiting: &'a mut FxHashSet<DefId>,
}

fn analyze_mir(analysis: MirAnalysis<'_, '_>) -> Option<Report> {
    let MirAnalysis { instance, packet_arguments, selection, slice_arguments, tcx, visiting } =
        analysis;
    if matches!(instance.def, ty::InstanceKind::Virtual(..)) {
        return Some(bodyless_callee_report(
            tcx.def_span(instance.def_id()),
            "a virtual call has no statically identified implementation body",
        ));
    }
    if tcx.is_intrinsic(instance.def_id(), sym::is_val_statically_known) {
        // This optimizer query yields a boolean and stays clear of memory and
        // of user code. Its result is deliberately unspecified, so a
        // reading checks both successors of every branch that consumes it.
        // The query leaves every hardware schedule open.
        let mut report = Report::new(1);
        disprove(
            &mut report,
            ALL & !WASM,
            tcx.def_span(instance.def_id()),
            "an optimizer-dependent query has no target-independent hardware schedule",
        );
        return Some(report);
    }
    if tcx.is_intrinsic(instance.def_id(), sym::ptr_offset_from)
        || tcx.is_intrinsic(instance.def_id(), sym::ptr_offset_from_unsigned)
    {
        // These are integer address subtraction followed by division by one
        // concrete, nonzero element size. They stay clear of memory and of
        // foreign implementations. Wasm retains checks on subsequent
        // accesses independently of this pointer arithmetic.
        let layout = memory_layout(tcx, instance.args.type_at(0))?;
        if layout.bytes == 0 {
            return Some(bodyless_callee_report(
                tcx.def_span(instance.def_id()),
                "pointer distance requires a nonzero-sized element type",
            ));
        }
        let mut report = Report::new(2);
        disprove(
            &mut report,
            ALL & !WASM,
            tcx.def_span(instance.def_id()),
            "dynamic pointer distance has no fixed memory provenance for hardware warrant",
        );
        return Some(report);
    }
    if tcx.is_intrinsic(instance.def_id(), sym::abort) {
        let mut report = Report::new(1);
        disprove(
            &mut report,
            ALL & !WASM,
            tcx.def_span(instance.def_id()),
            "this abort requires the Wasm trap ABI",
        );
        return Some(report);
    }
    if tcx.is_intrinsic(instance.def_id(), sym::caller_location) {
        // rustc materializes this location in immutable module data. Wasm
        // addresses that data through its ordinary linear-memory ABI, with
        // zero external calls and zero runtime allocation.
        let mut report = Report::new(1);
        disprove(
            &mut report,
            ALL & !WASM,
            tcx.def_span(instance.def_id()),
            "caller location requires the Wasm module-data ABI",
        );
        return Some(report);
    }
    if tcx.is_intrinsic(instance.def_id(), sym::bswap) {
        return Some(bswap_report(tcx, instance.def_id(), instance.args));
    }
    if tcx.intrinsic(instance.def_id()).is_some() {
        return Some(bodyless_callee_report(
            tcx.def_span(instance.def_id()),
            "this compiler intrinsic has no closed PolyASM lowering",
        ));
    }
    if tcx.is_foreign_item(instance.def_id()) {
        if find_attr!(tcx, instance.def_id(), Lang(item) => *item) == Some(LangItem::PanicImpl)
            && let Some(handler) = tcx.lang_items().panic_impl()
            && !tcx.is_foreign_item(handler)
        {
            // The compiler resolves core's weak panic declaration to the
            // actual Rust panic handler. This checks that same concrete body;
            // its apparent extern declaration stays an internal call.
            return analyze(tcx, Instance::mono(tcx, handler), visiting);
        }
        return Some(bodyless_callee_report(
            tcx.def_span(instance.def_id()),
            "this foreign function has no closed PolyASM lowering",
        ));
    }
    if !visiting.insert(instance.def_id()) {
        return None;
    }
    let body = tcx.instance_mir(instance.def);
    if let Some(dumper) = rustc_middle::mir::pretty::MirDumper::new(tcx, "polyasm-warrant", body) {
        dumper.dump_mir(body);
    }
    let mut report = Report::new(0);
    let mut matched = selection.is_none();
    let reachable = instantiated_reachable_blocks(tcx, instance, body);
    let components =
        rustc_data_structures::graph::scc::Sccs::<BasicBlock, BasicBlock>::new(&body.basic_blocks);
    let slice_witness =
        slices::analyze(slices::Analysis { arguments: slice_arguments, body, instance, tcx });
    let packet_witness =
        packet::analyze(packet::Analysis { arguments: packet_arguments, body, instance, tcx });
    let provenance_entries = memory_provenance_entries(tcx, instance, body);
    let implicit_unit_closure_environment = if selection.is_none()
        && tcx.def_kind(instance.def_id()) == DefKind::Closure
        && instance.args.as_closure().tupled_upvars_ty().is_unit()
    {
        body.args_iter().next()
    } else {
        None
    };
    if selection.is_none() && tcx.def_kind(instance.def_id()) == DefKind::Closure {
        if proven_closure_environment(tcx, instance, body).is_some() {
            disprove(
                &mut report,
                EBPF | XDP | LINUX_SAFE | STATIC_CLOCK | POINTER_ACCELERATORS,
                body.span,
                "the current closed device executor has no binder for a runtime closure environment",
            );
        } else if !instance.args.as_closure().tupled_upvars_ty().is_unit() {
            disprove_unknown_effects(
                &mut report,
                body.span,
                "the closure environment has no compiler-proven local value layout",
            );
        }
    }
    for (local, declaration) in body.local_decls.iter_enumerated() {
        let ty = monomorphize(tcx, instance, declaration.ty);
        if selection.is_none() && contains_float(tcx, ty) {
            disprove(
                &mut report,
                EBPF | XDP | VERILOG | LINUX_SAFE | P4,
                declaration.source_info.span,
                "floating-point values are not admitted by the eBPF/XDP/P4 integer subset",
            );
        }
        if selection.is_none() && Some(local) != implicit_unit_closure_environment {
            // A non-capturing closure still has one rust-call ABI self
            // argument. It is a compiler-generated reference to a zero-sized
            // environment, distinct from a pointer value in the guest program.
            reject_pointer_accelerators_type(tcx, &mut report, declaration.source_info.span, ty);
        }
    }
    for (block, data) in body.basic_blocks.iter_enumerated() {
        if !reachable.contains(&block) {
            continue;
        }
        // These scalar facts justify removal of an assertion edge. Unlike
        // memory provenance, they intentionally stay within a basic block
        // until they have an all-predecessor dataflow meet. Keeping them
        // block-local is conservative and keeps the traversal order of two
        // branch arms from settling an assertion at their join.
        let mut known_boolean = vec![None; body.local_decls.len()];
        let mut known_nonzero = vec![None; body.local_decls.len()];
        let mut known_overflow = vec![None; body.local_decls.len()];
        let mut provenance = provenance_entries[block.as_usize()]
            .clone()
            .unwrap_or_else(|| vec![MemoryProvenance::Unknown; body.local_decls.len()]);
        for (statement_index, statement) in data.statements.iter().enumerate() {
            let location = Location { block, statement_index };
            let span = statement.source_info.span;
            let selected = selection.is_none_or(|selection| selection.contains(location));
            matched |= selected;
            match &statement.kind {
                StatementKind::Assign(assignment) => {
                    let (destination, value) = &**assignment;
                    if selected {
                        let destination_ty =
                            monomorphize(tcx, instance, destination.ty(&body.local_decls, tcx).ty);
                        reject_float_type(tcx, &mut report, span, destination_ty);
                        reject_pointer_accelerators_type(tcx, &mut report, span, destination_ty);
                        add_timing(
                            &mut report,
                            timing_class_for_rvalue(tcx, instance, value),
                            span,
                        );
                    }
                    let mut ignored = Report::new(0);
                    let statement_report = if selected { &mut report } else { &mut ignored };
                    analyze_place_write(
                        tcx,
                        instance,
                        body,
                        *destination,
                        span,
                        &provenance,
                        statement_report,
                    );
                    if destination.projection.is_empty() {
                        known_overflow[destination.local.as_usize()] =
                            constant_overflow(tcx, instance, value);
                    }
                    let mut memory_report = Report::new(0);
                    analyze_rvalue(
                        tcx,
                        instance,
                        *destination,
                        value,
                        span,
                        body,
                        &mut provenance,
                        &mut memory_report,
                    );
                    if slice_witness.memory(location) || packet_witness.memory(location) {
                        // Relational byte-slice facts establish these
                        // accesses from the caller's live allocation and a
                        // dominating bounds check. They grant zero fixed-size
                        // allocations and zero exact clock claims.
                        let capabilities = EBPF | XDP | LINUX_SAFE;
                        memory_report.capabilities |= capabilities;
                        memory_report.disproven &= !capabilities;
                        for capability in [EBPF, XDP, LINUX_SAFE] {
                            memory_report.failures[capability.trailing_zeros() as usize] = None;
                        }
                    }
                    if memory_report.cycles.is_none() {
                        statement_report.cycles = None;
                    }
                    statement_report.merge(memory_report);
                    if rvalue_moves_value(value) {
                        known_boolean.fill(None);
                        known_nonzero.fill(None);
                        known_overflow.fill(None);
                    } else if destination.projection.is_empty() {
                        known_boolean[destination.local.as_usize()] = known_boolean_rvalue(
                            tcx,
                            instance,
                            value,
                            &provenance,
                            &known_boolean,
                            &known_nonzero,
                        );
                        known_nonzero[destination.local.as_usize()] =
                            known_nonzero_rvalue(tcx, instance, value, &known_nonzero);
                    } else {
                        known_boolean[destination.local.as_usize()] = None;
                        known_nonzero[destination.local.as_usize()] = None;
                        known_overflow[destination.local.as_usize()] = None;
                    }
                }
                StatementKind::SetDiscriminant { place, .. } => {
                    if selected {
                        analyze_place_write(
                            tcx,
                            instance,
                            body,
                            **place,
                            span,
                            &provenance,
                            &mut report,
                        );
                        add_timing(&mut report, TimingClass::Control, span);
                    }
                }
                StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                    clear_local_allocation(*local, &mut provenance);
                    known_boolean[local.as_usize()] = None;
                    known_nonzero[local.as_usize()] = None;
                    known_overflow[local.as_usize()] = None;
                }
                StatementKind::ConstEvalCounter | StatementKind::Nop => {}
                StatementKind::PlaceMention(place) => {
                    if selected {
                        analyze_place(tcx, instance, body, **place, span, &provenance, &mut report);
                    }
                }
                StatementKind::FakeRead(..)
                | StatementKind::AscribeUserType(..)
                | StatementKind::Coverage(..)
                | StatementKind::Intrinsic(..)
                | StatementKind::BackwardIncompatibleDropHint { .. } => {
                    if selected {
                        disprove_unknown_effects(
                            &mut report,
                            span,
                            "this MIR statement has no closed PolyASM effect model",
                        );
                    }
                }
            }
        }
        let terminator = data.terminator();
        let terminator_span = terminator.source_info.span;
        let location = Location { block, statement_index: data.statements.len() };
        if selection.is_some_and(|selection| selection.marker() == location) {
            matched = true;
            if selection.is_some_and(PolyasmStatementSelection::is_direct)
                && let TerminatorKind::Call { args, .. } = &terminator.kind
                && let [argument] = &args[..]
            {
                let operand = &argument.node;
                let ty = monomorphize(tcx, instance, operand.ty(&body.local_decls, tcx));
                reject_float_type(tcx, &mut report, argument.span, ty);
                reject_pointer_accelerators_type(tcx, &mut report, argument.span, ty);
                analyze_operand(
                    tcx,
                    instance,
                    body,
                    operand,
                    argument.span,
                    &provenance,
                    &mut report,
                );
                add_timing(
                    &mut report,
                    if matches!(operand, Operand::Constant(_)) {
                        TimingClass::Constant
                    } else {
                        TimingClass::Integer
                    },
                    argument.span,
                );
            }
            continue;
        }
        if selection.is_some_and(|selection| !selection.contains(location)) {
            continue;
        }
        matched = true;
        match &terminator.kind {
            TerminatorKind::Goto { target } => {
                if components.scc(*target) == components.scc(block) {
                    disprove(
                        &mut report,
                        EBPF | XDP | LINUX_SAFE,
                        terminator_span,
                        "a backward control-flow edge has no compiler-checked finite iteration bound for eBPF/XDP",
                    );
                    disprove_schedule(
                        &mut report,
                        terminator_span,
                        "a backward control-flow edge makes this schedule dynamic",
                    );
                } else {
                    add_timing(&mut report, TimingClass::Control, terminator_span);
                }
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                analyze_operand(
                    tcx,
                    instance,
                    body,
                    discr,
                    terminator_span,
                    &provenance,
                    &mut report,
                );
                add_timing(&mut report, TimingClass::Control, terminator_span);
                if targets
                    .all_targets()
                    .iter()
                    .any(|target| components.scc(*target) == components.scc(block))
                {
                    disprove(
                        &mut report,
                        EBPF | XDP | LINUX_SAFE,
                        terminator_span,
                        "a backward branch has no compiler-checked finite iteration bound for eBPF/XDP",
                    );
                }
                disprove_schedule(
                    &mut report,
                    terminator_span,
                    "a runtime branch prevents one exact static schedule",
                );
                let _ = targets;
            }
            TerminatorKind::Call { func, args, destination, target, .. } => {
                let direct_callee = func.const_fn_def();
                if let Some((def_id, call_args)) = direct_callee
                    && is_polyasm_static_invoke_marker(tcx, def_id)
                {
                    add_timing(&mut report, TimingClass::Call, terminator_span);
                    analyze_place_write(
                        tcx,
                        instance,
                        body,
                        *destination,
                        terminator_span,
                        &provenance,
                        &mut report,
                    );
                    let call_args = monomorphize(tcx, instance, call_args);
                    let selection =
                        if tcx.is_diagnostic_item(sym::polyasm_invoke_static_faster_on, def_id) {
                            polyasm_static_callable_selection_on_with_visiting(
                                tcx,
                                call_args,
                                terminator_span,
                                visiting,
                            )
                            .ok()
                        } else {
                            polyasm_static_callable_selection_with_visiting(
                                tcx,
                                call_args,
                                terminator_span,
                                visiting,
                            )
                        };
                    if let Some(selection) = selection
                        && let Some(callee_report) = analyze(tcx, selection.selected(), visiting)
                    {
                        merge_callee_report(&mut report, callee_report, terminator_span);
                    } else {
                        disprove_unknown_effects(
                            &mut report,
                            terminator_span,
                            "the exact static selection did not retain two closed compiler-checked schedules",
                        );
                    }
                    if target.is_some_and(|target| components.scc(target) == components.scc(block))
                    {
                        disprove(
                            &mut report,
                            EBPF | XDP | LINUX_SAFE,
                            terminator_span,
                            "a backward call edge has no compiler-checked finite iteration bound for eBPF/XDP",
                        );
                    }
                    if target.is_none_or(|target| components.scc(target) == components.scc(block)) {
                        disprove_schedule(
                            &mut report,
                            terminator_span,
                            "this call does not have a forward-returning control-flow edge",
                        );
                    }
                    continue;
                }
                if direct_callee.is_some_and(|(def_id, _)| is_polyasm_compiler_marker(tcx, def_id))
                {
                    continue;
                }
                // A call instruction has one deterministic control-transfer
                // tick in addition to the complete transitive callee cost.
                add_timing(
                    &mut report,
                    direct_callee
                        .filter(|(def_id, _)| is_atomic_operation(tcx, *def_id))
                        .map_or(TimingClass::Call, |_| TimingClass::Atomic),
                    terminator_span,
                );
                analyze_place_write(
                    tcx,
                    instance,
                    body,
                    *destination,
                    terminator_span,
                    &provenance,
                    &mut report,
                );
                for argument in args {
                    analyze_operand(
                        tcx,
                        instance,
                        body,
                        &argument.node,
                        terminator_span,
                        &provenance,
                        &mut report,
                    );
                }
                let Some((def_id, call_args)) = direct_callee else {
                    disprove_unknown_effects(
                        &mut report,
                        terminator_span,
                        "an indirect call has no statically identifiable callee body",
                    );
                    continue;
                };
                let call_args = monomorphize(tcx, instance, call_args);
                if packet_witness.intrinsic(block) {
                    disprove(
                        &mut report,
                        ALL & !(EBPF | XDP | LINUX_SAFE),
                        terminator_span,
                        "this packet instruction requires a live packet-machine context",
                    );
                } else if packet_witness.rejected(block) {
                    disprove_unknown_effects(
                        &mut report,
                        terminator_span,
                        "this packet instruction lacks its original window or a Linux-supported constant prefix",
                    );
                } else if tcx.is_diagnostic_item(sym::ptr_read_volatile, def_id)
                    || tcx.is_intrinsic(def_id, sym::volatile_load)
                {
                    let fixed_static_address = args.len() == 1
                        && operand_memory_provenance(tcx, instance, &args[0].node, &provenance)
                            .is_fixed_static();
                    if fixed_static_address && volatile_ebpf_type(call_args.type_at(0)) {
                        disprove(
                            &mut report,
                            XDP | LINUX_SAFE | POINTER_ACCELERATORS,
                            terminator_span,
                            "a fixed guest-memory volatile read is supported only by user-space eBPF",
                        );
                    } else {
                        disprove_unknown_effects(
                            &mut report,
                            terminator_span,
                            if fixed_static_address {
                                "a volatile read requires a native-width 32-bit or 64-bit integer scalar"
                            } else {
                                "a volatile read requires one compiler-proven fixed static address"
                            },
                        );
                    }
                } else if tcx.is_intrinsic(def_id, sym::atomic_xadd) {
                    let value_ty = call_args.type_at(0);
                    let access = args.first().and_then(|argument| {
                        operand_memory_provenance(tcx, instance, &argument.node, &provenance)
                            .access()
                    });
                    let supported_integer = matches!(
                        value_ty.kind(),
                        ty::Uint(ty::UintTy::U32 | ty::UintTy::U64)
                            | ty::Int(ty::IntTy::I32 | ty::IntTy::I64)
                    );
                    let writable = memory_layout(tcx, value_ty)
                        .zip(access)
                        .is_some_and(|(layout, access)| access.permits_write(layout));
                    if args.len() != 2
                        || !supported_integer
                        || call_args.type_at(1) != value_ty
                        || !writable
                    {
                        disprove_unknown_effects(
                            &mut report,
                            terminator_span,
                            "atomic addition requires one writable, aligned 32-bit or 64-bit integer allocation",
                        );
                    } else {
                        disprove(
                            &mut report,
                            EBPF | XDP | LINUX_SAFE | POINTER_ACCELERATORS,
                            terminator_span,
                            "this atomic linear-memory operation requires the Wasm, native, or Verilog memory ABI",
                        );
                    }
                } else if let Ok(Some(callee)) = Instance::try_resolve(
                    tcx,
                    ty::TypingEnv::fully_monomorphized(),
                    def_id,
                    call_args,
                ) {
                    if tcx.is_polyasm_explicit_endian_helper(callee.def_id()) {
                        disprove_schedule(
                            &mut report,
                            terminator_span,
                            "a sysroot byte-order conversion has no target-independent clock model",
                        );
                    } else if let Some(callee_report) = analyze_mir(MirAnalysis {
                        instance: callee,
                        packet_arguments: packet_witness.arguments(block),
                        selection: None,
                        slice_arguments: slice_witness.arguments(block),
                        tcx,
                        visiting,
                    }) {
                        merge_callee_report(&mut report, callee_report, terminator_span);
                    } else {
                        disprove_unknown_effects(
                            &mut report,
                            terminator_span,
                            "recursive or unresolved call graphs have no finite static schedule",
                        );
                    }
                } else {
                    disprove_unknown_effects(
                        &mut report,
                        terminator_span,
                        "the called function cannot be resolved after monomorphization",
                    );
                }
                if target.is_some_and(|target| components.scc(target) == components.scc(block)) {
                    disprove(
                        &mut report,
                        EBPF | XDP | LINUX_SAFE,
                        terminator_span,
                        "a backward call edge has no compiler-checked finite iteration bound for eBPF/XDP",
                    );
                }
                if target.is_none_or(|target| components.scc(target) == components.scc(block)) {
                    disprove_schedule(
                        &mut report,
                        terminator_span,
                        "this call does not have a forward-returning control-flow edge",
                    );
                }
            }
            TerminatorKind::Assert { cond, expected, target, .. } => {
                analyze_operand(
                    tcx,
                    instance,
                    body,
                    cond,
                    terminator_span,
                    &provenance,
                    &mut report,
                );
                add_timing(&mut report, TimingClass::Control, terminator_span);
                let condition =
                    known_bool_operand(tcx, instance, cond, &known_boolean, &known_overflow);
                let pointer_check_proven = match &terminator.kind {
                    TerminatorKind::Assert {
                        msg: AssertKind::MisalignedPointerDereference { required, found },
                        ..
                    } => constant_integer(tcx, instance, required)
                        .and_then(|(required, _, _)| u64::try_from(required).ok())
                        .is_some_and(|required| {
                            let provenance =
                                operand_memory_provenance(tcx, instance, found, &provenance);
                            required != 0
                                && provenance
                                    .access()
                                    .is_some_and(|access| effective_alignment(access) >= required)
                        }),
                    _ => false,
                };
                if condition != Some(*expected)
                    && !pointer_check_proven
                    && !slice_witness.assertion(block)
                    && !packet_witness.assertion(block)
                {
                    disprove(
                        &mut report,
                        ALL & !WASM,
                        terminator_span,
                        "an unresolved or failing assertion may enter an unmodelled panic path",
                    );
                } else if components.scc(*target) == components.scc(block) {
                    disprove(
                        &mut report,
                        EBPF | XDP | LINUX_SAFE,
                        terminator_span,
                        "a backward assertion edge has no compiler-checked finite iteration bound for eBPF/XDP",
                    );
                    disprove_schedule(
                        &mut report,
                        terminator_span,
                        "a backward assertion edge prevents one exact static schedule",
                    );
                }
            }
            TerminatorKind::Return => {
                add_timing(&mut report, TimingClass::Control, terminator_span)
            }
            TerminatorKind::Unreachable => {
                add_timing(&mut report, TimingClass::Control, terminator_span)
            }
            TerminatorKind::Drop { place, target, .. }
                if !monomorphize(tcx, instance, place.ty(&body.local_decls, tcx).ty)
                    .needs_drop(tcx, ty::TypingEnv::fully_monomorphized()) =>
            {
                // Generic iterator MIR retains a drop terminator for its
                // callable until monomorphization. A concrete drop-free
                // closure contributes only its forward control edge.
                add_timing(&mut report, TimingClass::Control, terminator_span);
                if components.scc(*target) == components.scc(block) {
                    disprove(
                        &mut report,
                        EBPF | XDP | LINUX_SAFE | STATIC_CLOCK,
                        terminator_span,
                        "a backward drop-free edge has no compiler-checked finite iteration bound",
                    );
                }
            }
            TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(..)
            | TerminatorKind::Drop { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::InlineAsm { .. }
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::Yield { .. } => disprove_unknown_effects(
                &mut report,
                terminator_span,
                "this control-flow operation has effects outside the PolyASM warrant model",
            ),
        }
    }
    visiting.remove(&instance.def_id());
    matched.then_some(report)
}

fn is_polyasm_compiler_marker(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    [
        sym::polyasm_bind_static_clock,
        sym::polyasm_offload,
        sym::polyasm_request_offload,
        sym::polyasm_require_always,
        sym::polyasm_require_not_always,
        sym::polyasm_require_statement,
    ]
    .into_iter()
    .any(|item| tcx.is_diagnostic_item(item, def_id))
}

fn is_polyasm_static_invoke_marker(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.is_diagnostic_item(sym::polyasm_invoke_static_faster, def_id)
        || tcx.is_diagnostic_item(sym::polyasm_invoke_static_faster_exact, def_id)
        || tcx.is_diagnostic_item(sym::polyasm_invoke_static_faster_on, def_id)
}

fn is_atomic_operation(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.opt_item_name(def_id).is_some_and(|name| name.as_str().contains("atomic"))
}

// Scalar byte reversal has a closed BPF lowering for these exact integer
// types alone. Support follows these types and stays away from storage size
// (e.g. pointers or aggregates), other intrinsics and device timing models.
fn bswap_report<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, args: ty::GenericArgsRef<'tcx>) -> Report {
    let span = tcx.def_span(def_id);
    if !matches!(
        args.type_at(0).kind(),
        ty::Uint(ty::UintTy::U16 | ty::UintTy::U32 | ty::UintTy::U64)
            | ty::Int(ty::IntTy::I16 | ty::IntTy::I32 | ty::IntTy::I64)
    ) {
        return bodyless_callee_report(
            span,
            "PolyASM byte-swap witness requires an exact 16-, 32-, or 64-bit integer type",
        );
    }
    let mut report = Report::new(0);
    disprove(
        &mut report,
        ALL & !(EBPF | XDP | LINUX_SAFE),
        span,
        "this scalar byte-swap warrant is limited to BPF properties and has no exact clock model",
    );
    report.cycles = None;
    report
}

fn bodyless_callee_report(span: Span, reason: &'static str) -> Report {
    let mut report = Report::new(0);
    disprove(&mut report, ALL, span, reason);
    report.cycles = None;
    report
}

fn known_bool_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    operand: &Operand<'tcx>,
    known_boolean: &[Option<bool>],
    known_overflow: &[Option<bool>],
) -> Option<bool> {
    match operand {
        Operand::Constant(constant) => {
            let constant = monomorphize(tcx, instance, constant.const_);
            (constant.ty() == tcx.types.bool)
                .then(|| constant.try_eval_bool(tcx, ty::TypingEnv::fully_monomorphized()))
                .flatten()
        }
        Operand::Copy(place) | Operand::Move(place) => {
            if place.projection.is_empty() {
                known_boolean.get(place.local.as_usize()).copied().flatten()
            } else {
                let [ProjectionElem::Field(field, _)] = place.projection.as_ref() else {
                    return None;
                };
                (field.as_usize() == 1)
                    .then(|| known_overflow.get(place.local.as_usize()).copied().flatten())
                    .flatten()
            }
        }
        Operand::RuntimeChecks(checks) => Some(checks.value(tcx.sess)),
    }
}

fn known_boolean_rvalue<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    value: &Rvalue<'tcx>,
    provenance: &[MemoryProvenance],
    known_boolean: &[Option<bool>],
    known_nonzero: &[Option<bool>],
) -> Option<bool> {
    match value {
        Rvalue::Use(operand, _) => known_bool_operand(tcx, instance, operand, known_boolean, &[]),
        Rvalue::BinaryOp(operation, operands) => {
            let lhs = known_bool_operand(tcx, instance, &operands.0, known_boolean, &[]);
            let rhs = known_bool_operand(tcx, instance, &operands.1, known_boolean, &[]);
            if let (Some((lhs, lhs_bits, lhs_signed)), Some((rhs, rhs_bits, rhs_signed))) = (
                constant_integer(tcx, instance, &operands.0),
                constant_integer(tcx, instance, &operands.1),
            ) && lhs_bits == rhs_bits
                && lhs_signed == rhs_signed
            {
                // These MIR constants contain concrete type projections at
                // times, such as `size_of::<T>()`, that become integers only
                // after instance substitution. Comparing them at their actual
                // signed width excludes dead generic panic branches.
                let ordering = if lhs_signed {
                    let shift = 128_u32.checked_sub(lhs_bits)?;
                    (((lhs << shift) as i128) >> shift).cmp(&(((rhs << shift) as i128) >> shift))
                } else {
                    lhs.cmp(&rhs)
                };
                match operation {
                    BinOp::Eq => return Some(ordering.is_eq()),
                    BinOp::Ne => return Some(!ordering.is_eq()),
                    BinOp::Lt => return Some(ordering.is_lt()),
                    BinOp::Le => return Some(!ordering.is_gt()),
                    BinOp::Gt => return Some(ordering.is_gt()),
                    BinOp::Ge => return Some(!ordering.is_lt()),
                    _ => {}
                }
            }
            match operation {
                BinOp::BitAnd => match (lhs, rhs) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                },
                BinOp::Eq | BinOp::Ne => {
                    let pointer_is_lhs =
                        operand_memory_provenance(tcx, instance, &operands.0, provenance)
                            .has_nonzero_address()
                            && constant_integer(tcx, instance, &operands.1)
                                .is_some_and(|(value, _, _)| value == 0);
                    let pointer_is_rhs =
                        operand_memory_provenance(tcx, instance, &operands.1, provenance)
                            .has_nonzero_address()
                            && constant_integer(tcx, instance, &operands.0)
                                .is_some_and(|(value, _, _)| value == 0);
                    let nonzero_is_lhs =
                        known_nonzero_operand(tcx, instance, &operands.0, known_nonzero)
                            == Some(true)
                            && constant_integer(tcx, instance, &operands.1)
                                .is_some_and(|(value, _, _)| value == 0);
                    let nonzero_is_rhs =
                        known_nonzero_operand(tcx, instance, &operands.1, known_nonzero)
                            == Some(true)
                            && constant_integer(tcx, instance, &operands.0)
                                .is_some_and(|(value, _, _)| value == 0);
                    (pointer_is_lhs || pointer_is_rhs || nonzero_is_lhs || nonzero_is_rhs)
                        .then_some(matches!(operation, BinOp::Ne))
                }
                _ => None,
            }
        }
        Rvalue::UnaryOp(UnOp::Not, operand) => {
            known_bool_operand(tcx, instance, operand, known_boolean, &[]).map(|value| !value)
        }
        _ => None,
    }
}

fn known_nonzero_rvalue<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    value: &Rvalue<'tcx>,
    known_nonzero: &[Option<bool>],
) -> Option<bool> {
    match value {
        Rvalue::Use(operand, _) => known_nonzero_operand(tcx, instance, operand, known_nonzero),
        Rvalue::BinaryOp(BinOp::BitOr, operands) => {
            let lhs = known_nonzero_operand(tcx, instance, &operands.0, known_nonzero);
            let rhs = known_nonzero_operand(tcx, instance, &operands.1, known_nonzero);
            match (lhs, rhs) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            }
        }
        _ => None,
    }
}

fn known_nonzero_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    operand: &Operand<'tcx>,
    known_nonzero: &[Option<bool>],
) -> Option<bool> {
    match operand {
        Operand::Constant(_) => {
            constant_integer(tcx, instance, operand).map(|(value, _, _)| value != 0)
        }
        Operand::Copy(place) | Operand::Move(place) if place.projection.is_empty() => {
            known_nonzero.get(place.local.as_usize()).copied().flatten()
        }
        Operand::Copy(_) | Operand::Move(_) | Operand::RuntimeChecks(_) => None,
    }
}

fn constant_overflow<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    value: &Rvalue<'tcx>,
) -> Option<bool> {
    let Rvalue::BinaryOp(operation, operands) = value else {
        return None;
    };
    if !matches!(
        operation,
        BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow
    ) {
        return None;
    }
    let (lhs, bits, signed) = constant_integer(tcx, instance, &operands.0)?;
    let (rhs, rhs_bits, rhs_signed) = constant_integer(tcx, instance, &operands.1)?;
    if bits != rhs_bits || signed != rhs_signed {
        return None;
    }
    if signed {
        let shift = 128_u32.checked_sub(bits)?;
        let lhs = ((lhs << shift) as i128) >> shift;
        let rhs = ((rhs << shift) as i128) >> shift;
        let value = match operation {
            BinOp::AddWithOverflow => lhs.checked_add(rhs),
            BinOp::SubWithOverflow => lhs.checked_sub(rhs),
            BinOp::MulWithOverflow => lhs.checked_mul(rhs),
            _ => unreachable!(),
        };
        let min = if bits == 128 { i128::MIN } else { -(1_i128 << (bits - 1)) };
        let max = if bits == 128 { i128::MAX } else { (1_i128 << (bits - 1)) - 1 };
        Some(value.is_none_or(|value| value < min || value > max))
    } else {
        let max = if bits == 128 { u128::MAX } else { (1_u128 << bits) - 1 };
        let value = match operation {
            BinOp::AddWithOverflow => lhs.checked_add(rhs),
            BinOp::SubWithOverflow => lhs.checked_sub(rhs),
            BinOp::MulWithOverflow => lhs.checked_mul(rhs),
            _ => unreachable!(),
        };
        Some(value.is_none_or(|value| value > max))
    }
}

fn constant_integer<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    operand: &Operand<'tcx>,
) -> Option<(u128, u32, bool)> {
    let Operand::Constant(constant) = operand else {
        return None;
    };
    let constant = monomorphize(tcx, instance, constant.const_);
    let value = constant.try_eval_bits(tcx, ty::TypingEnv::fully_monomorphized())?;
    let ty = monomorphize(tcx, instance, constant.ty());
    let (bits, signed) = match ty.kind() {
        ty::Int(ty::IntTy::I8) => (8, true),
        ty::Int(ty::IntTy::I16) => (16, true),
        ty::Int(ty::IntTy::I32) => (32, true),
        ty::Int(ty::IntTy::I64) => (64, true),
        ty::Int(ty::IntTy::I128) => (128, true),
        ty::Int(ty::IntTy::Isize) => (tcx.data_layout.pointer_size().bits() as u32, true),
        ty::Uint(ty::UintTy::U8) => (8, false),
        ty::Uint(ty::UintTy::U16) => (16, false),
        ty::Uint(ty::UintTy::U32) => (32, false),
        ty::Uint(ty::UintTy::U64) => (64, false),
        ty::Uint(ty::UintTy::U128) => (128, false),
        ty::Uint(ty::UintTy::Usize) => (tcx.data_layout.pointer_size().bits() as u32, false),
        _ => return None,
    };
    Some((value, bits, signed))
}

fn timing_class_for_rvalue<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    value: &Rvalue<'tcx>,
) -> TimingClass {
    match value {
        Rvalue::BinaryOp(operation, operands) => {
            if matches!(
                operation,
                BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt | BinOp::Cmp
            ) {
                TimingClass::Compare
            } else {
                let ty = monomorphize(
                    tcx,
                    instance,
                    operands.0.ty(&tcx.instance_mir(instance.def).local_decls, tcx),
                );
                if contains_float(tcx, ty) { TimingClass::Float } else { TimingClass::Integer }
            }
        }
        Rvalue::UnaryOp(_, operand) => {
            let ty = monomorphize(
                tcx,
                instance,
                operand.ty(&tcx.instance_mir(instance.def).local_decls, tcx),
            );
            if contains_float(tcx, ty) { TimingClass::Float } else { TimingClass::Integer }
        }
        Rvalue::Ref(..)
        | Rvalue::RawPtr(..)
        | Rvalue::Reborrow(..)
        | Rvalue::CopyForDeref(..)
        | Rvalue::Discriminant(..)
        | Rvalue::ThreadLocalRef(..) => TimingClass::Memory,
        Rvalue::Use(Operand::Constant(..), _) | Rvalue::Repeat(..) => TimingClass::Constant,
        Rvalue::Use(..)
        | Rvalue::Cast(..)
        | Rvalue::WrapUnsafeBinder(..)
        | Rvalue::Aggregate(..) => TimingClass::Integer,
    }
}

fn analyze_rvalue<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    destination: Place<'tcx>,
    value: &Rvalue<'tcx>,
    span: Span,
    body: &Body<'tcx>,
    provenance: &mut [MemoryProvenance],
    report: &mut Report,
) {
    clear_indirect_local_value(destination, provenance);
    let next = destination
        .projection
        .is_empty()
        .then(|| assigned_memory_provenance(tcx, instance, body, destination, value, provenance));
    match value {
        Rvalue::Ref(_, kind, place) => {
            disprove(
                report,
                POINTER_ACCELERATORS,
                span,
                "pointer values are not admitted by pointer-accelerator lowering",
            );
            analyze_place_access(
                tcx,
                instance,
                body,
                *place,
                span,
                provenance,
                report,
                matches!(kind, BorrowKind::Mut { .. }),
            );
        }
        Rvalue::RawPtr(kind, place) => {
            disprove(
                report,
                POINTER_ACCELERATORS,
                span,
                "pointer values are not admitted by pointer-accelerator lowering",
            );
            analyze_place_access(
                tcx,
                instance,
                body,
                *place,
                span,
                provenance,
                report,
                *kind == RawPtrKind::Mut,
            );
        }
        Rvalue::Reborrow(_, mutability, place) => {
            disprove(
                report,
                POINTER_ACCELERATORS,
                span,
                "pointer values are not admitted by pointer-accelerator lowering",
            );
            analyze_place_access(
                tcx,
                instance,
                body,
                *place,
                span,
                provenance,
                report,
                *mutability == hir::Mutability::Mut,
            );
        }
        Rvalue::Use(operand, _) => {
            analyze_operand(tcx, instance, body, operand, span, provenance, report);
        }
        Rvalue::Cast(kind, operand, target) => {
            analyze_operand(tcx, instance, body, operand, span, provenance, report);
            let source_provenance = operand_memory_provenance(tcx, instance, operand, provenance);
            // A conversion between two addressless values holds zero provenance
            // to preserve, so an integer read back from its own bytes is closed
            // exactly as the explicit little-endian call that spelled it is.
            if source_provenance == MemoryProvenance::Unknown
                && matches!(
                    kind,
                    CastKind::PtrToPtr
                        | CastKind::PointerCoercion(..)
                        | CastKind::PointerExposeProvenance
                        | CastKind::Transmute
                )
                && (contains_address(
                    tcx,
                    monomorphize(tcx, instance, operand.ty(&body.local_decls, tcx)),
                ) || contains_address(tcx, monomorphize(tcx, instance, *target)))
            {
                // Wasm carries the address bits through this cast and keeps
                // its normal linear-memory bounds checks on later accesses.
                // The cast keeps those checks in force, and a device with a
                // stricter pointer provenance rule keeps its own rule.
                disprove(
                    report,
                    ALL & !WASM,
                    span,
                    "this cast does not preserve a compiler-proven memory provenance",
                );
            }
        }
        Rvalue::WrapUnsafeBinder(operand, _) => {
            analyze_operand(tcx, instance, body, operand, span, provenance, report);
            disprove_unknown_effects(
                report,
                span,
                "unsafe-binder values have no closed PolyASM effect model",
            );
        }
        Rvalue::BinaryOp(operation, operands) => {
            analyze_operand(tcx, instance, body, &operands.0, span, provenance, report);
            analyze_operand(tcx, instance, body, &operands.1, span, provenance, report);
            let ty = monomorphize(
                tcx,
                instance,
                operands.0.ty(&tcx.instance_mir(instance.def).local_decls, tcx),
            );
            if contains_float(tcx, ty) {
                disprove(
                    report,
                    EBPF | XDP | VERILOG | LINUX_SAFE | P4,
                    span,
                    "floating-point arithmetic is outside the eBPF/XDP/P4 integer subset",
                );
            }
            if matches!(operation, BinOp::Div | BinOp::Rem) {
                disprove(
                    report,
                    EBPF | XDP | LINUX_SAFE | P4,
                    span,
                    "division and remainder are not admitted by the portable eBPF/XDP/P4 warrant",
                );
            }
        }
        Rvalue::UnaryOp(_, operand) | Rvalue::Repeat(operand, _) => {
            analyze_operand(tcx, instance, body, operand, span, provenance, report);
        }
        Rvalue::Aggregate(_, operands) => {
            for operand in operands {
                analyze_operand(tcx, instance, body, operand, span, provenance, report);
            }
        }
        Rvalue::CopyForDeref(place) | Rvalue::Discriminant(place) => {
            analyze_place(tcx, instance, body, *place, span, provenance, report);
        }
        Rvalue::ThreadLocalRef(_) => disprove_unknown_effects(
            report,
            span,
            "thread-local storage has no closed static hardware address",
        ),
    }
    if rvalue_moves_value(value) {
        provenance.fill(MemoryProvenance::Unknown);
    }
    if let Some(next) = next {
        provenance[destination.local.as_usize()] = next;
    }
}

fn analyze_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
    span: Span,
    provenance: &[MemoryProvenance],
    report: &mut Report,
) {
    if let Operand::Copy(place) | Operand::Move(place) = operand {
        analyze_place(tcx, instance, body, *place, span, provenance, report);
    }
}

fn analyze_place<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
    span: Span,
    provenance: &[MemoryProvenance],
    report: &mut Report,
) {
    analyze_place_access(tcx, instance, body, place, span, provenance, report, false);
}

fn analyze_place_write<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
    span: Span,
    provenance: &[MemoryProvenance],
    report: &mut Report,
) {
    analyze_place_access(tcx, instance, body, place, span, provenance, report, true);
}

fn analyze_place_access<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
    span: Span,
    provenance: &[MemoryProvenance],
    report: &mut Report,
    write: bool,
) {
    if place.projection.iter().any(|projection| {
        matches!(projection, ProjectionElem::Index(_) | ProjectionElem::Subslice { .. })
    }) {
        disprove(
            report,
            ALL & !WASM,
            span,
            "dynamic indexing has no compiler-proven fixed memory extent",
        );
        return;
    }
    if !place.projection.iter().any(|projection| matches!(projection, ProjectionElem::Deref)) {
        return;
    }
    disprove(
        report,
        POINTER_ACCELERATORS,
        span,
        "pointer values are not admitted by pointer-accelerator lowering",
    );
    let mut place_ty =
        PlaceTy::from_ty(monomorphize(tcx, instance, body.local_decls[place.local].ty));
    let mut dereferences = 0;
    for projection in place.projection.iter() {
        if matches!(projection, ProjectionElem::Deref) {
            dereferences += 1;
            let pointee = place_ty
                .ty
                .builtin_deref(true)
                .and_then(|ty| memory_layout(tcx, monomorphize(tcx, instance, ty)));
            let source = if dereferences == 1 {
                local_memory_provenance(place, provenance)
            } else {
                MemoryProvenance::Unknown
            };
            match source {
                MemoryProvenance::Proven { access, origin } if pointee.is_some() => {
                    if write && !access.permits_write(pointee.unwrap()) {
                        disprove_invalid_pointer_write(report, span);
                    } else if !access.permits(pointee.unwrap()) {
                        disprove_invalid_pointer_layout(report, span);
                    } else if matches!(origin, MemoryOrigin::FixedStatic(_)) {
                        disprove(
                            report,
                            XDP | LINUX_SAFE,
                            span,
                            "fixed guest memory is outside the Linux checker-owned eBPF stack",
                        );
                    }
                }
                MemoryProvenance::Proven { .. } | MemoryProvenance::Unknown => {
                    disprove(
                        report,
                        EBPF | XDP
                            | VERILOG
                            | LINUX_SAFE
                            | STATIC_MEMORY
                            | STATIC_MEMORY_UPPER
                            | STATIC_CLOCK
                            | ACCELERATOR
                            | CUDA
                            | DPA
                            | PTX
                            | P4,
                        span,
                        "this dereference has no compiler-proven in-bounds memory provenance",
                    );
                    report.cycles = None;
                }
            }
        }
        place_ty = place_ty.projection_ty(tcx, projection);
    }
}

fn local_memory_provenance(place: Place<'_>, provenance: &[MemoryProvenance]) -> MemoryProvenance {
    provenance.get(place.local.as_usize()).copied().unwrap_or(MemoryProvenance::Unknown)
}

fn place_address_provenance<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &Body<'tcx>,
    place: Place<'tcx>,
    provenance: &[MemoryProvenance],
    writable: bool,
) -> MemoryProvenance {
    let pointee = monomorphize(tcx, instance, place.ty(&body.local_decls, tcx).ty);
    let writable =
        writable || matches!(pointee.kind(), ty::Adt(definition, _) if definition.is_unsafe_cell());
    let mut ty = monomorphize(tcx, instance, body.local_decls[place.local].ty);
    let mut projections = place.projection.iter();
    let mut fact = if projections
        .clone()
        .next()
        .is_some_and(|projection| matches!(projection, ProjectionElem::Deref))
    {
        projections.next();
        let Some(pointee) = ty.builtin_deref(true).map(|ty| monomorphize(tcx, instance, ty)) else {
            return MemoryProvenance::Unknown;
        };
        let Some(layout) = memory_layout(tcx, pointee) else {
            return MemoryProvenance::Unknown;
        };
        ty = pointee;
        local_memory_provenance(place, provenance).restrict_write(writable).retarget(layout)
    } else {
        let Some(allocation) = memory_layout(tcx, ty) else {
            return MemoryProvenance::Unknown;
        };
        MemoryProvenance::Proven {
            access: MemoryAccess {
                address_only: false,
                allocation,
                offset: 0,
                pointee: allocation,
                writable,
            },
            origin: MemoryOrigin::Local(place.local),
        }
    };
    for projection in projections {
        let ProjectionElem::Field(field, _) = projection else {
            return MemoryProvenance::Unknown;
        };
        let Some((offset, field_ty, pointee)) = field_projection_index(tcx, ty, field.as_usize())
        else {
            return MemoryProvenance::Unknown;
        };
        fact = match fact {
            MemoryProvenance::Proven { access, origin } => {
                access.project(offset, pointee).map_or(MemoryProvenance::Unknown, |access| {
                    MemoryProvenance::Proven { access, origin }
                })
            }
            MemoryProvenance::Unknown => MemoryProvenance::Unknown,
        };
        ty = field_ty;
    }
    fact
}

fn operand_memory_provenance<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    operand: &Operand<'tcx>,
    provenance: &[MemoryProvenance],
) -> MemoryProvenance {
    match operand {
        Operand::Copy(place) | Operand::Move(place) if place.projection.is_empty() => {
            local_memory_provenance(*place, provenance)
        }
        Operand::Copy(_) | Operand::Move(_) => MemoryProvenance::Unknown,
        Operand::Constant(constant) => {
            let constant = monomorphize(tcx, instance, constant.const_);
            let Some(Scalar::Ptr(pointer, _)) =
                constant.try_eval_scalar(tcx, ty::TypingEnv::fully_monomorphized())
            else {
                return MemoryProvenance::Unknown;
            };
            let (pointer_provenance, offset) = pointer.prov_and_relative_offset();
            let Some(allocation @ GlobalAlloc::Static(def_id)) =
                tcx.try_get_global_alloc(pointer_provenance.alloc_id())
            else {
                return MemoryProvenance::Unknown;
            };
            if tcx.is_thread_local_static(def_id) || tcx.is_foreign_item(def_id) {
                MemoryProvenance::Unknown
            } else {
                let (bytes, align) =
                    allocation.size_and_align(tcx, ty::TypingEnv::fully_monomorphized());
                let Some(pointee) =
                    constant.ty().builtin_deref(true).and_then(|ty| memory_layout(tcx, ty))
                else {
                    return MemoryProvenance::Unknown;
                };
                MemoryProvenance::Proven {
                    access: MemoryAccess {
                        address_only: false,
                        allocation: MemoryLayout { align: align.bytes(), bytes: bytes.bytes() },
                        offset: offset.bytes(),
                        pointee,
                        writable: matches!(
                            tcx.def_kind(def_id),
                            DefKind::Static { mutability: hir::Mutability::Mut, .. }
                        ),
                    },
                    origin: MemoryOrigin::FixedStatic(def_id),
                }
            }
        }
        Operand::RuntimeChecks(_) => MemoryProvenance::Unknown,
    }
}

fn volatile_ebpf_type(ty: Ty<'_>) -> bool {
    matches!(
        ty.kind(),
        ty::Uint(ty::UintTy::U32 | ty::UintTy::U64 | ty::UintTy::Usize)
            | ty::Int(ty::IntTy::I32 | ty::IntTy::I64 | ty::IntTy::Isize)
    )
}

fn monomorphize<'tcx, T>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>, value: T) -> T
where
    T: TypeFoldable<TyCtxt<'tcx>> + Copy,
{
    instance.instantiate_mir_and_normalize_erasing_regions(
        tcx,
        ty::TypingEnv::fully_monomorphized(),
        EarlyBinder::bind(tcx, value),
    )
}

fn contains_address<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    contains_address_inner(tcx, ty, &mut FxHashSet::default())
}

fn contains_address_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    visiting: &mut FxHashSet<Ty<'tcx>>,
) -> bool {
    match ty.kind() {
        ty::Bool | ty::Char | ty::Float(_) | ty::Int(_) | ty::Never | ty::Uint(_) => false,
        ty::Tuple(fields) => {
            fields.iter().any(|field| contains_address_inner(tcx, field, visiting))
        }
        ty::Array(element, _) | ty::Pat(element, _) | ty::Slice(element) => {
            contains_address_inner(tcx, *element, visiting)
        }
        ty::Adt(definition, args) if visiting.insert(ty) => {
            let result = definition.variants().iter().any(|variant| {
                variant.fields.iter().any(|field| {
                    contains_address_inner(tcx, field.ty(tcx, args).skip_norm_wip(), visiting)
                })
            });
            visiting.remove(&ty);
            result
        }
        _ => true,
    }
}

fn contains_float<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    contains_float_inner(tcx, ty, &mut FxHashSet::default())
}

fn contains_float_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    visiting: &mut FxHashSet<DefId>,
) -> bool {
    match ty.kind() {
        ty::Float(_) => true,
        ty::Tuple(fields) => fields.iter().any(|field| contains_float_inner(tcx, field, visiting)),
        ty::Array(element, _)
        | ty::Pat(element, _)
        | ty::Slice(element)
        | ty::Ref(_, element, _)
        | ty::RawPtr(element, _) => contains_float_inner(tcx, *element, visiting),
        ty::Adt(definition, args) if visiting.insert(definition.did()) => {
            let result = definition.variants().iter().any(|variant| {
                variant.fields.iter().any(|field| {
                    contains_float_inner(tcx, field.ty(tcx, args).skip_norm_wip(), visiting)
                })
            });
            visiting.remove(&definition.did());
            result
        }
        // A repeated definition is either a finite nested instantiation such
        // as Wrapper<Wrapper<f32>>, or an expanding pointer-linked type such as
        // Node<T> -> *const Node<(T,)> . Inspect the finite type arguments at
        // that boundary and keep the definition expanded once.
        ty::Adt(_, args) => {
            args.types().any(|argument| contains_float_inner(tcx, argument, visiting))
        }
        _ => false,
    }
}

fn add_cycles(report: &mut Report, cycles: u64) {
    let Some(total) = report.cycles else { return };
    report.cycles = total.checked_add(cycles);
    if report.cycles.is_none() {
        disprove_schedule(
            report,
            Span::default(),
            "the exact static schedule exceeds the representable cycle count",
        );
    }
}

fn add_timing(report: &mut Report, class: TimingClass, span: Span) {
    if !report.timing.add(class, 1) {
        disprove_schedule(
            report,
            span,
            "the normalized board timing count exceeds the representable cycle count",
        );
    }
    add_cycles(report, 1);
}

fn merge_callee_report(report: &mut Report, callee: Report, span: Span) {
    let cycles = match (report.cycles, callee.cycles) {
        (Some(lhs), Some(rhs)) => match lhs.checked_add(rhs) {
            Some(cycles) => Some(cycles),
            None => {
                disprove_schedule(
                    report,
                    span,
                    "the exact static schedule exceeds the representable cycle count",
                );
                None
            }
        },
        _ => None,
    };
    report.merge(callee);
    report.cycles = cycles;
    for class in 0..INSTRUCTION_CLASS_COUNT {
        let Some(total) = report.timing.0[class].checked_add(callee.timing.0[class]) else {
            disprove_schedule(
                report,
                span,
                "the normalized board timing count exceeds the representable cycle count",
            );
            return;
        };
        report.timing.0[class] = total;
    }
}

fn polyasm_u64_const<'tcx>(tcx: TyCtxt<'tcx>, constant: ty::Const<'tcx>) -> Option<u64> {
    let value = constant.try_to_value()?;
    if value.ty != tcx.types.u64 {
        return None;
    }
    value.try_to_bits(tcx, ty::TypingEnv::fully_monomorphized())?.try_into().ok()
}

impl Report {
    fn new(cycles: u64) -> Self {
        Self {
            capabilities: ALL,
            disproven: 0,
            cycles: Some(cycles),
            timing: TimingCounts::new(),
            failures: [None; CAPABILITY_COUNT],
        }
    }

    fn failure(self, capability: u32) -> Option<Failure> {
        debug_assert!(capability.is_power_of_two());
        self.failures[capability.trailing_zeros() as usize]
    }

    fn merge(&mut self, other: Self) {
        for shift in 0..CAPABILITY_COUNT {
            let capability = 1 << shift;
            if (other.disproven & capability != 0 && self.disproven & capability == 0)
                || (self.capabilities & capability != 0 && other.capabilities & capability == 0)
            {
                self.failures[shift] = other.failures[shift];
            }
        }
        self.capabilities &= other.capabilities;
        self.disproven |= other.disproven;
    }

    fn warrant(self, capability: u32) -> Warrant {
        debug_assert!(capability.is_power_of_two());
        if self.capabilities & capability != 0 {
            Warrant::Proven
        } else if self.disproven & capability != 0 {
            Warrant::Disproven
        } else {
            Warrant::Deferred
        }
    }
}

impl TimingCounts {
    const fn new() -> Self {
        Self([0; INSTRUCTION_CLASS_COUNT])
    }

    fn add(&mut self, class: TimingClass, count: u64) -> bool {
        let slot = &mut self.0[class as usize];
        let Some(total) = slot.checked_add(count) else { return false };
        *slot = total;
        true
    }

    fn dominates(self, other: Self) -> bool {
        self.0.into_iter().zip(other.0).all(|(lhs, rhs)| lhs <= rhs)
    }

    fn schedule(self, board: &FpgaBoard) -> Option<u64> {
        self.0.into_iter().zip(board.cycles_table()).try_fold(0_u64, |total, (count, cycles)| {
            total.checked_add(count.checked_mul(u64::from(cycles))?)
        })
    }
}

fn static_clock_board(architecture: Architecture) -> Option<&'static FpgaBoard> {
    match architecture {
        Architecture::FpgaBoard(board) => Some(board),
        _ => None,
    }
}

fn defer(report: &mut Report, capabilities: u32, span: Span, reason: &'static str) {
    let deferred = capabilities & !report.disproven;
    report.capabilities &= !deferred;
    for (shift, failure) in report.failures.iter_mut().enumerate() {
        if deferred & (1 << shift) != 0 {
            failure.get_or_insert(Failure { reason, span });
        }
    }
}

fn disprove(report: &mut Report, capabilities: u32, span: Span, reason: &'static str) {
    let newly_disproven = capabilities & !report.disproven;
    report.capabilities &= !capabilities;
    report.disproven |= capabilities;
    for (shift, failure) in report.failures.iter_mut().enumerate() {
        if newly_disproven & (1 << shift) != 0 {
            *failure = Some(Failure { reason, span });
        }
    }
}

fn defer_schedule(report: &mut Report, span: Span, reason: &'static str) {
    defer(report, STATIC_CLOCK, span, reason);
    report.cycles = None;
}

fn disprove_schedule(report: &mut Report, span: Span, reason: &'static str) {
    disprove(report, STATIC_CLOCK, span, reason);
    report.cycles = None;
}

fn defer_unknown_effects(report: &mut Report, span: Span, reason: &'static str) {
    defer(report, ALL, span, reason);
    report.cycles = None;
}

fn disprove_unknown_effects(report: &mut Report, span: Span, reason: &'static str) {
    disprove(report, ALL, span, reason);
    report.cycles = None;
}

fn property(tcx: TyCtxt<'_>, ty: Ty<'_>) -> Option<Property> {
    let ty::Adt(definition, _) = ty.kind() else {
        return None;
    };
    let did = definition.did();
    for (name, property) in [
        ("polyasm_property_accelerator", Property::Capability(ACCELERATOR)),
        ("polyasm_property_cuda", Property::Capability(CUDA)),
        ("polyasm_property_dpa", Property::Capability(DPA)),
        ("polyasm_property_ebpf", Property::Capability(EBPF)),
        ("polyasm_property_llvm_bridge", Property::Capability(LLVM_BRIDGE)),
        ("polyasm_property_linux_safe", Property::Capability(LINUX_SAFE)),
        ("polyasm_property_native", Property::Capability(NATIVE)),
        ("polyasm_property_p4", Property::Capability(P4)),
        ("polyasm_property_ptx", Property::Capability(PTX)),
        ("polyasm_property_rust", Property::Capability(RUST)),
        ("polyasm_property_static_clock", Property::StaticClock),
        ("polyasm_property_static_memory", Property::Capability(STATIC_MEMORY)),
        ("polyasm_property_static_memory_upper", Property::Capability(STATIC_MEMORY_UPPER)),
        ("polyasm_property_verilog", Property::Capability(VERILOG)),
        ("polyasm_property_wasm", Property::Capability(WASM)),
        ("polyasm_property_xdp", Property::Capability(XDP)),
    ] {
        if tcx.is_diagnostic_item(Symbol::intern(name), did) {
            return Some(property);
        }
    }
    None
}

fn architecture(tcx: TyCtxt<'_>, ty: Ty<'_>) -> Option<Architecture> {
    let ty::Adt(definition, _) = ty.kind() else {
        return None;
    };
    let did = definition.did();
    for (name, architecture) in [
        ("polyasm_arch_aarch64", Architecture::Aarch64),
        ("polyasm_arch_cxl", Architecture::Cxl),
        ("polyasm_arch_dpu", Architecture::Dpu),
        ("polyasm_arch_ebpf", Architecture::Ebpf),
        ("polyasm_arch_fpga", Architecture::Fpga),
        ("polyasm_arch_gpu", Architecture::Gpu),
        ("polyasm_arch_host", Architecture::Host),
        ("polyasm_arch_npu", Architecture::Npu),
        ("polyasm_arch_polyasm", Architecture::Polyasm),
        ("polyasm_arch_riscv64", Architecture::Riscv64),
        ("polyasm_arch_wasm32", Architecture::Wasm32),
        ("polyasm_arch_x86_64", Architecture::X86_64),
        ("polyasm_arch_xdp", Architecture::Xdp),
    ] {
        if tcx.is_diagnostic_item(Symbol::intern(name), did) {
            return Some(architecture);
        }
    }
    for board in FPGA_BOARDS {
        if tcx.is_diagnostic_item(Symbol::intern(board.diagnostic_item()), did) {
            return Some(Architecture::FpgaBoard(board));
        }
    }
    None
}

fn accepts(architecture: Architecture, property: u32) -> bool {
    match architecture {
        Architecture::Host | Architecture::Polyasm => true,
        Architecture::Aarch64 | Architecture::Riscv64 | Architecture::X86_64 => matches!(
            property,
            NATIVE | RUST | STATIC_CLOCK | STATIC_MEMORY | STATIC_MEMORY_UPPER | WASM
        ),
        Architecture::Cxl => {
            matches!(property, STATIC_MEMORY | STATIC_MEMORY_UPPER | WASM)
        }
        Architecture::Dpu | Architecture::Gpu | Architecture::Npu => {
            matches!(property, ACCELERATOR | STATIC_CLOCK | STATIC_MEMORY | STATIC_MEMORY_UPPER)
        }
        Architecture::Ebpf => matches!(
            property,
            EBPF | LINUX_SAFE | STATIC_CLOCK | STATIC_MEMORY | STATIC_MEMORY_UPPER
        ),
        Architecture::Fpga => {
            matches!(property, STATIC_MEMORY | STATIC_MEMORY_UPPER | VERILOG)
        }
        Architecture::FpgaBoard(_) => {
            matches!(property, STATIC_CLOCK | STATIC_MEMORY | STATIC_MEMORY_UPPER | VERILOG)
        }
        Architecture::Wasm32 => {
            matches!(property, STATIC_CLOCK | STATIC_MEMORY | STATIC_MEMORY_UPPER | WASM)
        }
        Architecture::Xdp => matches!(
            property,
            LINUX_SAFE | STATIC_CLOCK | STATIC_MEMORY | STATIC_MEMORY_UPPER | XDP
        ),
    }
}
