//! Compile-time symbol metadata. Names remain audit/debug information and are
//! resolved to numeric indexes before an artifact is emitted.

use std::collections::{BTreeMap, BTreeSet};
use std::iter;

use polyasm_format::ir::{Architecture, Attributes, Request};
use rustc_codegen_ssa::polyasm::polyasm_offload_roots;
use rustc_data_structures::unord::{ExtendUnord, UnordMap};
use rustc_middle::mono::{
    MonoItem, is_polyasm_entry_instance, is_polyasm_witness_marker, polyasm_witness_callables,
    resolve_polyasm_callable,
};
use rustc_middle::ty::{self, Instance, InstanceKind, Ty, TyCtxt};
use rustc_span::def_id::LOCAL_CRATE;
use rustc_span::{Symbol, sym};

use super::object::OffloadRoot;

/// The WebAssembly module a host import declares when it leaves its namespace open.
const DEFAULT_HOST_MODULE: &str = "env";

/// Returns the audit name of every host call an image still names.
///
/// The map is read for every loaded crate, beyond the local one. A host call
/// is the only undefined symbol final linkage accepts, and this map also
/// decides which record is one: read locally, a crate calling another crate's
/// host import would emit an ordinary colocated call to a record final linkage
/// leaves bodyless.
pub(crate) fn syscall_aliases(tcx: TyCtxt<'_>) -> BTreeMap<String, String> {
    let mut aliases = UnordMap::<String, String>::default();
    for krate in iter::once(LOCAL_CRATE).chain(tcx.crates(()).iter().copied()) {
        aliases.extend_unord(tcx.wasm_import_module_map(krate).items().map(|(def_id, module)| {
            let instance = Instance::mono(tcx, *def_id);
            let symbol = tcx.symbol_name(instance).name.to_string();
            let attributes = tcx.codegen_fn_attrs(*def_id);
            let import = attributes.symbol_name.unwrap_or_else(|| tcx.item_name(*def_id));
            (symbol, host_call_name(module, import.as_str()))
        }));
    }
    aliases
        .to_sorted_stable_ord()
        .into_iter()
        .map(|(symbol, import)| (symbol.clone(), import.clone()))
        .collect()
}

/// Returns the closed syscall table's audit name for one host import.
///
/// An import is a WebAssembly `(module, name)` pair, so its audit name is the
/// pair joined. The default module is the exception: its name stays empty, and
/// `wasm-direct` registers every host function under it while spelling the
/// whole name in the import field, so joining it would invent a namespace
/// outside the closed table.
fn host_call_name(module: &str, import: &str) -> String {
    if module == DEFAULT_HOST_MODULE {
        return import.to_owned();
    }
    format!("{module}::{import}")
}

pub(crate) fn mono_functions<'tcx>(tcx: TyCtxt<'tcx>) -> BTreeMap<String, Instance<'tcx>> {
    let mut functions = BTreeMap::new();
    for cgu in tcx.collect_and_partition_mono_items(()).codegen_units {
        for (item, _) in cgu.items_in_deterministic_order(tcx) {
            if let MonoItem::Fn(instance) = item {
                functions.insert(tcx.symbol_name(instance).name.to_string(), instance);
            }
        }
    }
    functions
}

/// Returns the audit name of every compiler marker.
///
/// A marker carries a compile-time property alone, so `interchange::compile`
/// drops it from code generation and every crate leaves its symbol undefined.
/// That makes it the second bodyless symbol in every object, and unlike a
/// host call the closed syscall table holds zero descriptors to explain one.
/// Naming the markers here lets the backend report a marker that escaped as a
/// value.
pub(crate) fn witness_markers(tcx: TyCtxt<'_>) -> BTreeSet<String> {
    mono_functions(tcx)
        .into_iter()
        .filter(|(_, instance)| is_polyasm_witness_marker(tcx, instance.def_id()))
        .map(|(name, _)| name)
        .collect()
}

/// Returns the callables the final image keeps beyond its entry graph.
///
/// A marker subject becomes a root only when it runs: the callable a
/// static-clock comparison declined stays out of code generation, so this list
/// names running callables alone and final linkage retains defined symbols.
pub(crate) fn offload_roots(tcx: TyCtxt<'_>) -> Vec<OffloadRoot> {
    let mut roots = BTreeMap::<String, Request>::new();
    for instance in mono_functions(tcx).into_values() {
        let span = tcx.def_span(instance.def_id());
        for callable in
            polyasm_offload_roots(tcx, instance.def_id(), instance.args, span).into_iter().flatten()
        {
            if let Some(callable) = resolve_polyasm_callable(tcx, callable, span) {
                let name = tcx.symbol_name(callable).name.to_string();
                roots.entry(name).or_insert(Request::NONE);
            }
        }
    }
    roots.into_iter().map(|(name, request)| OffloadRoot { name, request }).collect()
}

/// Decodes a source marker's target wish. The linked call-site frame carries
/// the wish; the callee record remains a portable callable for other callers.
pub(crate) fn marker_request(tcx: TyCtxt<'_>, property: Ty<'_>, target: Ty<'_>) -> Request {
    let diagnostic_item = |ty: Ty<'_>| -> Option<Symbol> {
        let ty::Adt(definition, _) = ty.kind() else { return None };
        tcx.get_diagnostic_name(definition.did())
    };
    let property_name = diagnostic_item(property)
        .unwrap_or_else(|| tcx.dcx().fatal("PolyASM offload marker names an unknown property"));
    let property = Attributes::from_property_diagnostic_item(&property_name.to_string())
        .unwrap_or_else(|| tcx.dcx().fatal("PolyASM offload marker names an unknown property"));
    let target_name = diagnostic_item(target)
        .unwrap_or_else(|| tcx.dcx().fatal("PolyASM offload marker names an unknown architecture"));
    let target = Architecture::from_diagnostic_item(&target_name.to_string())
        .unwrap_or_else(|| tcx.dcx().fatal("PolyASM offload marker names an unknown architecture"));
    Request::onto(property, target)
}

/// Returns the negative `Always` markers attached to concrete callables.
///
/// Marker calls are compile-time operations and disappear from optimized MIR
/// at times. Their monomorphized marker instances remain compiler markers, so
/// the semantic backend consumes the generic arguments here in place of a
/// surviving runtime call.
pub(crate) fn negative_witness<'tcx>(tcx: TyCtxt<'tcx>) -> Vec<(String, Ty<'tcx>)> {
    let mut witness = Vec::new();
    for instance in mono_functions(tcx).into_values() {
        if !tcx.is_diagnostic_item(sym::polyasm_require_not_always, instance.def_id()) {
            continue;
        }
        let Some(callable) = polyasm_witness_callables(tcx, instance.def_id(), instance.args)[0]
            .and_then(|callable| {
                resolve_polyasm_callable(tcx, callable, tcx.def_span(instance.def_id()))
            })
        else {
            continue;
        };
        let name = tcx.symbol_name(callable).name.to_string();
        witness.push((name, instance.args.type_at(0)));
    }
    witness
}

pub(crate) fn entry_symbol(tcx: TyCtxt<'_>) -> Option<String> {
    let mut fallback = None;
    let mut polyasm_entry = None;
    let rust_entry = tcx.entry_fn(()).map(|(def_id, _)| def_id);
    for (name, instance) in mono_functions(tcx) {
        if is_polyasm_entry_instance(tcx, instance) {
            if polyasm_entry.replace(name.clone()).is_some() {
                tcx.dcx().fatal(
                    "the crate-root `polyasm_entry` function must have exactly one concrete instance",
                );
            }
        }
        if is_ordinary_entry_authority(instance.def, rust_entry) {
            fallback = Some(name);
        }
    }
    polyasm_entry.or(fallback)
}

fn is_ordinary_entry_authority(
    instance: InstanceKind<'_>,
    rust_entry: Option<rustc_hir::def_id::DefId>,
) -> bool {
    matches!(instance, InstanceKind::Item(def_id) if rust_entry == Some(def_id))
}
