//! PolyASM compiler facts shared with code generation backends.

use rustc_attr_ir::find_attr;
use rustc_hir::def::DefKind;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{LOCAL_CRATE, LocalDefId};
use rustc_span::{Span, Symbol};
pub use rustc_trait_selection::traits::{
    PolyasmStatementSelection, PolyasmStaticCallableSelection, PolyasmStaticCallableSelectionError,
    normalized_polyasm_property_holds, normalized_polyasm_statement_property_holds,
    normalized_polyasm_static_schedule, polyasm_offload_roots, polyasm_statement_selection,
    polyasm_static_callable_selection, polyasm_static_callable_selection_on,
    polyasm_static_clock_hz,
};

/// Whether one intrinsic is a PolyASM instruction: an item of the module
/// `core::polyasm::intrinsics` names as its diagnostic item.
///
/// Both codegen backends lower such a call to the symbol
/// `__polyasm_<intrinsic name>` over the operands the declaration states, and
/// `polyasm_format::library::instruction` answers the row that symbol is. An
/// instruction whose intrinsic carries a body passes that body as the row's
/// record: the portable answer a machine enters where it carries the row
/// elsewhere.
pub fn instruction(tcx: TyCtxt<'_>, def_id: rustc_span::def_id::DefId) -> bool {
    tcx.intrinsic(def_id).is_some()
        && tcx.is_diagnostic_item(rustc_span::sym::polyasm_intrinsics, tcx.parent(def_id))
}

/// The symbol a backend calls for one PolyASM instruction.
pub fn instruction_symbol(tcx: TyCtxt<'_>, def_id: rustc_span::def_id::DefId) -> String {
    format!("__polyasm_{}", tcx.item_name(def_id))
}

/// The crate a PolyASM guest depends on.
///
/// Every crate that links it, other than the crate itself, is a guest, and a
/// guest reaches its host only through the two macros below.
const GUEST_CORE: &str = "wasm_direct_core";

/// The two macros a guest publishes a function through, each named by the
/// crate that defines it and the function rustc expands it with.
///
/// `#[export]` writes the five lifecycle records of one work and `#[offload]`
/// writes a route's manifest readers around an `#[export]`. Each of them
/// spells its symbols itself, so an item whose identifier either of them
/// wrote carries the symbol the host reads.
const PUBLISHING_MACROS: [(&str, &str); 2] = [
    ("wasm_direct_bindgen", "attribute_sandbox_export"),
    ("wasm_direct_http_bindgen", "attribute_offload"),
];

/// Stops every symbol a PolyASM guest crate publishes by hand.
///
/// A guest publishes a function only through `#[export]` or `#[offload]`.
/// An item of a guest crate that carries `#[no_mangle]` or `#[export_name]`
/// is accepted when rustc's own expansion data says one of those two macros
/// wrote its identifier, and stopped otherwise, with the symbol named. A
/// symbol the language itself fixes (the panic handler, the allocator shims
/// and the other std-internal symbols) carries the std-internal flag and
/// stays accepted.
///
/// `wasm_direct_core` itself defines the allocator and runtime symbols every
/// guest links against, so its own items are its own to name.
pub fn reject_hand_published_symbols(tcx: TyCtxt<'_>) {
    let is_core = |name: Symbol| name.as_str() == GUEST_CORE;
    if is_core(tcx.crate_name(LOCAL_CRATE))
        || !tcx.crates(()).iter().any(|&krate| is_core(tcx.crate_name(krate)))
    {
        return;
    }
    for def_id in tcx.hir_crate_items(()).definitions() {
        if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn | DefKind::Static { .. })
        {
            continue;
        }
        let no_mangle = find_attr!(tcx, def_id, NoMangle(span) => *span);
        let export_name = find_attr!(tcx, def_id, ExportName { name, span } => (*name, *span));
        let attribute = match (export_name, no_mangle) {
            (Some((_, span)), _) | (None, Some(span)) => span,
            (None, None) => continue,
        };
        if tcx
            .codegen_fn_attrs(def_id)
            .flags
            .contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL)
        {
            continue;
        }
        let identifier = tcx.def_ident_span(def_id).unwrap_or_else(|| tcx.def_span(def_id));
        if written_by_publishing_macro(tcx, identifier) {
            continue;
        }
        let symbol =
            export_name.map_or_else(|| tcx.item_name(def_id.to_def_id()), |(name, _)| name);
        report(tcx, Rejection { attribute, def_id, identifier, symbol });
    }
}

/// Answers whether `#[export]` or `#[offload]` wrote the identifier at `span`.
///
/// The outermost expansion of the identifier's context is the macro that
/// produced its tokens. A guest that writes an item inside the body handed to
/// either macro keeps its own span, and so stays its own author.
fn written_by_publishing_macro(tcx: TyCtxt<'_>, span: Span) -> bool {
    let Some(macro_def_id) = span.ctxt().outer_expn_data().macro_def_id else {
        return false;
    };
    let Some(name) = tcx.opt_item_name(macro_def_id) else {
        return false;
    };
    let krate = tcx.crate_name(macro_def_id.krate);
    PUBLISHING_MACROS
        .iter()
        .any(|(defining, function)| krate.as_str() == *defining && name.as_str() == *function)
}

/// One hand-published symbol and where it was written.
struct Rejection {
    /// The `#[no_mangle]` or `#[export_name]` attribute that publishes it.
    attribute: Span,
    /// The item it publishes.
    def_id: LocalDefId,
    /// The identifier the item is spelled with.
    identifier: Span,
    /// The symbol it publishes.
    symbol: Symbol,
}

fn report(tcx: TyCtxt<'_>, rejection: Rejection) {
    let Rejection { attribute, def_id, identifier, symbol } = rejection;
    tcx.dcx()
        .struct_span_err(attribute, format!("PolyASM rejects the hand-published symbol `{symbol}`"))
        .with_span_label(
            identifier,
            format!("`{}` publishes `{symbol}` by hand", tcx.def_path_str(def_id)),
        )
        .with_note(
            "a guest of `wasm_direct_core` publishes a function only through \
             `#[wasm_direct_core::export]` or `#[wasm_direct_http::offload]`",
        )
        .emit();
}
