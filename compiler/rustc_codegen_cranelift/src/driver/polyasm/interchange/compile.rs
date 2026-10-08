//! Whole-crate interchange compilation and allocator-image orchestration.
//!
//! Every codegen unit is defined into one `polytime-frontend-cranelift`
//! interchange frontend, which is the Cranelift module of that unit; the
//! units merge into the crate's frontend, and the crate's frontend answers
//! the records the object carries.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cranelift_codegen::Context;
use cranelift_codegen::isa::TargetIsa;
use polyasm::object::InterchangeFragment;
use polyasm::object::vocabulary::ClosedAtomic;
use polytime_frontend_cranelift::CraneliftFrontend;
use rustc_codegen_ssa::base::{allocator_kind_for_codegen, allocator_shim_contents};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mono::{MonoItem, is_polyasm_witness_marker};
use rustc_middle::ty::TyCtxt;

use super::model::{CallableKind, CompiledImages, PortableImage, callable_kind};
use crate::debuginfo::TypeDebugContext;
use crate::prelude::Function as ClifFunction;

/// Answers the interchange frontend one codegen unit is defined into.
fn unit(
    isa: &Arc<dyn TargetIsa>,
    namespace: String,
    compiler_functions: BTreeSet<String>,
    callable_kinds: BTreeMap<String, CallableKind>,
    host_functions: BTreeSet<String>,
) -> CraneliftFrontend {
    let mut unit = CraneliftFrontend::from(Arc::clone(isa));
    unit.namespace = namespace;
    unit.compiler_functions = compiler_functions;
    unit.callable_kinds = callable_kinds;
    unit.host_functions = host_functions;
    unit
}

/// Answers the records one crate's frontend lowered.
fn lowered(tcx: TyCtxt<'_>, frontend: CraneliftFrontend) -> PortableImage {
    let (fragment, calls) =
        <(InterchangeFragment, BTreeMap<(String, String, u32), u32>)>::try_from(frontend)
            .unwrap_or_else(|error| tcx.dcx().fatal(error));
    PortableImage { fragment, calls }
}

fn allocator_image(
    tcx: TyCtxt<'_>,
    isa: &Arc<dyn TargetIsa>,
    host_functions: BTreeSet<String>,
) -> Option<PortableImage> {
    let kind = allocator_kind_for_codegen(tcx)?;
    let mut module =
        unit(isa, "polyasm.allocator".to_owned(), BTreeSet::new(), BTreeMap::new(), host_functions);
    crate::allocator::codegen(tcx, &mut module, &allocator_shim_contents(tcx, kind));
    Some(lowered(tcx, module))
}

/// Stops at a record a compiler marker left behind.
///
/// A marker names a compile-time property and owns zero runtime operations, so
/// `compile` drops it from every crate's mono items and every object leaves its
/// body out. Every call to one is replaced before a reference is imported, so a
/// surviving record means the marker escaped as a value. Leaving it would reach
/// final linkage as a bodyless import, where the closed syscall table holds
/// zero descriptors to name the fault.
fn reject_marker_records(tcx: TyCtxt<'_>, markers: &BTreeSet<String>, image: &PortableImage) {
    if let Some(function) =
        image.fragment.functions.iter().find(|item| markers.contains(&item.name))
    {
        tcx.dcx().fatal(format!(
            "PolyASM compiler-witness marker `{}` names a warrant and owns no runtime operation, so it has no address and cannot be used as a value",
            function.name
        ));
    }
}

pub(crate) fn compile(tcx: TyCtxt<'_>) -> CompiledImages {
    let isa = crate::build_isa(tcx.sess, false);
    let host_functions =
        crate::driver::polyasm::debug::syscall_aliases(tcx).into_keys().collect::<BTreeSet<_>>();
    let mut image = unit(&isa, String::new(), BTreeSet::new(), BTreeMap::new(), BTreeSet::new());
    for cgu in tcx.collect_and_partition_mono_items(()).codegen_units {
        let mono_items = cgu
            .items_in_deterministic_order(tcx)
            .into_iter()
            .filter(|(item, _)| match item {
                MonoItem::Fn(instance) => !is_polyasm_witness_marker(tcx, instance.def_id()),
                MonoItem::Static(_) | MonoItem::GlobalAsm(_) => true,
            })
            .collect::<Vec<_>>();
        let mut callable_kinds = BTreeMap::new();
        for (item, _) in &mono_items {
            let MonoItem::Fn(instance) = item else { continue };
            let name = tcx.symbol_name(*instance).name.to_string();
            let kind = callable_kind(tcx.def_kind(instance.def_id()));
            if kind == CallableKind::Unknown {
                tcx.dcx()
                    .fatal(format!("PolyASM MonoItem `{name}` has a non-callable definition kind"));
            }
            if let Some(previous) = callable_kinds.insert(name.clone(), kind)
                && previous != kind
            {
                tcx.dcx().fatal(format!(
                    "conflicting MonoItem callable kinds for `{name}` in one codegen unit"
                ));
            }
        }
        let compiler_functions = callable_kinds.keys().cloned().collect::<BTreeSet<_>>();
        if let Some(name) =
            compiler_functions.iter().find(|name| ClosedAtomic::named(name).is_some())
        {
            tcx.dcx()
                .fatal(format!("user code cannot define reserved PolyASM atomic helper `{name}`"));
        }
        let mut module = unit(
            &isa,
            cgu.name().to_string(),
            compiler_functions,
            callable_kinds,
            host_functions.clone(),
        );
        let mut type_debug = TypeDebugContext::default();
        let mut functions = Vec::new();
        let mut global_asm = String::new();

        crate::driver::predefine_mono_items(tcx, &mut module, &mono_items);
        for (mono_item, _item_data) in mono_items {
            match mono_item {
                MonoItem::Fn(instance) => {
                    let flags = tcx.codegen_instance_attrs(instance.def).flags;
                    if flags.contains(CodegenFnAttrFlags::NAKED) {
                        tcx.dcx().fatal("naked functions are not representable in PolyASM");
                    }
                    let function = crate::base::codegen_fn(
                        tcx,
                        cgu.name(),
                        None,
                        &mut type_debug,
                        ClifFunction::new(),
                        &mut module,
                        instance,
                    );
                    functions.push(function);
                }
                MonoItem::Static(def_id) => {
                    crate::constant::codegen_static(tcx, &mut module, def_id);
                }
                MonoItem::GlobalAsm(_) => {
                    tcx.dcx().fatal("global assembly is not representable in PolyASM");
                }
            }
        }
        crate::main_shim::maybe_create_entry_wrapper(tcx, &mut module, false, cgu.is_primary());

        let mut context = Context::new();
        for function in functions {
            crate::base::compile_fn(
                &tcx.sess.prof,
                tcx.dcx(),
                tcx.output_filenames(()),
                crate::pretty_clif::should_write_ir(tcx.sess),
                &mut context,
                &mut module,
                None,
                &mut global_asm,
                function,
            );
        }
        if !global_asm.trim().is_empty() {
            tcx.dcx().fatal("inline assembly is not representable in PolyASM");
        }
        image.extend([module]);
    }

    let markers = crate::driver::polyasm::debug::witness_markers(tcx);
    let allocator = allocator_image(tcx, &isa, host_functions);
    let regular = lowered(tcx, image);
    reject_marker_records(tcx, &markers, &regular);
    if let Some(allocator) = &allocator {
        reject_marker_records(tcx, &markers, allocator);
    }

    CompiledImages { allocator, regular }
}
