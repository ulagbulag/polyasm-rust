//! Native PolyASM emission from rustc's monomorphized MIR graph.
//!
//! This path runs inside rustc alone, apart from wrapper compilers and platform linkers.

use std::collections::BTreeMap;
use std::fs;

use rustc_codegen_ssa::{CompiledModule, CompiledModules, ModuleKind};
use rustc_middle::dep_graph::WorkProductMap;
use rustc_middle::mono::CodegenUnitNameBuilder;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_session::config::{OutputFilenames, OutputType, OutputTypes};
use rustc_session::{EarlySession, Session};
use rustc_span::def_id::LOCAL_CRATE;

mod debug;
mod domain;
mod interchange;
mod lifecycle;
mod link;
mod object;

pub(crate) const MEMORY_COPY8_INTRINSIC: &str = "__rustc_polyasm_memory_copy8";
pub(crate) const MEMORY_COPY64_INTRINSIC: &str = "__rustc_polyasm_memory_copy64";

pub(crate) struct OngoingCodegen {
    allocator_bytes: Option<Vec<u8>>,
    allocator_module_name: String,
    bytes: Vec<u8>,
    module_name: String,
}

pub(crate) fn is_target(sess: &EarlySession) -> bool {
    sess.is_polyasm_target()
}

/// Names the Cranelift interchange triple this PolyASM image lowers through.
///
/// Cranelift gives a pointer a type only under a concrete ISA, and Pulley is
/// the bytecode ISA whose two widths are the two widths PolyASM admits. The
/// triple is therefore read off the width the target spec fixed in place of
/// being pinned: `polyasm64-unknown-unknown` lowers through the 64-bit Pulley
/// and the other two through the 32-bit one, which makes an emitted `.poly`
/// differ between the widths beyond its triple's name.
/// The interchange triple leaves the operating system out. Pulley uses that honest
/// target fact to select the compact integer register bank that fits PolyASM's
/// narrow machines; ordinary hosted Pulley triples keep their full bank.
///
/// The width is compared against a literal because `Target::pointer_width` is
/// a bare `u32` in this compiler and every rustc site that reads it compares it
/// the same way; naming the two admitted widths here would add a second
/// spelling of them to the tree beside the first.
/// Names the pointer width the closed syscall table holds this image to.
///
/// The table leaves the scalar of a slot that carries a guest address open, so the
/// width reaches it from the target that fixed it. This is the one place the
/// compiler answers that, and [`interchange_triple`] reads the same field, so
/// the ISA a body lowers through and the rule its host calls follow stay
/// together.
pub(crate) fn pointer_width(sess: &Session) -> polyasm_format::syscall::PointerWidth {
    if sess.target.pointer_width == 64 {
        polyasm_format::syscall::PointerWidth::Bits64
    } else {
        polyasm_format::syscall::PointerWidth::Bits32
    }
}

pub(crate) fn interchange_triple(sess: &Session) -> &'static str {
    if sess.target.pointer_width == 64 { "pulley64-unknown-none" } else { "pulley32-unknown-none" }
}

pub(crate) fn run(tcx: TyCtxt<'_>) -> Box<OngoingCodegen> {
    rustc_codegen_ssa::polyasm::reject_hand_published_symbols(tcx);
    let crate_name = tcx.crate_name(LOCAL_CRATE).to_string();
    let module_name = CodegenUnitNameBuilder::new(tcx)
        .build_cgu_name(LOCAL_CRATE, ["polyasm"], Some("cgu"))
        .to_string();
    let allocator_module_name = CodegenUnitNameBuilder::new(tcx)
        .build_cgu_name(LOCAL_CRATE, &["crate"], Some("allocator"))
        .to_string();
    let (bytes, allocator_bytes) = if !tcx.sess.opts.output_types.should_codegen() {
        (Vec::new(), None)
    } else {
        let interchange::CompiledImages { allocator, regular } = interchange::compile(tcx);
        let negative_witness = debug::negative_witness(tcx);
        let relocatable = with_no_trimmed_paths!(object::Object {
            crate_name: crate_name.clone(),
            domain: domain::collect(tcx, &regular, &negative_witness),
            entry: debug::entry_symbol(tcx),
            offload_roots: debug::offload_roots(tcx),
            interchange: regular.relocatable(),
            syscall_aliases: debug::syscall_aliases(tcx),
            work: lifecycle::collect(tcx, &regular),
        });
        let bytes = encode_object(tcx, relocatable);
        let allocator_bytes = allocator.map(|image| {
            encode_object(
                tcx,
                object::Object {
                    crate_name: format!("{crate_name}.allocator"),
                    domain: domain::Fragment {
                        call_requests: Vec::new(),
                        functions: Vec::new(),
                        index_names: Vec::new(),
                        sources: Vec::new(),
                    },
                    entry: None,
                    offload_roots: Vec::new(),
                    interchange: image.relocatable(),
                    syscall_aliases: BTreeMap::new(),
                    work: None,
                },
            )
        });
        (bytes, allocator_bytes)
    };
    Box::new(OngoingCodegen { allocator_bytes, allocator_module_name, bytes, module_name })
}

fn encode_object(tcx: TyCtxt<'_>, object: object::Object) -> Vec<u8> {
    object.encode().unwrap_or_else(|error| {
        tcx.dcx().fatal(format!("failed to encode relocatable PolyASM object: {error}"))
    })
}

impl OngoingCodegen {
    pub(crate) fn join(
        self,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CompiledModules, WorkProductMap) {
        let OngoingCodegen { allocator_bytes, allocator_module_name, bytes, module_name } = self;
        let mut modules = Vec::new();
        let mut allocator_module = None;
        if sess.opts.output_types.should_codegen() {
            let object = outputs.temp_path_for_cgu(OutputType::Object, &module_name);
            fs::write(&object, bytes).unwrap_or_else(|error| {
                sess.dcx()
                    .fatal(format!("failed to write PolyASM module {}: {error}", object.display()))
            });
            modules.push(CompiledModule {
                name: module_name,
                kind: ModuleKind::Regular,
                object: Some(object),
                global_asm_object: None,
                dwarf_object: None,
                bytecode: None,
                assembly: None,
                llvm_ir: None,
            });
            allocator_module = allocator_bytes.map(|bytes| {
                let object = outputs.temp_path_for_cgu(OutputType::Object, &allocator_module_name);
                fs::write(&object, bytes).unwrap_or_else(|error| {
                    sess.dcx().fatal(format!(
                        "failed to write PolyASM allocator module {}: {error}",
                        object.display()
                    ))
                });
                CompiledModule {
                    name: allocator_module_name,
                    kind: ModuleKind::Allocator,
                    object: Some(object),
                    global_asm_object: None,
                    dwarf_object: None,
                    bytecode: None,
                    assembly: None,
                    llvm_ir: None,
                }
            });
        }
        let compiled_modules = CompiledModules { modules, allocator_module };
        // The PolyASM backend emits one crate-wide object in place of rustc's
        // individually reusable CGUs. Materialize requested outputs through
        // the standard path, and keep orphan incremental work products under
        // a synthetic CGU dep-node unpublished.
        // Cranelift's IR request already wrote its per-function `.clif`
        // files. It supplies zero LLVM modules for the standard finalizer to
        // copy from a synthetic `.rcgu.ll` path.
        let mut final_outputs = outputs.clone();
        final_outputs.outputs = OutputTypes::new(
            &outputs
                .outputs
                .iter()
                .filter(|(kind, _)| **kind != OutputType::LlvmAssembly)
                .map(|(kind, path)| (*kind, path.clone()))
                .collect::<Vec<_>>(),
        );
        rustc_codegen_ssa::back::write::produce_final_output_artifacts(
            sess,
            &compiled_modules,
            &final_outputs,
        );
        (compiled_modules, WorkProductMap::default())
    }
}

pub(crate) use link::publish;
