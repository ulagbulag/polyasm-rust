//! Final `.poly` publication and ordinary Rust archive linkage.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use ::object::read::archive::ArchiveFile;
use polyasm_format::ir::Program;
use polytime_core::frontend::Frontend;
use polytime_core::geometry::stack_base_for;
use polytime_frontend_cranelift::CraneliftFrontend;
use rustc_codegen_ssa::back::archive::ArArchiveBuilderBuilder;
use rustc_codegen_ssa::back::link::{each_linked_rlib, ensure_removed, link_binary};
use rustc_codegen_ssa::base::needs_allocator_shim_for_linking;
use rustc_codegen_ssa::{CompiledModules, CrateInfo};
use rustc_metadata::EncodedMetadata;
use rustc_session::Session;
use rustc_session::config::{OutFileName, OutputFilenames};
use rustc_session::output::out_filename;
use rustc_structures::CrateType;

use super::object;

pub(crate) fn publish(
    sess: &Session,
    compiled_modules: CompiledModules,
    mut crate_info: CrateInfo,
    metadata: EncodedMetadata,
    outputs: &OutputFilenames,
) {
    let mut published_direct = false;
    for direct_kind in [CrateType::Executable, CrateType::Cdylib] {
        if !crate_info.crate_types.contains(&direct_kind) || !outputs.outputs.should_link() {
            continue;
        }
        let Some(local_object) =
            compiled_modules.modules.first().and_then(|module| module.object.as_ref())
        else {
            sess.dcx().fatal("PolyASM code generation produced no module");
        };
        let local = read_local_object(sess, local_object);
        let allocator_object =
            needs_allocator_shim_for_linking(&crate_info.dependency_formats, direct_kind)
                .then(|| {
                    compiled_modules
                        .allocator_module
                        .as_ref()
                        .and_then(|module| module.object.as_deref())
                })
                .flatten();
        let objects = linked_objects(sess, local, allocator_object, &crate_info, direct_kind);
        let width = super::pointer_width(sess);
        let mut frontend = CraneliftFrontend::from(crate::build_isa(sess, false));
        frontend.heap_end = stack_base_for(width);
        frontend.library = direct_kind == CrateType::Cdylib;
        frontend.width = width;
        let program = frontend
            .translate(&objects)
            .unwrap_or_else(|error| sess.dcx().fatal(error.to_string()));
        let Some(image) = program.encoded().map(<[u8]>::to_vec) else {
            sess.dcx().fatal("the PolyASM frontend answered a program without its container")
        };
        let output = out_filename(sess, direct_kind, outputs, crate_info.local_crate_name);
        let image = match &output {
            OutFileName::Real(path) => match rustc_session::polyasm::port_image(sess, path) {
                Some(port) => bundle(sess, &image, &port),
                None => image,
            },
            OutFileName::Stdout => image,
        };
        write_output(sess, &image, &output);
        published_direct = true;
        if sess.opts.json_artifact_notifications
            && let OutFileName::Real(path) = &output
        {
            sess.dcx().emit_artifact_notification(path, "link");
        }
    }

    crate_info
        .crate_types
        .retain(|kind| !matches!(kind, CrateType::Executable | CrateType::Cdylib));
    if !crate_info.crate_types.is_empty() {
        link_binary(
            sess,
            &ArArchiveBuilderBuilder,
            compiled_modules,
            crate_info,
            metadata,
            outputs,
            "cranelift",
        );
    } else if published_direct {
        remove_direct_temps(sess, &compiled_modules);
    }
}

/// Bundles this session's image with the image its companion compiled at the
/// other pointer width, one port per width.
fn bundle(sess: &Session, image: &[u8], port: &Path) -> Vec<u8> {
    let companion = fs::read(port).unwrap_or_else(|error| {
        sess.dcx().fatal(format!("failed to read the PolyASM port {}: {error}", port.display()))
    });
    let width = u8::try_from(rustc_session::polyasm::port_width())
        .ok()
        .and_then(|bits| polyasm_format::syscall::PointerWidth::try_from(bits).ok())
        .unwrap_or_else(|| sess.dcx().fatal("the PolyASM port tuple names no PolyASM width"));
    polyasm_format::port::bundle(&[
        polyasm_format::port::Port { width: super::pointer_width(sess), image },
        polyasm_format::port::Port { width, image: &companion },
    ])
    .unwrap_or_else(|error| sess.dcx().fatal(format!("bundling the PolyASM ports failed: {error}")))
}

fn read_local_object(sess: &Session, path: &Path) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        sess.dcx().fatal(format!("failed to read local PolyASM object {}: {error}", path.display()))
    })
}

/// Publishes one linked PolyASM image under its final output name.
///
/// A `.poly` image is bytes rather than text, so it is written whole and
/// apart from the textual `OutFileName::overwrite` path: a real output name
/// receives the image directly, and `-o -` receives it on stdout, the one
/// other name `out_filename` returns.
fn write_output(sess: &Session, image: &[u8], output: &OutFileName) {
    match output {
        OutFileName::Real(path) => fs::write(path, image).unwrap_or_else(|error| {
            sess.dcx().fatal(format!("failed to write PolyASM image {}: {error}", path.display()))
        }),
        OutFileName::Stdout => {
            let mut stdout = io::stdout().lock();
            stdout.write_all(image).and_then(|()| stdout.flush()).unwrap_or_else(|error| {
                sess.dcx().fatal(format!("failed to write the PolyASM image to stdout: {error}"))
            })
        }
    }
}

fn remove_direct_temps(sess: &Session, compiled_modules: &CompiledModules) {
    if sess.opts.cg.save_temps {
        return;
    }
    for module in compiled_modules.modules.iter().chain(compiled_modules.allocator_module.iter()) {
        for path in [
            module.object.as_deref(),
            module.global_asm_object.as_deref(),
            module.dwarf_object.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            ensure_removed(sess.dcx(), path);
        }
    }
}

/// Writes every object of the crate graph back to back, the local one first,
/// which is the run the interchange frontend translates.
fn linked_objects(
    sess: &Session,
    local: Vec<u8>,
    allocator: Option<&Path>,
    crate_info: &CrateInfo,
    crate_type: CrateType,
) -> Vec<u8> {
    let mut objects = local;
    if let Some(allocator) = allocator {
        let allocator_bytes = fs::read(allocator).unwrap_or_else(|error| {
            sess.dcx().fatal(format!(
                "failed to read local PolyASM allocator object {}: {error}",
                allocator.display()
            ))
        });
        objects.extend_from_slice(&allocator_bytes);
    }
    let mut rlibs = Vec::new();
    each_linked_rlib(crate_info, Some(crate_type), &mut |_, path| {
        rlibs.push(path.to_path_buf());
    })
    .unwrap_or_else(|_| {
        sess.dcx().fatal("failed to enumerate the statically linked PolyASM dependency rlibs")
    });
    rlibs.sort();
    rlibs.dedup();
    for path in rlibs {
        let bytes = fs::read(&path).unwrap_or_else(|error| {
            sess.dcx().fatal(format!(
                "failed to read PolyASM dependency archive {}: {error}",
                path.display()
            ))
        });
        let archive = ArchiveFile::parse(bytes.as_slice()).unwrap_or_else(|error| {
            sess.dcx()
                .fatal(format!("invalid PolyASM dependency archive {}: {error}", path.display()))
        });
        let mut found = 0_usize;
        let mut bitcode = 0_usize;
        for member in archive.members() {
            let member = member.unwrap_or_else(|error| {
                sess.dcx().fatal(format!(
                    "invalid member in PolyASM dependency archive {}: {error}",
                    path.display()
                ))
            });
            let data = member.data(bytes.as_slice()).unwrap_or_else(|error| {
                sess.dcx().fatal(format!(
                    "cannot read member in PolyASM dependency archive {}: {error}",
                    path.display()
                ))
            });
            if is_llvm_bitcode(data) {
                bitcode += 1;
                continue;
            }
            if !data.starts_with(&object::MAGIC) {
                continue;
            }
            found += 1;
            objects.extend_from_slice(data);
        }
        if found == 0 && bitcode != 0 {
            sess.dcx().fatal(format!(
                "linked Rust archive {} carries {bitcode} LLVM bitcode member(s) and no relocatable PolyASM object, which is what a sysroot built by the LLVM backend holds and what that backend links from; this crate is being lowered by the Cranelift backend against it. Pass `-Zcodegen-backend=llvm` to compile against this sysroot, or link a sysroot the Cranelift backend built",
                path.display()
            ));
        }
        if found == 0 {
            sess.dcx().fatal(format!(
                "linked Rust archive {} contains no relocatable PolyASM object; rebuild the complete target sysroot with this toolchain",
                path.display()
            ));
        }
    }
    objects
}

/// Whether these bytes are an LLVM bitcode module.
///
/// A sysroot the LLVM backend built for a PolyASM target holds bitcode in the
/// place this backend holds relocatable objects, on purpose: that backend
/// gathers the bitcode of the whole crate graph and runs the emitter over it
/// once, so its rlibs carry bitcode alone. Reading one of those archives here
/// therefore means the wrong backend rather than a broken sysroot, and
/// telling the two apart is the difference between a message that names the
/// flag to pass and one that sends the reader to rebuild a sysroot that was
/// already right.
///
/// Both spellings are accepted, matching what the LLVM backend's own gathering
/// accepts: the raw `BC\xc0\xde` magic, and the wrapper LLVM puts in front of
/// it on some targets.
fn is_llvm_bitcode(data: &[u8]) -> bool {
    data.starts_with(b"BC\xc0\xde") || data.starts_with(&0x0B17C0DEu32.to_le_bytes())
}
