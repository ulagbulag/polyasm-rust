//! Final `.poly` publication for the LLVM backend.
//!
//! The Cranelift backend builds a PolyASM object itself and links the objects
//! it finds in the rlibs. This backend translates LLVM IR into PolyASM in its
//! own process through `polytime-frontend-llvm`'s `LlvmFrontend`, reached
//! through its `polytime-core` `Frontend` trait: the frontend binds the LLVM
//! this compiler links, so the modules it reads are the bitcode this LLVM
//! writes, and its answer is the program the `.poly` container spells.
//!
//! The lowering is handed modules, and every record a call names sits in one
//! of them. So this gathers the whole graph the way the Cranelift backend
//! gathers its own objects: the modules this run produced, plus the bitcode
//! members of every rlib it links, and the lowering links them into one
//! before it lowers anything. LTO stays optional; a profile that turns it on
//! leaves less for the lowering to join.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{fs, io};

use ::object::read::archive::ArchiveFile;
use polyasm_format::ir::Program as _;
use polytime_core::frontend::Frontend as _;
use polytime_frontend_llvm::LlvmFrontend;
use rustc_codegen_ssa::back::archive::ArArchiveBuilderBuilder;
use rustc_codegen_ssa::back::link::{each_linked_rlib, ensure_removed, link_binary};
use rustc_codegen_ssa::{CompiledModules, CrateInfo};
use rustc_metadata::EncodedMetadata;
use rustc_session::Session;
use rustc_session::config::{OutFileName, OutputFilenames};
use rustc_session::output::out_filename;
use rustc_structures::CrateType;

/// The crate-root source item rustc selects as a final PolyASM entry.
const ENTRY_ITEM: &str = "polyasm_entry";

/// The program name the lowering settings parse under.
const LOWERING_NAME: &str = "polytime-frontend-llvm";

/// Whether this session emits PolyASM rather than a platform binary.
pub(crate) fn is_target(sess: &Session) -> bool {
    sess.is_polyasm_target()
}

/// Publishes the `.poly` image and links whatever archives were also asked
/// for.
pub(crate) fn publish(
    sess: &Session,
    compiled_modules: CompiledModules,
    mut crate_info: CrateInfo,
    metadata: EncodedMetadata,
    outputs: &OutputFilenames,
) {
    let mut published = false;
    for direct_kind in [CrateType::Executable, CrateType::Cdylib] {
        if !crate_info.crate_types.contains(&direct_kind) || !outputs.outputs.should_link() {
            continue;
        }
        let staged = outputs.temp_path_for_diagnostic("polyasm-bitcode");
        let modules = bitcode(sess, &compiled_modules, &crate_info, direct_kind, &staged);
        let output = out_filename(sess, direct_kind, outputs, crate_info.local_crate_name);
        let crate_name = crate_info.local_crate_name.as_str();
        let entry =
            format!("{}{}{}{}", crate_name.len(), crate_name, ENTRY_ITEM.len(), ENTRY_ITEM,);
        let port = match &output {
            OutFileName::Real(path) => rustc_session::polyasm::port_image(sess, path),
            OutFileName::Stdout => None,
        };
        let bytes = lower(sess, &modules, entry, port.as_ref());
        if !sess.opts.cg.save_temps {
            let _ = fs::remove_dir_all(&staged);
        }
        write_output(sess, &bytes, &output);
        published = true;
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
            "llvm",
        );
    } else if published {
        remove_temps(sess, &compiled_modules);
    }
}

/// Answers every bitcode module the image is lowered from.
///
/// The modules this run produced come first, then one file per bitcode member
/// of every rlib it links, unpacked into `staged` because the lowering reads
/// paths and an archive member lives inside its archive. An rlib carries its
/// bitcode member when this toolchain built its sysroot, and the stop here
/// names any other rlib before an image calls into records it lacks.
fn bitcode(
    sess: &Session,
    compiled_modules: &CompiledModules,
    crate_info: &CrateInfo,
    crate_type: CrateType,
    staged: &PathBuf,
) -> Vec<PathBuf> {
    let mut modules = Vec::new();
    for module in compiled_modules.modules.iter().chain(compiled_modules.allocator_module.iter()) {
        match module.object.as_deref() {
            Some(path) => modules.push(path.to_owned()),
            None => sess.dcx().fatal(format!(
                "PolyASM code generation left module `{}` without bitcode",
                module.name
            )),
        }
    }
    if modules.is_empty() {
        sess.dcx().fatal("PolyASM code generation produced no module");
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
    if !rlibs.is_empty() {
        fs::create_dir_all(staged).unwrap_or_else(|error| {
            sess.dcx().fatal(format!(
                "failed to make the PolyASM staging directory {}: {error}",
                staged.display()
            ))
        });
    }
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
            if !is_bitcode(data) {
                continue;
            }
            let unpacked = staged.join(format!("{}.{found}.bc", stem(&path)));
            fs::write(&unpacked, data).unwrap_or_else(|error| {
                sess.dcx().fatal(format!(
                    "failed to stage PolyASM bitcode {}: {error}",
                    unpacked.display()
                ))
            });
            modules.push(unpacked);
            found += 1;
        }
        if found == 0 {
            sess.dcx().fatal(format!(
                "linked Rust archive {} carries its LLVM bitcode member once this \
                 toolchain builds the complete target sysroot; rebuild it with this toolchain",
                path.display()
            ));
        }
    }
    modules
}

/// Whether these bytes are an LLVM bitcode module.
///
/// Both spellings are accepted: the raw `BC\xc0\xde` magic, and the bitcode
/// wrapper LLVM puts in front of it on some targets.
fn is_bitcode(data: &[u8]) -> bool {
    data.starts_with(b"BC\xc0\xde") || data.starts_with(&0x0B17C0DEu32.to_le_bytes())
}

/// Answers a file name that stays distinct when two archives are staged.
fn stem(path: &Path) -> String {
    path.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Lowers the modules into one `.poly` container and answers its bytes.
///
/// The first module is the source the frontend translates, and every other
/// module is linked into it first. `port` names the image the companion
/// session compiled at the other pointer width of a width-unfixed crate; the
/// image is bundled beside it as one port per width. The settings are the
/// frontend's own fields, read from its `POLYTIME_LLVM_*` variables, with this
/// crate's entry and with identical bodies kept apart.
fn lower(sess: &Session, modules: &[PathBuf], entry: String, port: Option<&PathBuf>) -> Vec<u8> {
    let Some((first, rest)) = modules.split_first() else {
        sess.dcx().fatal("PolyASM code generation produced no module")
    };
    let mut arguments: Vec<OsString> = vec![
        LOWERING_NAME.into(),
        "--llvm-entry".into(),
        entry.into(),
        "--llvm-merge-functions".into(),
        "false".into(),
    ];
    for module in rest {
        arguments.push("--llvm-link".into());
        arguments.push(module.into());
    }
    let frontend = <LlvmFrontend as clap::Parser>::try_parse_from(arguments).unwrap_or_else(
        |error| sess.dcx().fatal(format!("the PolyASM lowering settings read with {error}")),
    );
    let source = fs::read(first).unwrap_or_else(|error| {
        sess.dcx().fatal(format!("failed to read PolyASM module {}: {error}", first.display()))
    });
    let program = frontend
        .translate(&source)
        .unwrap_or_else(|error| sess.dcx().fatal(format!("the PolyASM lowering stops: {error}")));
    let Some(image) = program.encoded() else {
        sess.dcx().fatal("the PolyASM lowering answers a program without its container")
    };
    match port {
        Some(port) => bundle(sess, image, port),
        None => image.to_vec(),
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
    let own = if sess.target.pointer_width == 64 {
        polyasm_format::syscall::PointerWidth::Bits64
    } else {
        polyasm_format::syscall::PointerWidth::Bits32
    };
    polyasm_format::port::bundle(&[
        polyasm_format::port::Port { width: own, image },
        polyasm_format::port::Port { width, image: &companion },
    ])
    .unwrap_or_else(|error| sess.dcx().fatal(format!("bundling the PolyASM ports failed: {error}")))
}

/// Writes one published image under its final output name.
///
/// A `.poly` image is bytes, distinct from text, so it is written whole in
/// place of the textual overwrite path; `-o -` receives it on stdout, which is
/// the only other name `out_filename` returns.
fn write_output(sess: &Session, image: &[u8], output: &OutFileName) {
    match output {
        OutFileName::Real(path) => fs::write(path, image).unwrap_or_else(|error| {
            sess.dcx()
                .fatal(format!("failed to write the PolyASM image {}: {error}", path.display()))
        }),
        OutFileName::Stdout => {
            let mut stdout = io::stdout().lock();
            stdout.write_all(image).and_then(|()| stdout.flush()).unwrap_or_else(|error| {
                sess.dcx().fatal(format!("failed to write the PolyASM image to stdout: {error}"))
            })
        }
    }
}

/// Removes the per-module temporaries a published image leaves behind.
fn remove_temps(sess: &Session, compiled_modules: &CompiledModules) {
    if sess.opts.cg.save_temps {
        return;
    }
    for module in compiled_modules.modules.iter().chain(compiled_modules.allocator_module.iter()) {
        for path in [
            module.object.as_deref(),
            module.bytecode.as_deref(),
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
