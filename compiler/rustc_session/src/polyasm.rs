//! The companion session that gives a width-unfixed PolyASM crate its other
//! pointer width.
//!
//! `polyasm-unknown-unknown` leaves the pointer width to the machine an image
//! lands on, and one session fixes every layout at one width. The driver
//! therefore compiles such a crate twice: the session itself at the tuple's
//! own width, and a companion `rustc` under [`POLYASM_PORT_TUPLE`] whose
//! outputs, dependencies and incremental state sit one directory below the
//! session's own, in a directory named after that tuple ([`port_path`]). A
//! dependency of the companion is the companion output of an earlier crate,
//! so each width reads the metadata and the archives of its own width
//! throughout the crate graph.
//!
//! The session waits for its companion before it announces its metadata and
//! before it publishes a final image ([`join`]). A dependent crate and its own
//! companion therefore find the companion outputs in place, and a final image
//! bundles both compilations as ports (`polyasm_format::port`).

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use rustc_target::spec::{POLYASM_PORT_TUPLE, POLYASM_UNFIXED_TUPLE, Target, TargetTuple};

use crate::config::{Options, OutputType};
use crate::{EarlyDiagCtxt, Session};

/// The companion `rustc` of this process, while it runs.
struct Companion {
    /// The running companion.
    child: Child,

    /// Where the companion writes its diagnostics.
    log: PathBuf,
}

/// The one companion this process starts, held until [`join`] or the guard
/// takes it.
static COMPANION: Mutex<Option<Companion>> = Mutex::new(None);

/// Stops a companion the session leaves behind.
///
/// A session that ends before it joins its companion (an error, a fatal
/// diagnostic) ends the companion with it, so a failed build leaves one
/// process fewer behind and the next build starts from a quiet directory.
pub struct CompanionGuard(());

impl Drop for CompanionGuard {
    fn drop(&mut self) {
        if let Some(mut companion) = take() {
            let _ = companion.child.kill();
            let _ = companion.child.wait();
            let _ = fs::remove_file(&companion.log);
        }
    }
}

fn take() -> Option<Companion> {
    COMPANION.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take()
}

/// Answers the path the companion uses for one output or input path of the
/// session: the same file name, one directory below, in the directory named
/// after [`POLYASM_PORT_TUPLE`].
pub fn port_path(path: &Path) -> PathBuf {
    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => parent.join(POLYASM_PORT_TUPLE).join(name),
        _ => Path::new(POLYASM_PORT_TUPLE).join(path),
    }
}

/// Answers the directory the companion uses for one directory of the session.
pub fn port_dir(dir: &Path) -> PathBuf {
    dir.join(POLYASM_PORT_TUPLE)
}

/// Answers the pointer width, in bits, the companion compiles at.
pub fn port_width() -> u64 {
    Target::expect_builtin(&TargetTuple::from_tuple(POLYASM_PORT_TUPLE)).pointer_width.into()
}

/// Answers whether these options compile for the width-unfixed tuple.
pub fn is_unfixed(opts: &Options) -> bool {
    matches!(
        &opts.target_triple,
        TargetTuple::TargetTuple(tuple) if tuple.as_str() == POLYASM_UNFIXED_TUPLE
    )
}

/// Starts the companion of one session.
///
/// The companion runs for a session of the width-unfixed tuple that reads its
/// crate from a file and links an output; a session that only checks,
/// prints or reads standard input runs alone. `args` are the session's
/// expanded arguments after the program name, and `out_dir` is the
/// directory the session writes into.
pub fn start(
    early_dcx: &EarlyDiagCtxt,
    args: &[String],
    opts: &Options,
    from_file: bool,
    out_dir: &Path,
) -> Option<CompanionGuard> {
    if !is_unfixed(opts)
        || !from_file
        || !opts.prints.is_empty()
        || !opts.output_types.contains_key(&OutputType::Exe)
    {
        return None;
    }
    let dir = port_dir(out_dir);
    fs::create_dir_all(&dir).unwrap_or_else(|error| {
        early_dcx.early_fatal(format!(
            "the `{POLYASM_PORT_TUPLE}` companion directory {} cannot be made: {error}",
            dir.display()
        ))
    });
    let log = dir.join(format!("rustc-{}.log", std::process::id()));
    let written = File::create(&log).unwrap_or_else(|error| {
        early_dcx.early_fatal(format!(
            "the `{POLYASM_PORT_TUPLE}` companion log {} cannot be made: {error}",
            log.display()
        ))
    });
    let program = std::env::current_exe().unwrap_or_else(|error| {
        early_dcx.early_fatal(format!("this `rustc` answers no path of its own: {error}"))
    });
    let child = Command::new(&program)
        .args(companion_args(args, out_dir))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(written)
        .spawn()
        .unwrap_or_else(|error| {
            early_dcx.early_fatal(format!(
                "the `{POLYASM_PORT_TUPLE}` companion {} does not start: {error}",
                program.display()
            ))
        });
    *COMPANION.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) =
        Some(Companion { child, log });
    Some(CompanionGuard(()))
}

/// Waits for the companion of this session, once.
///
/// A companion that stops with a failure stops the session as well: its
/// diagnostics reach the error stream whole, in the format the session was
/// asked for, followed by one diagnostic of the session naming the companion.
pub fn join(sess: &Session) {
    let Some(mut companion) = take() else {
        return;
    };
    let status = companion.child.wait();
    let log = fs::read(&companion.log).unwrap_or_default();
    let _ = fs::remove_file(&companion.log);
    match status {
        Ok(status) if status.success() => {}
        status => {
            let mut stderr = std::io::stderr().lock();
            let _ = stderr.write_all(&log);
            let _ = stderr.flush();
            sess.dcx().fatal(format!(
                "the `{POLYASM_PORT_TUPLE}` companion of this `{POLYASM_UNFIXED_TUPLE}` \
                 compilation ended with {status:?}"
            ));
        }
    }
}

/// Answers where the companion left its image for one final image of the
/// session, once the companion is joined.
///
/// A session of a fixed tuple, or one whose companion stayed idle, answers
/// `None` and publishes its own image alone.
pub fn port_image(sess: &Session, output: &Path) -> Option<PathBuf> {
    if !is_unfixed(&sess.opts) {
        return None;
    }
    join(sess);
    let image = port_path(output);
    image.is_file().then_some(image)
}

/// Rewrites the session's arguments into the companion's.
///
/// The tuple becomes [`POLYASM_PORT_TUPLE`]; every output, every archive and
/// metadata dependency, every search directory that holds companion outputs
/// and the incremental directory move one directory down ([`port_path`],
/// [`port_dir`]). Every other argument carries over as written, so both
/// widths compile the same crate under the same options.
fn companion_args(args: &[String], out_dir: &Path) -> Vec<String> {
    let mut rewritten = Vec::with_capacity(args.len());
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let Some((name, value)) = valued(arg, &mut iter) else {
            rewritten.push(arg.clone());
            continue;
        };
        let value = match name {
            "--target" => POLYASM_PORT_TUPLE.to_owned(),
            "--out-dir" => path_string(&port_dir(Path::new(&value))),
            "-o" => path_string(&port_path(Path::new(&value))),
            "--extern" => match value.split_once('=') {
                Some((crate_name, path)) if path.ends_with(".rlib") || path.ends_with(".rmeta") => {
                    format!("{crate_name}={}", path_string(&port_path(Path::new(path))))
                }
                _ => value,
            },
            "--emit" => value
                .split(',')
                .map(|item| match item.split_once('=') {
                    Some((kind, path)) => {
                        format!("{kind}={}", path_string(&port_path(Path::new(path))))
                    }
                    None => item.to_owned(),
                })
                .collect::<Vec<_>>()
                .join(","),
            "-L" => {
                let (kind, dir) = match value.split_once('=') {
                    Some((kind, dir)) => (Some(kind), dir),
                    None => (None, value.as_str()),
                };
                let dir = Path::new(dir);
                if dir == out_dir || port_dir(dir).is_dir() {
                    let dir = path_string(&port_dir(dir));
                    kind.map_or(dir.clone(), |kind| format!("{kind}={dir}"))
                } else {
                    value
                }
            }
            _ => match value.strip_prefix("incremental=") {
                Some(dir) => format!("incremental={}", path_string(&port_dir(Path::new(dir)))),
                None => value,
            },
        };
        rewritten.push(name.to_owned());
        rewritten.push(value);
    }
    rewritten
}

/// Splits one valued option into its name and its value, reading the value
/// from the next argument where the option stands alone.
fn valued<'a>(
    arg: &str,
    rest: &mut impl Iterator<Item = &'a String>,
) -> Option<(&'static str, String)> {
    const LONG: [&str; 5] = ["--target", "--out-dir", "--extern", "--emit", "--codegen"];
    const SHORT: [&str; 3] = ["-o", "-L", "-C"];
    for name in LONG {
        if arg == name {
            return rest.next().map(|value| (canonical(name), value.clone()));
        }
        if let Some(value) = arg.strip_prefix(name).and_then(|tail| tail.strip_prefix('=')) {
            return Some((canonical(name), value.to_owned()));
        }
    }
    for name in SHORT {
        if arg == name {
            return rest.next().map(|value| (name, value.clone()));
        }
        if let Some(value) = arg.strip_prefix(name) {
            return Some((name, value.to_owned()));
        }
    }
    None
}

/// Answers the spelling an option is rewritten under.
fn canonical(name: &'static str) -> &'static str {
    match name {
        "--codegen" => "-C",
        name => name,
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
