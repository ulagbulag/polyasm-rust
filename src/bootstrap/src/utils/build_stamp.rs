//! Module for managing build stamp files.
//!
//! Contains the core implementation of how bootstrap utilizes stamp files on build processes.

use std::path::{Path, PathBuf};
use std::{fs, io};

use sha2::digest::Digest;

use crate::core::backend::CodegenBackendKind;
use crate::core::builder::Builder;
use crate::core::compiler::Compiler;
use crate::core::config::TargetSelection;
use crate::core::session::Mode;
use crate::utils::helpers::{hex_encode, mtime, t};

#[cfg(test)]
mod tests;

/// Manages a stamp file to track build state. The file is created in the given
/// directory and can have custom content and name.
#[derive(Clone)]
pub struct BuildStamp {
    path: PathBuf,
    stamp: String,
}

impl BuildStamp {
    /// Creates a new `BuildStamp` for a given directory.
    ///
    /// By default, stamp will be an empty file named `.stamp` within the specified directory.
    pub fn new(dir: &Path) -> Self {
        // Avoid using `is_dir()` as the directory may not exist yet.
        // It is more appropriate to assert that the path is not a file.
        assert!(!dir.is_file(), "can't be a file path");
        Self { path: dir.join(".stamp"), stamp: String::new() }
    }

    /// Returns path of the stamp file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the value of the stamp.
    ///
    /// Note that this is empty by default and is populated using `BuildStamp::add_stamp`.
    /// It is not read from an actual file, but rather it holds the value that will be used
    /// when `BuildStamp::write` is called.
    pub fn stamp(&self) -> &str {
        &self.stamp
    }

    /// Adds specified stamp content to the current value.
    ///
    /// This method can be used incrementally e.g., `add_stamp("x").add_stamp("y").add_stamp("z")`.
    pub fn add_stamp<S: ToString>(mut self, stamp: S) -> Self {
        self.stamp.push_str(&stamp.to_string());
        self
    }

    /// Adds a prefix to stamp's name.
    ///
    /// Prefix cannot start or end with a dot (`.`).
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        assert!(
            !prefix.starts_with('.') && !prefix.ends_with('.'),
            "prefix can not start or end with '.'"
        );

        let stamp_filename = self.path.file_name().unwrap().to_str().unwrap();
        let stamp_filename = stamp_filename.strip_prefix('.').unwrap_or(stamp_filename);
        self.path.set_file_name(format!(".{prefix}-{stamp_filename}"));

        self
    }

    /// Removes the stamp file if it exists.
    pub fn remove(&self) -> io::Result<()> {
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(e) => {
                if e.kind() == io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Creates the stamp file.
    pub fn write(&self) -> io::Result<()> {
        fs::write(&self.path, &self.stamp)
    }

    /// Checks if the stamp file is up-to-date.
    ///
    /// It is considered up-to-date if file content matches with the stamp string.
    pub fn is_up_to_date(&self) -> bool {
        match fs::read(&self.path) {
            Ok(h) => self.stamp.as_bytes() == h.as_slice(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => false,
            Err(e) => {
                panic!("failed to read stamp file `{}`: {}", self.path.display(), e);
            }
        }
    }
}

/// Clear out `dir` if `input` is newer.
///
/// After this executes, it will also ensure that `dir` exists.
pub fn clear_if_dirty(builder: &Builder<'_>, dir: &Path, input: &Path) -> bool {
    let stamp = BuildStamp::new(dir);
    let mut cleared = false;
    if mtime(stamp.path()) < mtime(input) {
        builder.do_if_verbose(|| {
            println!(
                "Removing dirty directory `{}` because `{}` changed",
                dir.display(),
                input.display(),
            )
        });
        let _ = fs::remove_dir_all(dir);
        cleared = true;
    } else if stamp.path().exists() {
        return cleared;
    }
    t!(fs::create_dir_all(dir));
    t!(fs::File::create(stamp.path()));
    cleared
}

/// Cargo's output path for librustc_codegen_llvm in a given stage, compiled by a particular
/// compiler for the specified target and backend.
pub fn codegen_backend_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: TargetSelection,
    backend: &CodegenBackendKind,
) -> BuildStamp {
    BuildStamp::new(&builder.cargo_out(compiler, Mode::Codegen, target))
        .with_prefix(&format!("lib{}", backend.crate_name()))
}

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular `build_compiler` for the specified `target`.
pub fn libstd_stamp(
    builder: &Builder<'_>,
    build_compiler: Compiler,
    target: TargetSelection,
) -> BuildStamp {
    BuildStamp::new(&builder.cargo_out(build_compiler, Mode::Std, target)).with_prefix("libstd")
}

/// Cargo's output path for librustc in a given stage, compiled by a particular
/// `build_compiler` for the specified target.
pub fn librustc_stamp(
    builder: &Builder<'_>,
    build_compiler: Compiler,
    target: TargetSelection,
) -> BuildStamp {
    BuildStamp::new(&builder.cargo_out(build_compiler, Mode::Rustc, target)).with_prefix("librustc")
}

/// Computes a SHA-256 stamp of the state of a repository/submodule and additional input.
///
/// The state is the path, length and modification time of every file under `dir`, read from
/// the filesystem in sorted order with `target` and `.git` directories skipped; the additional
/// input joins it in the same SHA-256 computation.
///
/// # Parameters
///
/// - `dir`: A reference to the directory path of the target repository/submodule.
/// - `additional_input`: An additional input to be included in the stamp.
pub fn generate_smart_stamp_hash(
    builder: &Builder<'_>,
    dir: &Path,
    additional_input: &str,
) -> String {
    // The PolyASM source policy runs Git-free, so the working tree's state
    // comes from the filesystem. Walking `dir` for each file's path, length and
    // modification time answers the same question the diff and status answered
    // -- has anything under here changed since the artefact was stamped -- from
    // the files alone.
    let _ = builder;
    let mut state = String::new();
    let mut pending = ::std::collections::VecDeque::from([dir.to_path_buf()]);
    while let Some(directory) = pending.pop_front() {
        let Ok(entries) = ::std::fs::read_dir(&directory) else { continue };
        let mut sorted: Vec<_> = entries.flatten().map(|entry| entry.path()).collect();
        sorted.sort();
        for path in sorted {
            let Ok(metadata) = ::std::fs::symlink_metadata(&path) else { continue };
            if metadata.is_dir() {
                if path.file_name().is_some_and(|name| name == "target" || name == ".git") {
                    continue;
                }
                pending.push_back(path);
                continue;
            }
            let modified = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(::std::time::UNIX_EPOCH).ok())
                .map_or(0, |since| since.as_nanos());
            state.push_str(&format!("{}\0{}\0{}\n", path.display(), metadata.len(), modified));
        }
    }
    let (diff, status) = (String::new(), state);

    let mut hasher = sha2::Sha256::new();

    hasher.update(diff);
    hasher.update(status);
    hasher.update(additional_input);

    hex_encode(hasher.finalize().as_slice())
}
