//! Build configuration for Rust's release channels.
//!
//! Implements the stable/beta/nightly channel distinctions by setting various
//! flags like the `unstable_features`, calculating variables like `release` and
//! `package_vers`, and otherwise indicating to the compiler what it should
//! print out as part of its version information.

use std::fs;
use std::path::Path;

use super::exec::ExecutionContext;
use crate::core::session::Session;
use crate::utils::helpers::t;

#[derive(Clone, Default)]
pub enum GitInfo {
    /// This is not a git repository.
    #[default]
    Absent,
    /// This is not a git repository, but the info can be fetched from the
    /// `git-commit-info` file.
    RecordedForTarball(Info),
}

#[derive(Clone)]
pub struct Info {
    pub commit_date: String,
    pub sha: String,
    pub short_sha: String,
}

impl GitInfo {
    pub fn new(_omit_git_id: bool, dir: &Path, _exec_ctx: impl AsRef<ExecutionContext>) -> GitInfo {
        // PolyASM runs Git-free, read-only version discovery included.
        // A packaged source provides immutable recorded metadata; a live
        // checkout is otherwise treated as an unmanaged source tree.
        match read_commit_info_file(dir) {
            Some(info) => GitInfo::RecordedForTarball(info),
            None => GitInfo::Absent,
        }
    }

    pub fn info(&self) -> Option<&Info> {
        match self {
            GitInfo::Absent => None,
            GitInfo::RecordedForTarball(info) => Some(info),
        }
    }

    pub fn sha(&self) -> Option<&str> {
        self.info().map(|s| &s.sha[..])
    }

    pub fn sha_short(&self) -> Option<&str> {
        self.info().map(|s| &s.short_sha[..])
    }

    pub fn commit_date(&self) -> Option<&str> {
        self.info().map(|s| &s.commit_date[..])
    }

    pub fn version(&self, sess: &Session, num: &str) -> String {
        let mut version = sess.release(num);
        if let Some(inner) = self.info() {
            version.push_str(" (");
            version.push_str(&inner.short_sha);
            version.push(' ');
            version.push_str(&inner.commit_date);
            version.push(')');
        }
        version
    }

    /// Returns whether this directory has a `.git` directory which should be managed by bootstrap.
    pub fn is_managed_git_subrepository(&self) -> bool {
        match self {
            GitInfo::Absent | GitInfo::RecordedForTarball(_) => false,
        }
    }

    /// Returns whether this is being built from a tarball.
    pub fn is_from_tarball(&self) -> bool {
        match self {
            GitInfo::Absent => false,
            GitInfo::RecordedForTarball(_) => true,
        }
    }
}

/// Read the commit information from the `git-commit-info` file given the
/// project root.
pub fn read_commit_info_file(root: &Path) -> Option<Info> {
    if let Ok(contents) = fs::read_to_string(root.join("git-commit-info")) {
        let mut lines = contents.lines();
        let sha = lines.next();
        let short_sha = lines.next();
        let commit_date = lines.next();
        let info = match (commit_date, sha, short_sha) {
            (Some(commit_date), Some(sha), Some(short_sha)) => Info {
                commit_date: commit_date.to_owned(),
                sha: sha.to_owned(),
                short_sha: short_sha.to_owned(),
            },
            _ => panic!("the `git-commit-info` file is malformed"),
        };
        Some(info)
    } else {
        None
    }
}

/// Write the commit information to the `git-commit-info` file given the project
/// root.
pub fn write_commit_info_file(root: &Path, info: &Info) {
    let commit_info = format!("{}\n{}\n{}\n", info.sha, info.short_sha, info.commit_date);
    t!(fs::write(root.join("git-commit-info"), commit_info));
}

/// Write the commit hash to the `git-commit-hash` file given the project root.
pub fn write_commit_hash_file(root: &Path, sha: &str) {
    t!(fs::write(root.join("git-commit-hash"), sha));
}
