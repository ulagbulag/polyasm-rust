use std::path::{Path, PathBuf};

use crate::ci::CiEnv;

const DISABLED: &str = "Git execution is disabled by the PolyASM source policy";

#[derive(Debug)]
pub struct GitConfig<'a> {
    pub nightly_branch: &'a str,
    pub git_merge_commit_email: &'a str,
}

/// Represents the result of checking whether a set of paths
/// have been modified locally or not.
#[derive(PartialEq, Debug, Clone)]
pub enum PathFreshness {
    /// Artifacts should be downloaded from this upstream commit,
    /// there are no local modifications.
    LastModifiedUpstream { upstream: String },
    /// There are local modifications to a certain set of paths.
    /// "Local" essentially means "not-upstream" here.
    /// `upstream` is the latest upstream merge commit that made modifications to the
    /// set of paths.
    HasLocalModifications { upstream: String, modifications: Vec<PathBuf> },
    /// No upstream commit was found.
    /// This should not happen in most reasonable circumstances, but one never knows.
    MissingUpstream,
}

/// Returns an unavailable freshness result and leaves repository state untouched.
pub fn check_path_modifications(
    git_dir: &Path,
    config: &GitConfig<'_>,
    target_paths: &[&str],
    ci_env: CiEnv,
) -> Result<PathFreshness, String> {
    assert!(!target_paths.is_empty());
    for path in target_paths {
        assert!(Path::new(path.trim_start_matches(":!")).is_relative());
    }

    let _ = (git_dir, config, ci_env);
    Ok(PathFreshness::MissingUpstream)
}

/// Returns true if any of the passed `paths` have changed since the `base` commit.
pub fn changes_since(git_dir: &Path, base: &str, paths: &[&str]) -> Result<Vec<PathBuf>, String> {
    let _ = (git_dir, base, paths);
    Err(DISABLED.to_string())
}

/// Returns the most recent (ordered chronologically) commit found in the local history that
/// should exist upstream. We identify upstream commits by the e-mail of the commit
/// author.
///
/// If we are in CI, we simply return our first parent.
pub fn get_closest_upstream_commit(
    git_dir: Option<&Path>,
    config: &GitConfig<'_>,
    env: CiEnv,
) -> Result<Option<String>, String> {
    let _ = (git_dir, config, env);
    Ok(None)
}

/// Returns the files that have been modified in the current branch compared to the master branch.
/// This includes committed changes, uncommitted changes, and changes that are not even staged.
///
/// The `extensions` parameter can be used to filter the files by their extension.
/// Does not include removed files.
/// If `extensions` is empty, all files will be returned.
pub fn get_git_modified_files(
    config: &GitConfig<'_>,
    git_dir: Option<&Path>,
    extensions: &[&str],
) -> Result<Vec<String>, String> {
    let _ = (config, git_dir, extensions);
    Err(DISABLED.to_string())
}

/// Returns the files that haven't been added to git yet.
pub fn get_git_untracked_files(git_dir: Option<&Path>) -> Result<Option<Vec<String>>, String> {
    let _ = git_dir;
    Ok(None)
}
