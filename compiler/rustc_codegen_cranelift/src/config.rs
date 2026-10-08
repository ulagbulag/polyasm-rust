use std::sync::OnceLock;

use clap::Parser;
use polyasm_format::options::BuildOptions;

/// Configuration of cg_clif as passed in through `-Cllvm-args`.
#[derive(Debug)]
pub struct BackendConfig {
    /// Should the crate be AOT compiled or JIT executed.
    ///
    /// Defaults to AOT compilation. Can be set using `-Cllvm-args=jit-mode`.
    pub jit_mode: bool,

    /// When JIT mode is enable pass these arguments to the program.
    pub jit_args: Vec<String>,
}

static BUILD_OPTIONS: OnceLock<BuildOptions> = OnceLock::new();

pub fn build_options() -> &'static BuildOptions {
    BUILD_OPTIONS.get_or_init(|| BuildOptions::parse_from(["rustc_codegen_cranelift"]))
}

impl BackendConfig {
    /// Parse the configuration passed in using `-Cllvm-args`.
    pub fn from_opts(opts: &[String]) -> Result<Self, String> {
        let mut config =
            BackendConfig { jit_mode: false, jit_args: build_options().jit_args.clone() };

        for opt in opts {
            if opt.starts_with("-import-instr-limit") {
                // Silently ignore -import-instr-limit. It is set by rust's build system even when
                // testing cg_clif.
                continue;
            }
            match &**opt {
                "jit-mode" => config.jit_mode = true,
                _ => return Err(format!("Unknown option `{}`", opt)),
            }
        }

        Ok(config)
    }
}
