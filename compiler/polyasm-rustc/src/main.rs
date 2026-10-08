//! The PolyASM compiler: the Rust compiler with the PolyASM extension of
//! `polyasm-rust` on every compilation.

// We need this feature as it changes `dylib` linking behavior and allows us to link to `rustc_driver`.
#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_session;

use std::process::ExitCode;

use polyasm_rust::PolyasmRust;
use rustc_driver::{
    DEFAULT_BUG_REPORT_URL, args, catch_with_exit_code, compiler_entrypoint, init_rustc_env_logger,
    install_ctrlc_handler, install_ice_hook,
};
use rustc_session::{EarlyDiagCtxt, config::ErrorOutputType};

// The C allocator of the compiler, as in `rustc-main`.
rustc_driver::override_c_allocator_in_binary!();

fn main() -> ExitCode {
    let early_dcx = EarlyDiagCtxt::new(ErrorOutputType::default());

    init_rustc_env_logger(&early_dcx);
    install_ice_hook(DEFAULT_BUG_REPORT_URL, |_| ());
    install_ctrlc_handler();

    catch_with_exit_code(|| compiler_entrypoint(&args::raw_args(&early_dcx), &mut PolyasmRust))
}
