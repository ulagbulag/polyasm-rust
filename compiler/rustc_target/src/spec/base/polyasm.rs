//! Target options shared by the three PolyASM triples.
//!
//! `polyasm-unknown-unknown`, `polyasm32-unknown-unknown` and
//! `polyasm64-unknown-unknown` name one architecture at three pointer widths.
//! Everything apart from the width is the same for all three, so it is stated
//! once here and the width-bearing fields are stated by each target.

use crate::spec::{CfgAbi, Os, TargetOptions, base};

/// Builds the options every PolyASM triple carries.
pub(crate) fn options() -> TargetOptions {
    let mut options = base::wasm::options();
    options.os = Os::Unknown;
    options.cfg_abi = CfgAbi::Polyasm;
    options.has_thread_local = false;

    // rustc_codegen_cranelift links a `.poly` image itself. It is a final
    // loadable image, distinct from a platform dynamic library.
    options.dynamic_linking = false;
    options.only_cdylib = false;
    options.dll_prefix = "".into();
    options.dll_suffix = ".poly".into();
    options.exe_suffix = ".poly".into();
    options.default_codegen_backend = Some("cranelift".into());
    options
}
