//! PolyASM bytecode target with a 64-bit pointer fixed by the triple.
//!
//! PolyASM leaves byte order to each memory operation. Order is what it asks, and
//! the registry answers it in the instruction: loads and stores are spelled
//! little and big, and each atomic carries its own spelling at every width. A
//! target that settled on one order would leave every record spelled the other
//! way meaning something else.
//!
//! This triple states 64 as an equality. An image built here addresses beyond
//! four gigabytes on a 64-bit machine instance. The companion session of a
//! `polyasm-unknown-unknown` crate compiles under this triple, so the 64-bit
//! port of a width-unfixed image is the image this triple compiles from the
//! same crate.

use crate::spec::{Arch, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    Target {
        // Compiler components which require an LLVM-compatible data model use
        // this triple. PolyASM code generation itself is handled by Cranelift.
        llvm_target: "wasm64-unknown-unknown".into(),
        metadata: TargetMetadata {
            description: Some("PolyASM bytecode, 64-bit pointers".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20".into(),
        arch: Arch::Polyasm,
        options: base::polyasm::options(),
    }
}
