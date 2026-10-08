//! PolyASM bytecode target with a 32-bit pointer fixed by the triple.
//!
//! PolyASM leaves byte order to each memory operation. Order is what it asks, and
//! the registry answers it in the instruction: loads and stores are spelled
//! little and big, and each atomic carries its own spelling at every width. A
//! target that settled on one order would leave every record spelled the other
//! way meaning something else.
//!
//! This triple states 32 as an equality: the image declares that its addresses
//! are 32 bits wide, and a loader runs it on a 32-bit address window. The
//! 32-bit port of a `polyasm-unknown-unknown` image is the image this triple
//! compiles from the same crate.

use crate::spec::{Arch, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    Target {
        // Compiler components which require an LLVM-compatible data model use
        // this triple. PolyASM code generation itself is handled by Cranelift.
        llvm_target: "wasm32-unknown-unknown".into(),
        metadata: TargetMetadata {
            description: Some("PolyASM bytecode, 32-bit pointers".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20".into(),
        arch: Arch::Polyasm,
        options: base::polyasm::options(),
    }
}
