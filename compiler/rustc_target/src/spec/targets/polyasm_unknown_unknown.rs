//! Portable PolyASM bytecode target whose pointer width every triple leaves open.
//!
//! PolyASM leaves byte order to each memory operation, and the registry
//! answers it in the instruction: loads and stores are spelled little and big,
//! and each atomic carries its own spelling at every width. A target that
//! settled on one order would leave every record spelled the other way meaning
//! something else.
//!
//! # Which of the two meanings this triple carries
//!
//! A triple that leaves the width open means one of two things: an image whose width the
//! image itself states and the toolchain leaves alone, or an image that
//! silently takes the width of whatever host compiled it. This target is the
//! first. A `.poly` is a deployable image and a host is an accident of where
//! the compiler ran; letting the compiling machine decide the width of a
//! shipped image would make the same source produce a different image rule on
//! a different builder, which is exactly what an image format exists to
//! prevent.
//!
//! # How one image carries both widths
//!
//! rustc compiles every body of a session against exactly one pointer width:
//! a layout, a `size_of`, an evaluated static and a relocation field all fix
//! it. So a width-unfixed crate compiles once per width. This tuple's own
//! session lowers the crate at 32 bits, the narrowest address form the ISA
//! defines, and a companion session lowers the same crate under
//! `polyasm64-unknown-unknown` (`rustc_session::polyasm`). The final `.poly`
//! carries both compilations whole, one port section per width
//! (`polyasm_format::port`).
//!
//! A loader ports the image to the width of the machine instance it lands on
//! and then reads a fixed-width image: the 32-bit port is the image a
//! `polyasm32-unknown-unknown` build answers and the 64-bit port is the image
//! a `polyasm64-unknown-unknown` build answers, so a 64-bit instance
//! addresses memory beyond four gigabytes whenever the program asks for it,
//! and a 32-bit instance serves every program whose memory fits its window.
//!
//! `polytime_core::Capability::POLYASM` states 64 because it describes the
//! machine a width-unfixed image lands on by default, and that machine takes
//! the 64-bit port.

use crate::spec::{Arch, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    Target {
        // Compiler components which require an LLVM-compatible data model use
        // this triple. PolyASM code generation itself is handled by Cranelift.
        llvm_target: "wasm32-unknown-unknown".into(),
        metadata: TargetMetadata {
            description: Some("PolyASM portable bytecode".into()),
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
