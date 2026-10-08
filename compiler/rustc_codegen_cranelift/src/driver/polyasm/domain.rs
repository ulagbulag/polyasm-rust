//! Normalized MIR to the canonical PolyASM semantic graph.
//!
//! The interchange image is the mandatory portable fallback, while this graph
//! is the compiler authority used to admit and lower functions to eBPF,
//! Verilog, and the other semantic backends. Both representations originate
//! from the same set of monomorphized MIR instances and are joined by stable
//! symbol identity. The graph's model is `polyasm::object::domain`'s; this
//! module reads MIR into it.

mod capability;
mod function;
mod model;
mod module_lowering;
mod source;

pub(crate) use model::Fragment;
pub(crate) use module_lowering::collect;
use polyasm::object::domain::signature;
