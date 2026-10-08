//! Relocatable Cranelift module backend.

mod atomic;
mod backend;
mod instruction;
mod packet_range;
mod state;

pub(super) use state::InterchangeModule;
