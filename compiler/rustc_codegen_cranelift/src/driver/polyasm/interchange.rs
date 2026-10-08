//! Cranelift-backed interchange executable image.
mod compile;
mod image;
mod model;

pub(crate) use compile::compile;
pub(crate) use model::{
    CallableKind, CompiledImages, FunctionSignature, PortableImage, callable_kind,
};
