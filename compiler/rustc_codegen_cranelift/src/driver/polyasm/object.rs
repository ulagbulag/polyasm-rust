//! The compiler's relocatable PolyASM object, `POBJ`.
//!
//! The object model and its codec are `polyasm::object`'s, the one statement
//! of the format this compiler writes and `polytime-frontend-cranelift`
//! reads. This module names the parts the compiler fills in.

pub(crate) use polyasm::object::{MAGIC, Object, OffloadRoot};
