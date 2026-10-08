//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![feature(extern_types)]
#![feature(file_buffered)]
#![feature(impl_trait_in_assoc_type)]
#![feature(iter_intersperse)]
#![feature(macro_derive)]
#![feature(once_cell_try)]
#![feature(try_blocks)]
// tidy-alphabetical-end

pub use crate::back::llvm_backend::{LlvmCodegenBackend, ModuleLlvm};

mod abi;
mod allocator;
pub mod asm;
mod attributes;
mod back;
mod base;
pub mod builder;
mod callee;
mod common;
mod consts;
pub mod context;
mod coverageinfo;
mod debuginfo;
mod declare;
mod diagnostics;
mod intrinsic;
pub mod llvm;
mod llvm_util;
mod macros;
mod mono_item;
mod type_;
pub mod type_of;
mod typetree;
mod va_arg;
mod value;

pub mod extensions;
