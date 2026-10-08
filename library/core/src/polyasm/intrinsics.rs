//! The PolyASM instructions a guest reaches beyond the Rust language, and
//! the helpers the math of `wasm-direct-core` reaches.
//!
//! The core instruction set of PolyASM is the instruction set of WebAssembly
//! over virtual registers, and the expressions of Rust spell almost every
//! instruction of it. The functions here spell the size and the growth of
//! linear memory, as the calls of the declarations `polyasm.memory_size` and
//! `polyasm.memory_grow` the LLVM frontend of `polytime` reads, and the bulk
//! copy and fill of linear memory. Every extension instruction set is a set
//! of `extern "polyasm" fn` items under [`super::extension`].

use crate::intrinsics;

/// Writes `MemorySize`: answers the pages linear memory spans.
#[inline]
#[stable(feature = "polyasm", since = "CURRENT_RUSTC_VERSION")]
pub fn emit_memory_size() -> usize {
    unsafe extern "C" {
        #[link_name = "polyasm.memory_size"]
        safe fn memory_size() -> usize;
    }
    memory_size()
}

/// Writes `MemoryGrow`: grows linear memory by `pages` and answers the pages
/// it spanned before, or `usize::MAX` where it keeps its size.
#[inline]
#[stable(feature = "polyasm", since = "CURRENT_RUSTC_VERSION")]
pub fn emit_memory_grow(pages: usize) -> usize {
    unsafe extern "C" {
        #[link_name = "polyasm.memory_grow"]
        safe fn memory_grow(pages: usize) -> usize;
    }
    memory_grow(pages)
}

/// Writes `MemoryCopy`: copies `length` bytes from `from` to `to`, the two
/// ranges free to overlap.
///
/// # Safety
///
/// `from` names `length` bytes the caller reads and `to` names `length` bytes
/// the caller writes.
#[inline]
#[stable(feature = "polyasm", since = "CURRENT_RUSTC_VERSION")]
pub unsafe fn emit_memory_copy(to: *mut u8, from: *const u8, length: usize) {
    // SAFETY: forwarded from the caller.
    unsafe { intrinsics::copy(from, to, length) }
}

/// Writes `MemoryFill`: writes `value` into `length` bytes from `to`.
///
/// # Safety
///
/// `to` names `length` bytes the caller writes.
#[inline]
#[stable(feature = "polyasm", since = "CURRENT_RUSTC_VERSION")]
pub unsafe fn emit_memory_fill(to: *mut u8, value: u8, length: usize) {
    // SAFETY: forwarded from the caller.
    unsafe { intrinsics::write_bytes(to, value, length) }
}

/// Views a pointer into linear memory as a pointer to `T`, whose bytes
/// PolyASM keeps in the little-endian order of WebAssembly.
///
/// # Safety
///
/// `pointer` is aligned for `T`, and every byte the caller reads through the
/// answer holds a little-endian value of `T`.
#[inline]
#[must_use]
#[stable(feature = "polyasm", since = "CURRENT_RUSTC_VERSION")]
pub unsafe fn little_endian_pointer_mut<S, T>(pointer: *mut S) -> *mut T {
    pointer.cast()
}

/// Declares the elementary helpers of the math of `wasm-direct-core`, each
/// one operation over one value.
macro_rules! elementary {
    ($($(#[$doc:meta])* $name:ident($ty:ident) = $body:path;)+) => {$(
        $(#[$doc])*
        #[inline]
        #[must_use]
        #[stable(feature = "polyasm", since = "CURRENT_RUSTC_VERSION")]
        pub fn $name(value: $ty) -> $ty {
            $body(value)
        }
    )+};
}

elementary! {
    /// The binary32 square root, the `Sqrt` operation of `Unary`.
    fsqrt32(f32) = intrinsics::sqrt;
    /// The binary64 square root, the `Sqrt` operation of `Unary`.
    fsqrt64(f64) = intrinsics::sqrt;
}
