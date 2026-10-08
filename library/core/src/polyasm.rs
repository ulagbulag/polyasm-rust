//! Explicit byte-order designation for the PolyASM bytecode target.
//!
//! `polyasm-unknown-unknown` leaves the native byte order open, so reinterpreting
//! a wide value as its bytes takes an order the source names. The
//! `.poly` container stores wide values in the same little-endian encoding as
//! `wasm32-unknown-unknown`, so the views below are exact and copy-free.
//!
//! These functions are the designation itself: calling one states that the
//! bytes are read and written in little-endian order.

use crate::{mem, slice};

/// Every PolyASM instruction a guest reaches, as the compiler intrinsic it is.
#[stable(feature = "polyasm_instructions", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_intrinsics"]
pub mod intrinsics;

/// PolyASM's own fixed-width byte vectors, declared in [`intrinsics`].
#[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
pub use intrinsics::vector;

/// Packet capabilities and the operand types of the packet instructions.
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub mod packet;

#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub use packet::{PacketContext, PacketWindow};

/// Views every element of `values` as its little-endian bytes.
#[inline]
#[must_use]
#[stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
pub const fn little_endian_bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: `T` is `Copy`-shaped storage owned by the caller, the length is
    // its exact byte length, and `u8` has the loosest alignment of all.
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * size_of::<T>()) }
}

/// Views every element of `values` as its mutable little-endian bytes.
#[inline]
#[must_use]
#[stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
pub const fn little_endian_bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
    let len = values.len() * size_of::<T>();
    // SAFETY: as in `little_endian_bytes`, and the exclusive borrow is kept.
    unsafe { slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len) }
}

/// Reads `len` little-endian elements of `T` out of `bytes`.
///
/// # Safety
///
/// `bytes` must be aligned for `T` and hold at least `len * size_of::<T>()`
/// initialized bytes that were written as little-endian `T` values.
#[inline]
#[must_use]
#[stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
pub const unsafe fn little_endian_values<T>(bytes: &[u8], len: usize) -> &[T] {
    debug_assert!(bytes.len() >= len * size_of::<T>());
    // SAFETY: the caller guarantees the length, alignment, and provenance.
    unsafe { slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len) }
}

/// Reads one little-endian `T` out of the start of `bytes`.
///
/// # Safety
///
/// `bytes` must be aligned for `T` and hold at least `size_of::<T>()`
/// initialized bytes that were written as one little-endian `T` value.
#[inline]
#[must_use]
#[stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
pub const unsafe fn little_endian_value<T>(bytes: &[u8]) -> &T {
    debug_assert!(bytes.len() >= size_of::<T>());
    // SAFETY: the caller guarantees the length, alignment, and provenance.
    unsafe { &*bytes.as_ptr().cast::<T>() }
}

/// Reinterprets one little-endian raw pointer in place, every byte where it stands.
///
/// # Safety
///
/// `pointer` must be aligned for `T` and, for as many bytes as the caller
/// reads, point to values written in little-endian order.
#[inline]
#[must_use]
#[stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
pub const unsafe fn little_endian_pointer<S, T>(pointer: *const S) -> *const T {
    let _ = mem::align_of::<T>();
    pointer.cast::<T>()
}

/// Reinterprets one mutable little-endian raw pointer.
///
/// # Safety
///
/// As in [`little_endian_pointer`], and the caller must own the write.
#[inline]
#[must_use]
#[stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
pub const unsafe fn little_endian_pointer_mut<S, T>(pointer: *mut S) -> *mut T {
    let _ = mem::align_of::<T>();
    pointer.cast::<T>()
}

/// Reinterprets one value as another of the same size in little-endian order.
///
/// # Safety
///
/// `S` and `T` must have the same size, and every bit pattern of `S` must be a
/// valid `T` once its bytes are read in little-endian order.
#[inline]
#[must_use]
#[stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
pub const unsafe fn little_endian_transmute<S, T>(value: S) -> T {
    // SAFETY: the caller guarantees the size and validity of the conversion.
    unsafe { mem::transmute_copy(&mem::ManuallyDrop::new(value)) }
}

/// Reinterprets one big-endian value as another of the same size.
///
/// # Safety
///
/// `S` and `T` must have the same size, and every bit pattern of `S` must be a
/// valid `T` once its bytes are read in big-endian order.
#[inline]
#[must_use]
#[stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_stable(feature = "polyasm_explicit_endian", since = "CURRENT_RUSTC_VERSION")]
pub const unsafe fn big_endian_transmute<S, T>(value: S) -> T {
    // SAFETY: the caller guarantees the size and validity of the conversion.
    unsafe { mem::transmute_copy(&mem::ManuallyDrop::new(value)) }
}
