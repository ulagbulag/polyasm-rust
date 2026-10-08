//! Every PolyASM instruction a guest reaches, as the compiler intrinsic it is.
//!
//! PolyASM is a CISC for multi-node, heterogeneous, single-thread programs.
//! Each declaration in this file is one row of the PolyASM registry
//! (`polyasm_format::executable::Opcode`) and one method of polytime's
//! `Emitter` trait (`emit_<Row>`): the compiler lowers a call of it to that
//! row, and polytime answers the row on each machine with that machine's own
//! implementation or with the trait's default. This file is the one place a
//! guest-visible PolyASM instruction is declared, so the rows a guest spells
//! and the methods polytime answers stay one list.
//!
//! The name of each intrinsic is the snake-case spelling of its row
//! (`gemm` is `Gemm`, `f32_from_bf16` is `F32FromBf16`), so the lowering in
//! both codegen backends and both PolyASM frontends reads the row off the
//! intrinsic's own name with zero per-row tables: the backends call
//! `__polyasm_<name>`, and `polyasm_format::library::instruction` answers the
//! row for that symbol.
//!
//! The macros below state each family once: operator rows over an in-memory
//! descriptor, elementary rows over registers, counted runs of bytes, vector
//! rows and packet rows. Adding an instruction is one line in its family's
//! table, one registry row and one `Emitter` method with its default; an
//! operator row's `Emitter` method, its default, its machine writers and its
//! host entry all expand `polyasm_format::operator_table`, so an operator
//! instruction is one line there beside the line here.
//!
//! # Operator rows and their portable bodies
//!
//! An operator row reads one descriptor (`polyasm_format::operator`): a head
//! word, operand records naming address, encoding, extents and strides, and
//! the row's attribute words. Each operator intrinsic is generic over a
//! [`Portable`] answer, and its body is that answer: the instruction's
//! fallback. The compiler emits the row with the fallback instance as the
//! row's record, so the image always carries the portable body of every row
//! it spells, bound by the intrinsic itself. A machine whose `Emitter`
//! overrides the row answers the descriptor with its own engines (the NPU
//! tensor, vector and DMA engines, BLAS-style CPU kernels, the interpreter's
//! native answer); every other machine inherits the trait default, which
//! enters the record.
//!
//! Every word of a descriptor is a little-endian `u64`, so the bytes a guest
//! writes are identical on `polyasm`, `polyasm32` and `polyasm64`; the
//! register that carries the descriptor's address is pointer-wide.

/// The portable answer of one operator instruction: the body a machine runs
/// where it carries the row elsewhere, over the descriptor at `descriptor`.
///
/// The answer is the row's status word: zero where the operator completed.
#[stable(feature = "polyasm_instructions", since = "CURRENT_RUSTC_VERSION")]
pub trait Portable {
    /// Answers the descriptor at `descriptor` with the portable body.
    ///
    /// # Safety
    ///
    /// `descriptor` points at one whole descriptor, and every operand record
    /// in it names memory the caller owns for the direction its role names.
    #[stable(feature = "polyasm_instructions", since = "CURRENT_RUSTC_VERSION")]
    unsafe fn portable(descriptor: *const u64) -> u64;
}

/// Declares the operator instructions: one generic intrinsic per row, whose
/// body is the [`Portable`] answer it is instantiated with.
macro_rules! operators {
    ($($(#[$doc:meta])* $name:ident => $row:literal;)+) => {$(
        $(#[$doc])*
        #[doc = ""]
        #[doc = concat!("The `", $row, "` row; polytime answers it with `Emitter::emit_", $row,
            "`. The answer is the row's status word, zero where the operator completed.")]
        #[doc = ""]
        #[doc = "# Safety"]
        #[doc = ""]
        #[doc = "As in [`Portable::portable`]."]
        #[rustc_intrinsic]
        #[rustc_nounwind]
        #[stable(feature = "polyasm_instructions", since = "CURRENT_RUSTC_VERSION")]
        pub unsafe fn $name<P: Portable>(descriptor: *const u64) -> u64 {
            // SAFETY: forwarded from the caller.
            unsafe { P::portable(descriptor) }
        }
    )+};
}

operators! {
    /// `Y = α·(A·B) + β·Y + bias` over batched matrices.
    gemm => "Gemm";
    /// Rows routed to weight groups, combined under their route weights.
    gemm_grouped => "GemmGrouped";
    /// Grouped convolution over one to three spatial axes.
    convolve => "Convolve";
    /// Max or average pooling over one to three spatial axes.
    pool => "Pool";
    /// One unary operation over every element.
    map_unary => "MapUnary";
    /// One binary operation over every pair, broadcasting by strides.
    map_binary => "MapBinary";
    /// `Y = C ? A : B` over every element.
    select => "Select";
    /// `Y = act(clamp(G)) · (clamp(U) + offset)`.
    gated => "Gated";
    /// RMS, layer or group normalisation over the innermost axis.
    normalize => "Normalize";
    /// Softmax over the innermost axis, with mask, sink and soft-cap.
    softmax => "Softmax";
    /// One fold over the axes a bitmask names.
    reduce => "Reduce";
    /// The `k` largest or smallest values of the innermost axis.
    top_k => "TopK";
    /// A running fold or a linear recurrence along one axis.
    scan => "Scan";
    /// Rotary pairs from cosine and sine tables.
    rotate => "Rotate";
    /// Table elements selected by an index operand.
    gather => "Gather";
    /// Source elements written into a table at an index operand.
    scatter => "Scatter";
    /// Every element carried into another encoding or layout.
    convert => "Convert";
    /// A constant or an arithmetic sequence.
    fill => "Fill";
    /// Scaled dot-product attention over a key and value cache.
    attend => "Attend";
    /// One ternary operation over every triple, broadcasting by strides:
    /// fused multiply-add, `addcmul`, `addcdiv`, `lerp`, `clamp`.
    map_ternary => "MapTernary";
    /// Elements narrowed into a quantised encoding under block scales and
    /// zero points, read or computed per block.
    quantize => "Quantize";
    /// The elements a mask selects, packed in row-major order, and their
    /// count: `masked_select`, `nonzero`.
    compact => "Compact";
}

/// Declares the elementary instructions: one intrinsic per register row.
macro_rules! elementary {
    ($($(#[$doc:meta])* $name:ident($($argument:ident: $from:ty),+) -> $into:ty => $row:literal;)+) => {$(
        $(#[$doc])*
        #[doc = ""]
        #[doc = concat!("The `", $row, "` row; polytime answers it with `Emitter::emit_", $row,
            "`, and every lane computes the bits of `polytime_core::elementary`.")]
        #[cfg(target_abi = "polyasm")]
        #[rustc_intrinsic]
        #[rustc_nounwind]
        #[must_use]
        #[stable(feature = "polyasm_instructions", since = "CURRENT_RUSTC_VERSION")]
        pub fn $name($($argument: $from),+) -> $into;
    )+};
}

elementary! {
    /// The binary32 exponential.
    fexp32(value: f32) -> f32 => "Fexp32";
    /// The binary64 exponential.
    fexp64(value: f64) -> f64 => "Fexp64";
    /// The binary32 natural logarithm.
    flog32(value: f32) -> f32 => "Flog32";
    /// The binary64 natural logarithm.
    flog64(value: f64) -> f64 => "Flog64";
    /// The binary32 sine.
    fsin32(value: f32) -> f32 => "Fsin32";
    /// The binary64 sine.
    fsin64(value: f64) -> f64 => "Fsin64";
    /// The binary32 cosine.
    fcos32(value: f32) -> f32 => "Fcos32";
    /// The binary64 cosine.
    fcos64(value: f64) -> f64 => "Fcos64";
    /// The binary32 hyperbolic tangent.
    ftanh32(value: f32) -> f32 => "Ftanh32";
    /// The binary64 hyperbolic tangent.
    ftanh64(value: f64) -> f64 => "Ftanh64";
    /// The binary32 error function.
    ferf32(value: f32) -> f32 => "Ferf32";
    /// The binary64 error function.
    ferf64(value: f64) -> f64 => "Ferf64";
    /// Rounds a binary32 value half away from zero.
    fround32(value: f32) -> f32 => "Fround32";
    /// Rounds a binary64 value half away from zero.
    fround64(value: f64) -> f64 => "Fround64";
    /// Raises a binary32 base to a binary32 exponent.
    fpow32(base: f32, exponent: f32) -> f32 => "Fpow32";
    /// Raises a binary64 base to a binary64 exponent.
    fpow64(base: f64, exponent: f64) -> f64 => "Fpow64";
    /// Widens IEEE binary16 bits to binary32 exactly.
    f32_from_f16(bits: u16) -> f32 => "F32FromF16";
    /// Narrows binary32 to IEEE binary16 bits, ties to even.
    f16_from_f32(value: f32) -> u16 => "F16FromF32";
    /// Widens bfloat16 bits to binary32 exactly.
    f32_from_bf16(bits: u16) -> f32 => "F32FromBf16";
    /// Narrows binary32 to bfloat16 bits, ties to even.
    bf16_from_f32(value: f32) -> u16 => "Bf16FromF32";
    /// Widens OCP FP8 E4M3 bits to binary32 exactly.
    f32_from_f8_e4m3(bits: u8) -> f32 => "F32FromF8E4m3";
    /// Narrows binary32 to OCP FP8 E4M3 bits, ties to even.
    f8_e4m3_from_f32(value: f32) -> u8 => "F8E4m3FromF32";
    /// Widens OCP FP8 E5M2 bits to binary32 exactly.
    f32_from_f8_e5m2(bits: u8) -> f32 => "F32FromF8E5m2";
    /// Narrows binary32 to OCP FP8 E5M2 bits, ties to even.
    f8_e5m2_from_f32(value: f32) -> u8 => "F8E5m2FromF32";
    /// The binary32 square root.
    fsqrt32(value: f32) -> f32 => "Fsqrt32";
    /// The binary64 square root.
    fsqrt64(value: f64) -> f64 => "Fsqrt64";
    /// Rounds a binary32 value toward negative infinity.
    ffloor32(value: f32) -> f32 => "Ffloor32";
    /// Rounds a binary64 value toward negative infinity.
    ffloor64(value: f64) -> f64 => "Ffloor64";
    /// Rounds a binary32 value toward positive infinity.
    fceil32(value: f32) -> f32 => "Fceil32";
    /// Rounds a binary64 value toward positive infinity.
    fceil64(value: f64) -> f64 => "Fceil64";
    /// Rounds a binary32 value toward zero.
    ftrunc32(value: f32) -> f32 => "Ftrunc32";
    /// Rounds a binary64 value toward zero.
    ftrunc64(value: f64) -> f64 => "Ftrunc64";
    /// Rounds a binary32 value to the nearest integer, ties to even.
    fnearest32(value: f32) -> f32 => "Fnearest32";
    /// Rounds a binary64 value to the nearest integer, ties to even.
    fnearest64(value: f64) -> f64 => "Fnearest64";
}

/// Moves `bytes` bytes from `source` to `destination`; the two runs are free
/// to overlap. The `MemoryMove` row, answered by `Emitter::emit_MemoryMove`.
///
/// The body is the operation stated for constant evaluation; a PolyASM
/// lowering writes the row.
///
/// # Safety
///
/// As in [`crate::ptr::copy`], counted in bytes.
#[rustc_intrinsic]
#[rustc_nounwind]
#[rustc_const_stable(feature = "polyasm_instructions", since = "CURRENT_RUSTC_VERSION")]
#[stable(feature = "polyasm_instructions", since = "CURRENT_RUSTC_VERSION")]
pub const unsafe fn memory_move(destination: *mut u8, source: *const u8, bytes: usize) {
    // SAFETY: the caller owns both runs for the whole byte count, and the
    // copy reads the source in full before it writes any destination byte.
    unsafe { crate::intrinsics::copy(source, destination, bytes) }
}

/// Writes `fill` into each of `bytes` bytes at `destination`. The
/// `MemorySet` row, answered by `Emitter::emit_MemorySet`.
///
/// The body is the operation stated for constant evaluation; a PolyASM
/// lowering writes the row.
///
/// # Safety
///
/// As in [`crate::ptr::write_bytes`], counted in bytes.
#[rustc_intrinsic]
#[rustc_nounwind]
#[rustc_const_stable(feature = "polyasm_instructions", since = "CURRENT_RUSTC_VERSION")]
#[stable(feature = "polyasm_instructions", since = "CURRENT_RUSTC_VERSION")]
pub const unsafe fn memory_set(destination: *mut u8, fill: u8, bytes: usize) {
    // SAFETY: the caller owns the destination run for the whole byte count.
    unsafe { crate::intrinsics::write_bytes(destination, fill, bytes) }
}

/// Compares `bytes` bytes at `left` against `bytes` bytes at `right`. The
/// `MemoryCompare` row, answered by `Emitter::emit_MemoryCompare`.
///
/// Answers the sign alone: negative where the first run is below the second,
/// zero where they are equal, positive where the first is above. The
/// magnitude is whatever the comparing silicon leaves, so a caller reads the
/// sign.
///
/// # Safety
///
/// Both runs are readable for the whole of `bytes` bytes.
#[rustc_intrinsic]
#[rustc_nounwind]
#[must_use]
#[rustc_const_unstable(feature = "const_cmp", issue = "143800")]
#[stable(feature = "polyasm_instructions", since = "CURRENT_RUSTC_VERSION")]
pub const unsafe fn memory_compare(left: *const u8, right: *const u8, bytes: usize) -> i32 {
    // SAFETY: the caller guarantees both runs are readable for the count.
    unsafe { crate::intrinsics::compare_bytes(left, right, bytes) }
}

// ---------------------------------------------------------------------------
// Vector rows, over PolyASM's own fixed-width byte vectors.
// ---------------------------------------------------------------------------

/// PolyASM's own fixed-width byte vectors.
///
/// These stand apart from WebAssembly's `v128`: PolyASM publishes three
/// widths of its own, the executor holds them in a vector bank separate from
/// the integer file, and the AOT lowering hands each one to the host's vector
/// unit. A guest that compares sixteen, thirty-two or sixty-four bytes at a
/// time names the width it wants and gets exactly that.
#[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
pub mod vector {
    #[cfg(not(target_abi = "polyasm"))]
    use crate::intrinsics::simd::simd_shl;
    use crate::intrinsics::simd::{
        simd_add, simd_and, simd_bitmask, simd_eq, simd_extract, simd_ge, simd_gt, simd_insert,
        simd_le, simd_lt, simd_mul, simd_ne, simd_neg, simd_or, simd_reduce_all, simd_reduce_any,
        simd_saturating_add, simd_saturating_sub, simd_select, simd_shr, simd_shuffle, simd_sub,
        simd_xor,
    };
    #[cfg(not(target_abi = "polyasm"))]
    use crate::intrinsics::simd::{simd_extract_dyn, simd_insert_dyn};
    // PolyASM's own vector instructions: a shift by one scalar amount, an
    // unaligned whole-vector load and store, and the run-time byte shuffle.
    // The remaining vector operations are the portable `simd_*` intrinsics,
    // which every backend lowers to the same vector rows.
    /// Shifts every lane of a vector left by one scalar amount.
    ///
    /// `T` must be a vector of integers.
    ///
    /// Shifts `lhs` left by `bits`, shifting in zeros. Unlike [`simd_shl`] the
    /// amount is one scalar rather than a second vector, so a backend names a
    /// whole-vector shift directly, with zero splat analysis of an amount
    /// vector.
    ///
    /// # Safety
    ///
    /// `bits` must be in `0..<int>::BITS`.
    #[cfg(target_abi = "polyasm")]
    #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_intrinsic]
    #[rustc_nounwind]
    pub unsafe fn simd_shl_scalar<T>(lhs: T, bits: u32) -> T;

    /// Shifts every lane of a vector right by one scalar amount.
    ///
    /// `T` must be a vector of integers.
    ///
    /// Shifts `lhs` right by `bits`, shifting in sign bits for signed lane types
    /// and zeros for unsigned ones. Unlike [`simd_shr`] the amount is one scalar
    /// rather than a second vector, so a backend names a whole-vector shift
    /// directly, with zero splat analysis of an amount vector.
    ///
    /// # Safety
    ///
    /// `bits` must be in `0..<int>::BITS`.
    #[cfg(target_abi = "polyasm")]
    #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_intrinsic]
    #[rustc_nounwind]
    pub unsafe fn simd_shr_scalar<T>(lhs: T, bits: u32) -> T;

    #[doc = "Reads one whole vector out of a possibly unaligned address."]
    #[doc = ""]
    #[doc = "`U` must be a vector, and `pointer` must be readable for that vector's"]
    #[doc = "whole width. The address is any byte, with zero alignment assumed."]
    #[doc = ""]
    #[doc = "Unlike a byte copy into a temporary, the destination is a vector-typed"]
    #[doc = "value from the start, so a backend that owns a vector load names one"]
    #[doc = "in place of a width-sized untyped move."]
    #[doc = ""]
    #[doc = "# Safety"]
    #[doc = ""]
    #[doc = "`pointer` must be valid for reads of `size_of::<U>()` bytes."]
    #[cfg(target_abi = "polyasm")]
    #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_intrinsic]
    #[rustc_nounwind]
    pub unsafe fn simd_load_unaligned<T, U>(pointer: *const T) -> U;

    #[doc = "Writes one whole vector to a possibly unaligned address."]
    #[doc = ""]
    #[doc = "`U` must be a vector, and `pointer` must be writable for that vector's"]
    #[doc = "whole width. The address is any byte, with zero alignment assumed."]
    #[doc = ""]
    #[doc = "Unlike a byte copy out of a temporary, the source is a vector-typed"]
    #[doc = "value from the start, so a backend that owns a vector store names one"]
    #[doc = "in place of a width-sized untyped move."]
    #[doc = ""]
    #[doc = "# Safety"]
    #[doc = ""]
    #[doc = "`pointer` must be valid for writes of `size_of::<U>()` bytes."]
    #[cfg(target_abi = "polyasm")]
    #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_intrinsic]
    #[rustc_nounwind]
    pub unsafe fn simd_store_unaligned<T, U>(pointer: *mut T, value: U);

    /// Shuffles the 8-bit lanes of one vector by the run-time indices in another.
    ///
    /// `T` must be a vector of 8-bit integer lanes, and both arguments and the
    /// result share that one type.
    ///
    /// Element `i` of the result is `table[indices[i] & (LANES - 1)]` when bit
    /// seven of `indices[i]` is clear, and zero when it is set. For a vector wider
    /// than 128 bits every 128-bit lane is indexed on its own, which is what the
    /// hardware byte shuffles of every supported target answer.
    #[cfg(target_abi = "polyasm")]
    #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_intrinsic]
    #[rustc_nounwind]
    pub unsafe fn simd_swizzle_dyn<T>(table: T, indices: T) -> T;

    #[repr(simd)]
    struct ShuffleIndex<const LEN: usize>([u32; LEN]);

    const fn align_right<const LANES: usize>(shift: usize) -> [u32; LANES] {
        let mut index = [0u32; LANES];
        let mut lane = 0;
        while lane < LANES {
            index[lane] = (LANES - shift + lane) as u32;
            lane += 1;
        }
        index
    }

    const fn align_right_in_lane<const LANES: usize>(shift: usize) -> [u32; LANES] {
        let mut index = [0u32; LANES];
        let mut lane = 0;
        while lane < LANES {
            let base = lane / 16 * 16;
            let offset = 16 - shift + lane % 16;
            index[lane] = if offset < 16 {
                (base + offset) as u32
            } else {
                (LANES + base + offset - 16) as u32
            };
            lane += 1;
        }
        index
    }

    const fn interleave<const LANES: usize>(offset: usize) -> [u32; LANES] {
        let mut index = [0u32; LANES];
        let mut lane = 0;
        while lane < LANES {
            let base = lane / 16 * 16;
            let inner = lane % 16;
            let source = base + offset + inner / 2;
            index[lane] = if inner % 2 == 0 { source as u32 } else { (LANES + source) as u32 };
            lane += 1;
        }
        index
    }

    const fn swap_lanes<const LANES: usize>() -> [u32; LANES] {
        let mut index = [0u32; LANES];
        let mut lane = 0;
        while lane < LANES {
            let base = lane / 32 * 32;
            index[lane] = (base + (lane % 32 + 16) % 32) as u32;
            lane += 1;
        }
        index
    }

    const fn broadcast_lane<const LANES: usize>() -> [u32; LANES] {
        let mut index = [0u32; LANES];
        let mut lane = 0;
        while lane < LANES {
            index[lane] = (lane % 16) as u32;
            lane += 1;
        }
        index
    }

    macro_rules! vector_types {
        ($($vector:ident, $lane:ty, $lanes:literal;)+) => {$(
            #[doc = concat!("A PolyASM vector register holding ", stringify!($lanes),
                " lanes of `", stringify!($lane), "`.")]
            #[repr(simd)]
            #[derive(Copy, Clone)]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub struct $vector(pub [$lane; $lanes]);

            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            impl crate::fmt::Debug for $vector {
                fn fmt(
                    &self,
                    formatter: &mut crate::fmt::Formatter<'_>,
                ) -> crate::fmt::Result {
                    formatter.write_str(stringify!($vector))
                }
            }
        )+};
    }

    vector_types! {
        I8x16, i8, 16;
        U8x16, u8, 16;
        I16x8, i16, 8;
        U16x8, u16, 8;
        I32x4, i32, 4;
        U32x4, u32, 4;
        I64x2, i64, 2;
        U64x2, u64, 2;
        I8x32, i8, 32;
        U8x32, u8, 32;
        I16x16, i16, 16;
        U16x16, u16, 16;
        I32x8, i32, 8;
        U32x8, u32, 8;
        I64x4, i64, 4;
        U64x4, u64, 4;
        I8x64, i8, 64;
        U8x64, u8, 64;
        I16x32, i16, 32;
        U16x32, u16, 32;
        I32x16, i32, 16;
        U32x16, u32, 16;
        I64x8, i64, 8;
        U64x8, u64, 8;
    }

    macro_rules! vector_memory {
        ($($vector:ident, $lane:ty, $bits:ty, $lanes:literal,
            $splat:ident, $load:ident, $store:ident, $load_aligned:ident,
            $store_aligned:ident;)+) => {$(
            #[doc = concat!("Answers a `", stringify!($vector),
                "` whose every lane holds `bits`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $splat(bits: $bits) -> $vector {
                $vector([bits as $lane; $lanes])
            }

            #[doc = concat!("Reads one unaligned `", stringify!($vector), "` out of `data`.")]
            #[doc = ""]
            #[doc = "# Safety"]
            #[doc = ""]
            #[doc = "`data` must be readable for the vector's whole width."]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            #[cfg(target_abi = "polyasm")]
            pub unsafe fn $load(data: *const u8) -> $vector {
                unsafe { simd_load_unaligned::<u8, $vector>(data) }
            }

            #[doc = concat!("Reads one unaligned `", stringify!($vector), "` out of `data`.")]
            #[doc = ""]
            #[doc = "# Safety"]
            #[doc = ""]
            #[doc = "`data` must be readable for the vector's whole width."]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            #[cfg(not(target_abi = "polyasm"))]
            pub unsafe fn $load(data: *const u8) -> $vector {
                unsafe { data.cast::<$vector>().read_unaligned() }
            }

            #[doc = concat!("Writes one unaligned `", stringify!($vector), "` to `data`.")]
            #[doc = ""]
            #[doc = "# Safety"]
            #[doc = ""]
            #[doc = "`data` must be writable for the vector's whole width."]
            #[inline]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            #[cfg(target_abi = "polyasm")]
            pub unsafe fn $store(data: *mut u8, value: $vector) {
                unsafe { simd_store_unaligned::<u8, $vector>(data, value) }
            }

            #[doc = concat!("Writes one unaligned `", stringify!($vector), "` to `data`.")]
            #[doc = ""]
            #[doc = "# Safety"]
            #[doc = ""]
            #[doc = "`data` must be writable for the vector's whole width."]
            #[inline]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            #[cfg(not(target_abi = "polyasm"))]
            pub unsafe fn $store(data: *mut u8, value: $vector) {
                unsafe { data.cast::<$vector>().write_unaligned(value) }
            }

            #[doc = concat!("Reads one aligned `", stringify!($vector), "` out of `data`.")]
            #[doc = ""]
            #[doc = "# Safety"]
            #[doc = ""]
            #[doc = "`data` must be readable for the vector's whole width and aligned to it."]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub unsafe fn $load_aligned(data: *const u8) -> $vector {
                unsafe { data.cast::<$vector>().read() }
            }

            #[doc = concat!("Writes one aligned `", stringify!($vector), "` to `data`.")]
            #[doc = ""]
            #[doc = "# Safety"]
            #[doc = ""]
            #[doc = "`data` must be writable for the vector's whole width and aligned to it."]
            #[inline]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub unsafe fn $store_aligned(data: *mut u8, value: $vector) {
                unsafe { data.cast::<$vector>().write(value) }
            }
        )+};
    }

    vector_memory! {
        I8x16, i8, u8, 16,
            splat_i8x16, load_i8x16, store_i8x16, load_aligned_i8x16, store_aligned_i8x16;
        U8x16, u8, u8, 16,
            splat_u8x16, load_u8x16, store_u8x16, load_aligned_u8x16, store_aligned_u8x16;
        I16x8, i16, u16, 8,
            splat_i16x8, load_i16x8, store_i16x8, load_aligned_i16x8, store_aligned_i16x8;
        U16x8, u16, u16, 8,
            splat_u16x8, load_u16x8, store_u16x8, load_aligned_u16x8, store_aligned_u16x8;
        I32x4, i32, u32, 4,
            splat_i32x4, load_i32x4, store_i32x4, load_aligned_i32x4, store_aligned_i32x4;
        U32x4, u32, u32, 4,
            splat_u32x4, load_u32x4, store_u32x4, load_aligned_u32x4, store_aligned_u32x4;
        I64x2, i64, u64, 2,
            splat_i64x2, load_i64x2, store_i64x2, load_aligned_i64x2, store_aligned_i64x2;
        U64x2, u64, u64, 2,
            splat_u64x2, load_u64x2, store_u64x2, load_aligned_u64x2, store_aligned_u64x2;
        I8x32, i8, u8, 32,
            splat_i8x32, load_i8x32, store_i8x32, load_aligned_i8x32, store_aligned_i8x32;
        U8x32, u8, u8, 32,
            splat_u8x32, load_u8x32, store_u8x32, load_aligned_u8x32, store_aligned_u8x32;
        I16x16, i16, u16, 16,
            splat_i16x16, load_i16x16, store_i16x16, load_aligned_i16x16, store_aligned_i16x16;
        U16x16, u16, u16, 16,
            splat_u16x16, load_u16x16, store_u16x16, load_aligned_u16x16, store_aligned_u16x16;
        I32x8, i32, u32, 8,
            splat_i32x8, load_i32x8, store_i32x8, load_aligned_i32x8, store_aligned_i32x8;
        U32x8, u32, u32, 8,
            splat_u32x8, load_u32x8, store_u32x8, load_aligned_u32x8, store_aligned_u32x8;
        I64x4, i64, u64, 4,
            splat_i64x4, load_i64x4, store_i64x4, load_aligned_i64x4, store_aligned_i64x4;
        U64x4, u64, u64, 4,
            splat_u64x4, load_u64x4, store_u64x4, load_aligned_u64x4, store_aligned_u64x4;
        I8x64, i8, u8, 64,
            splat_i8x64, load_i8x64, store_i8x64, load_aligned_i8x64, store_aligned_i8x64;
        U8x64, u8, u8, 64,
            splat_u8x64, load_u8x64, store_u8x64, load_aligned_u8x64, store_aligned_u8x64;
        I16x32, i16, u16, 32,
            splat_i16x32, load_i16x32, store_i16x32, load_aligned_i16x32, store_aligned_i16x32;
        U16x32, u16, u16, 32,
            splat_u16x32, load_u16x32, store_u16x32, load_aligned_u16x32, store_aligned_u16x32;
        I32x16, i32, u32, 16,
            splat_i32x16, load_i32x16, store_i32x16, load_aligned_i32x16, store_aligned_i32x16;
        U32x16, u32, u32, 16,
            splat_u32x16, load_u32x16, store_u32x16, load_aligned_u32x16, store_aligned_u32x16;
        I64x8, i64, u64, 8,
            splat_i64x8, load_i64x8, store_i64x8, load_aligned_i64x8, store_aligned_i64x8;
        U64x8, u64, u64, 8,
            splat_u64x8, load_u64x8, store_u64x8, load_aligned_u64x8, store_aligned_u64x8;
    }

    macro_rules! vector_bitwise {
        ($($vector:ident, $lane:ty, $bits:ty, $lanes:literal,
            $and:ident, $or:ident, $xor:ident, $not:ident, $andnot:ident;)+) => {$(
            #[doc = concat!("Answers the bitwise conjunction of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $and(left: $vector, right: $vector) -> $vector {
                unsafe { simd_and(left, right) }
            }

            #[doc = concat!("Answers the bitwise disjunction of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $or(left: $vector, right: $vector) -> $vector {
                unsafe { simd_or(left, right) }
            }

            #[doc = concat!("Answers the bitwise exclusive disjunction of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $xor(left: $vector, right: $vector) -> $vector {
                unsafe { simd_xor(left, right) }
            }

            #[doc = concat!("Answers the bitwise complement of one `",
                stringify!($vector), "`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $not(value: $vector) -> $vector {
                unsafe { simd_xor(value, $vector([!(0 as $bits) as $lane; $lanes])) }
            }

            #[doc = concat!("Answers the conjunction of `right` with the complement of `left`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $andnot(left: $vector, right: $vector) -> $vector {
                unsafe {
                    simd_and(simd_xor(left, $vector([!(0 as $bits) as $lane; $lanes])), right)
                }
            }
        )+};
    }

    vector_bitwise! {
        I8x16, i8, u8, 16,
            and_i8x16, or_i8x16, xor_i8x16, not_i8x16, andnot_i8x16;
        U8x16, u8, u8, 16,
            and_u8x16, or_u8x16, xor_u8x16, not_u8x16, andnot_u8x16;
        I16x8, i16, u16, 8,
            and_i16x8, or_i16x8, xor_i16x8, not_i16x8, andnot_i16x8;
        U16x8, u16, u16, 8,
            and_u16x8, or_u16x8, xor_u16x8, not_u16x8, andnot_u16x8;
        I32x4, i32, u32, 4,
            and_i32x4, or_i32x4, xor_i32x4, not_i32x4, andnot_i32x4;
        U32x4, u32, u32, 4,
            and_u32x4, or_u32x4, xor_u32x4, not_u32x4, andnot_u32x4;
        I64x2, i64, u64, 2,
            and_i64x2, or_i64x2, xor_i64x2, not_i64x2, andnot_i64x2;
        U64x2, u64, u64, 2,
            and_u64x2, or_u64x2, xor_u64x2, not_u64x2, andnot_u64x2;
        I8x32, i8, u8, 32,
            and_i8x32, or_i8x32, xor_i8x32, not_i8x32, andnot_i8x32;
        U8x32, u8, u8, 32,
            and_u8x32, or_u8x32, xor_u8x32, not_u8x32, andnot_u8x32;
        I16x16, i16, u16, 16,
            and_i16x16, or_i16x16, xor_i16x16, not_i16x16, andnot_i16x16;
        U16x16, u16, u16, 16,
            and_u16x16, or_u16x16, xor_u16x16, not_u16x16, andnot_u16x16;
        I32x8, i32, u32, 8,
            and_i32x8, or_i32x8, xor_i32x8, not_i32x8, andnot_i32x8;
        U32x8, u32, u32, 8,
            and_u32x8, or_u32x8, xor_u32x8, not_u32x8, andnot_u32x8;
        I64x4, i64, u64, 4,
            and_i64x4, or_i64x4, xor_i64x4, not_i64x4, andnot_i64x4;
        U64x4, u64, u64, 4,
            and_u64x4, or_u64x4, xor_u64x4, not_u64x4, andnot_u64x4;
        I8x64, i8, u8, 64,
            and_i8x64, or_i8x64, xor_i8x64, not_i8x64, andnot_i8x64;
        U8x64, u8, u8, 64,
            and_u8x64, or_u8x64, xor_u8x64, not_u8x64, andnot_u8x64;
        I16x32, i16, u16, 32,
            and_i16x32, or_i16x32, xor_i16x32, not_i16x32, andnot_i16x32;
        U16x32, u16, u16, 32,
            and_u16x32, or_u16x32, xor_u16x32, not_u16x32, andnot_u16x32;
        I32x16, i32, u32, 16,
            and_i32x16, or_i32x16, xor_i32x16, not_i32x16, andnot_i32x16;
        U32x16, u32, u32, 16,
            and_u32x16, or_u32x16, xor_u32x16, not_u32x16, andnot_u32x16;
        I64x8, i64, u64, 8,
            and_i64x8, or_i64x8, xor_i64x8, not_i64x8, andnot_i64x8;
        U64x8, u64, u64, 8,
            and_u64x8, or_u64x8, xor_u64x8, not_u64x8, andnot_u64x8;
    }

    macro_rules! vector_arithmetic {
        ($($vector:ident, $signed:ident, $lane:ty, $lanes:literal,
            $add:ident, $sub:ident, $mul:ident, $saturating_add:ident,
            $saturating_sub:ident, $min:ident, $max:ident;)+) => {$(
            #[doc = concat!("Answers the wrapping per-lane sum of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $add(left: $vector, right: $vector) -> $vector {
                unsafe { simd_add(left, right) }
            }

            #[doc = concat!("Answers the wrapping per-lane difference of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $sub(left: $vector, right: $vector) -> $vector {
                unsafe { simd_sub(left, right) }
            }

            #[doc = concat!("Answers the wrapping per-lane product of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $mul(left: $vector, right: $vector) -> $vector {
                unsafe { simd_mul(left, right) }
            }

            #[doc = concat!("Answers the saturating per-lane sum of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $saturating_add(left: $vector, right: $vector) -> $vector {
                unsafe { simd_saturating_add(left, right) }
            }

            #[doc = concat!("Answers the saturating per-lane difference of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $saturating_sub(left: $vector, right: $vector) -> $vector {
                unsafe { simd_saturating_sub(left, right) }
            }

            #[doc = concat!("Answers the per-lane minimum of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $min(left: $vector, right: $vector) -> $vector {
                unsafe {
                    let order: $signed = simd_lt(left, right);
                    simd_select(order, left, right)
                }
            }

            #[doc = concat!("Answers the per-lane maximum of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $max(left: $vector, right: $vector) -> $vector {
                unsafe {
                    let order: $signed = simd_gt(left, right);
                    simd_select(order, left, right)
                }
            }
        )+};
    }

    vector_arithmetic! {
        I8x16, I8x16, i8, 16,
            add_i8x16, sub_i8x16, mul_i8x16, saturating_add_i8x16, saturating_sub_i8x16, min_i8x16, max_i8x16;
        U8x16, I8x16, u8, 16,
            add_u8x16, sub_u8x16, mul_u8x16, saturating_add_u8x16, saturating_sub_u8x16, min_u8x16, max_u8x16;
        I16x8, I16x8, i16, 8,
            add_i16x8, sub_i16x8, mul_i16x8, saturating_add_i16x8, saturating_sub_i16x8, min_i16x8, max_i16x8;
        U16x8, I16x8, u16, 8,
            add_u16x8, sub_u16x8, mul_u16x8, saturating_add_u16x8, saturating_sub_u16x8, min_u16x8, max_u16x8;
        I32x4, I32x4, i32, 4,
            add_i32x4, sub_i32x4, mul_i32x4, saturating_add_i32x4, saturating_sub_i32x4, min_i32x4, max_i32x4;
        U32x4, I32x4, u32, 4,
            add_u32x4, sub_u32x4, mul_u32x4, saturating_add_u32x4, saturating_sub_u32x4, min_u32x4, max_u32x4;
        I64x2, I64x2, i64, 2,
            add_i64x2, sub_i64x2, mul_i64x2, saturating_add_i64x2, saturating_sub_i64x2, min_i64x2, max_i64x2;
        U64x2, I64x2, u64, 2,
            add_u64x2, sub_u64x2, mul_u64x2, saturating_add_u64x2, saturating_sub_u64x2, min_u64x2, max_u64x2;
        I8x32, I8x32, i8, 32,
            add_i8x32, sub_i8x32, mul_i8x32, saturating_add_i8x32, saturating_sub_i8x32, min_i8x32, max_i8x32;
        U8x32, I8x32, u8, 32,
            add_u8x32, sub_u8x32, mul_u8x32, saturating_add_u8x32, saturating_sub_u8x32, min_u8x32, max_u8x32;
        I16x16, I16x16, i16, 16,
            add_i16x16, sub_i16x16, mul_i16x16, saturating_add_i16x16, saturating_sub_i16x16, min_i16x16, max_i16x16;
        U16x16, I16x16, u16, 16,
            add_u16x16, sub_u16x16, mul_u16x16, saturating_add_u16x16, saturating_sub_u16x16, min_u16x16, max_u16x16;
        I32x8, I32x8, i32, 8,
            add_i32x8, sub_i32x8, mul_i32x8, saturating_add_i32x8, saturating_sub_i32x8, min_i32x8, max_i32x8;
        U32x8, I32x8, u32, 8,
            add_u32x8, sub_u32x8, mul_u32x8, saturating_add_u32x8, saturating_sub_u32x8, min_u32x8, max_u32x8;
        I64x4, I64x4, i64, 4,
            add_i64x4, sub_i64x4, mul_i64x4, saturating_add_i64x4, saturating_sub_i64x4, min_i64x4, max_i64x4;
        U64x4, I64x4, u64, 4,
            add_u64x4, sub_u64x4, mul_u64x4, saturating_add_u64x4, saturating_sub_u64x4, min_u64x4, max_u64x4;
        I8x64, I8x64, i8, 64,
            add_i8x64, sub_i8x64, mul_i8x64, saturating_add_i8x64, saturating_sub_i8x64, min_i8x64, max_i8x64;
        U8x64, I8x64, u8, 64,
            add_u8x64, sub_u8x64, mul_u8x64, saturating_add_u8x64, saturating_sub_u8x64, min_u8x64, max_u8x64;
        I16x32, I16x32, i16, 32,
            add_i16x32, sub_i16x32, mul_i16x32, saturating_add_i16x32, saturating_sub_i16x32, min_i16x32, max_i16x32;
        U16x32, I16x32, u16, 32,
            add_u16x32, sub_u16x32, mul_u16x32, saturating_add_u16x32, saturating_sub_u16x32, min_u16x32, max_u16x32;
        I32x16, I32x16, i32, 16,
            add_i32x16, sub_i32x16, mul_i32x16, saturating_add_i32x16, saturating_sub_i32x16, min_i32x16, max_i32x16;
        U32x16, I32x16, u32, 16,
            add_u32x16, sub_u32x16, mul_u32x16, saturating_add_u32x16, saturating_sub_u32x16, min_u32x16, max_u32x16;
        I64x8, I64x8, i64, 8,
            add_i64x8, sub_i64x8, mul_i64x8, saturating_add_i64x8, saturating_sub_i64x8, min_i64x8, max_i64x8;
        U64x8, I64x8, u64, 8,
            add_u64x8, sub_u64x8, mul_u64x8, saturating_add_u64x8, saturating_sub_u64x8, min_u64x8, max_u64x8;
    }

    macro_rules! vector_shifts {
        ($($vector:ident, $lane:ty, $lanes:literal, $shl:ident, $shr:ident;)+) => {$(
            #[doc = concat!("Shifts every lane of a `", stringify!($vector),
                "` left by `bits`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            #[cfg(target_abi = "polyasm")]
            pub fn $shl(value: $vector, bits: u32) -> $vector {
                unsafe { simd_shl_scalar(value, bits) }
            }

            #[doc = concat!("Shifts every lane of a `", stringify!($vector),
                "` left by `bits`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            #[cfg(not(target_abi = "polyasm"))]
            pub fn $shl(value: $vector, bits: u32) -> $vector {
                unsafe { simd_shl(value, $vector([bits as $lane; $lanes])) }
            }

            #[doc = concat!("Shifts every lane of a `", stringify!($vector),
                "` right by `bits`, carrying the lane's own signedness.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            #[cfg(target_abi = "polyasm")]
            pub fn $shr(value: $vector, bits: u32) -> $vector {
                unsafe { simd_shr_scalar(value, bits) }
            }

            #[doc = concat!("Shifts every lane of a `", stringify!($vector),
                "` right by `bits`, carrying the lane's own signedness.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            #[cfg(not(target_abi = "polyasm"))]
            pub fn $shr(value: $vector, bits: u32) -> $vector {
                unsafe { simd_shr(value, $vector([bits as $lane; $lanes])) }
            }
        )+};
    }

    vector_shifts! {
        I8x16, i8, 16,
            shl_i8x16, shr_i8x16;
        U8x16, u8, 16,
            shl_u8x16, shr_u8x16;
        I16x8, i16, 8,
            shl_i16x8, shr_i16x8;
        U16x8, u16, 8,
            shl_u16x8, shr_u16x8;
        I32x4, i32, 4,
            shl_i32x4, shr_i32x4;
        U32x4, u32, 4,
            shl_u32x4, shr_u32x4;
        I64x2, i64, 2,
            shl_i64x2, shr_i64x2;
        U64x2, u64, 2,
            shl_u64x2, shr_u64x2;
        I8x32, i8, 32,
            shl_i8x32, shr_i8x32;
        U8x32, u8, 32,
            shl_u8x32, shr_u8x32;
        I16x16, i16, 16,
            shl_i16x16, shr_i16x16;
        U16x16, u16, 16,
            shl_u16x16, shr_u16x16;
        I32x8, i32, 8,
            shl_i32x8, shr_i32x8;
        U32x8, u32, 8,
            shl_u32x8, shr_u32x8;
        I64x4, i64, 4,
            shl_i64x4, shr_i64x4;
        U64x4, u64, 4,
            shl_u64x4, shr_u64x4;
        I8x64, i8, 64,
            shl_i8x64, shr_i8x64;
        U8x64, u8, 64,
            shl_u8x64, shr_u8x64;
        I16x32, i16, 32,
            shl_i16x32, shr_i16x32;
        U16x32, u16, 32,
            shl_u16x32, shr_u16x32;
        I32x16, i32, 16,
            shl_i32x16, shr_i32x16;
        U32x16, u32, 16,
            shl_u32x16, shr_u32x16;
        I64x8, i64, 8,
            shl_i64x8, shr_i64x8;
        U64x8, u64, 8,
            shl_u64x8, shr_u64x8;
    }

    macro_rules! vector_compare {
        ($($vector:ident, $signed:ident,
            $eq:ident, $ne:ident, $lt:ident, $le:ident, $gt:ident, $ge:ident,
            $select:ident;)+) => {$(
            #[doc = concat!("Answers all-ones in each lane where two `",
                stringify!($vector), "` values agree.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $eq(left: $vector, right: $vector) -> $signed {
                unsafe { simd_eq(left, right) }
            }

            #[doc = concat!("Answers all-ones in each lane where two `",
                stringify!($vector), "` values differ.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $ne(left: $vector, right: $vector) -> $signed {
                unsafe { simd_ne(left, right) }
            }

            #[doc = concat!("Answers all-ones in each lane where `left` is below `right`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $lt(left: $vector, right: $vector) -> $signed {
                unsafe { simd_lt(left, right) }
            }

            #[doc = concat!("Answers all-ones in each lane where `left` is at most `right`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $le(left: $vector, right: $vector) -> $signed {
                unsafe { simd_le(left, right) }
            }

            #[doc = concat!("Answers all-ones in each lane where `left` is above `right`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $gt(left: $vector, right: $vector) -> $signed {
                unsafe { simd_gt(left, right) }
            }

            #[doc = concat!("Answers all-ones in each lane where `left` is at least `right`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $ge(left: $vector, right: $vector) -> $signed {
                unsafe { simd_ge(left, right) }
            }

            #[doc = concat!("Answers `if_true` in every lane whose `mask` lane is all-ones ",
                "and `if_false` in every other.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $select(mask: $signed, if_true: $vector, if_false: $vector) -> $vector {
                unsafe { simd_select(mask, if_true, if_false) }
            }
        )+};
    }

    vector_compare! {
        I8x16, I8x16,
            eq_i8x16, ne_i8x16, lt_i8x16, le_i8x16, gt_i8x16, ge_i8x16, select_i8x16;
        U8x16, I8x16,
            eq_u8x16, ne_u8x16, lt_u8x16, le_u8x16, gt_u8x16, ge_u8x16, select_u8x16;
        I16x8, I16x8,
            eq_i16x8, ne_i16x8, lt_i16x8, le_i16x8, gt_i16x8, ge_i16x8, select_i16x8;
        U16x8, I16x8,
            eq_u16x8, ne_u16x8, lt_u16x8, le_u16x8, gt_u16x8, ge_u16x8, select_u16x8;
        I32x4, I32x4,
            eq_i32x4, ne_i32x4, lt_i32x4, le_i32x4, gt_i32x4, ge_i32x4, select_i32x4;
        U32x4, I32x4,
            eq_u32x4, ne_u32x4, lt_u32x4, le_u32x4, gt_u32x4, ge_u32x4, select_u32x4;
        I64x2, I64x2,
            eq_i64x2, ne_i64x2, lt_i64x2, le_i64x2, gt_i64x2, ge_i64x2, select_i64x2;
        U64x2, I64x2,
            eq_u64x2, ne_u64x2, lt_u64x2, le_u64x2, gt_u64x2, ge_u64x2, select_u64x2;
        I8x32, I8x32,
            eq_i8x32, ne_i8x32, lt_i8x32, le_i8x32, gt_i8x32, ge_i8x32, select_i8x32;
        U8x32, I8x32,
            eq_u8x32, ne_u8x32, lt_u8x32, le_u8x32, gt_u8x32, ge_u8x32, select_u8x32;
        I16x16, I16x16,
            eq_i16x16, ne_i16x16, lt_i16x16, le_i16x16, gt_i16x16, ge_i16x16, select_i16x16;
        U16x16, I16x16,
            eq_u16x16, ne_u16x16, lt_u16x16, le_u16x16, gt_u16x16, ge_u16x16, select_u16x16;
        I32x8, I32x8,
            eq_i32x8, ne_i32x8, lt_i32x8, le_i32x8, gt_i32x8, ge_i32x8, select_i32x8;
        U32x8, I32x8,
            eq_u32x8, ne_u32x8, lt_u32x8, le_u32x8, gt_u32x8, ge_u32x8, select_u32x8;
        I64x4, I64x4,
            eq_i64x4, ne_i64x4, lt_i64x4, le_i64x4, gt_i64x4, ge_i64x4, select_i64x4;
        U64x4, I64x4,
            eq_u64x4, ne_u64x4, lt_u64x4, le_u64x4, gt_u64x4, ge_u64x4, select_u64x4;
        I8x64, I8x64,
            eq_i8x64, ne_i8x64, lt_i8x64, le_i8x64, gt_i8x64, ge_i8x64, select_i8x64;
        U8x64, I8x64,
            eq_u8x64, ne_u8x64, lt_u8x64, le_u8x64, gt_u8x64, ge_u8x64, select_u8x64;
        I16x32, I16x32,
            eq_i16x32, ne_i16x32, lt_i16x32, le_i16x32, gt_i16x32, ge_i16x32, select_i16x32;
        U16x32, I16x32,
            eq_u16x32, ne_u16x32, lt_u16x32, le_u16x32, gt_u16x32, ge_u16x32, select_u16x32;
        I32x16, I32x16,
            eq_i32x16, ne_i32x16, lt_i32x16, le_i32x16, gt_i32x16, ge_i32x16, select_i32x16;
        U32x16, I32x16,
            eq_u32x16, ne_u32x16, lt_u32x16, le_u32x16, gt_u32x16, ge_u32x16, select_u32x16;
        I64x8, I64x8,
            eq_i64x8, ne_i64x8, lt_i64x8, le_i64x8, gt_i64x8, ge_i64x8, select_i64x8;
        U64x8, I64x8,
            eq_u64x8, ne_u64x8, lt_u64x8, le_u64x8, gt_u64x8, ge_u64x8, select_u64x8;
    }

    macro_rules! vector_lanes {
        ($($vector:ident, $lane:ty, $mask:ty,
            $bitmask:ident, $extract:ident, $replace:ident;)+) => {$(
            #[doc = concat!("Gathers the high bit of every lane of a `", stringify!($vector),
                "` into the low bits of an integer.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $bitmask(value: $vector) -> $mask {
                unsafe { simd_bitmask(value) }
            }

            #[doc = concat!("Answers lane `INDEX` of a `", stringify!($vector), "`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $extract<const INDEX: u32>(value: $vector) -> $lane {
                unsafe { simd_extract(value, INDEX) }
            }

            #[doc = concat!("Answers a `", stringify!($vector),
                "` whose lane `INDEX` is `lane` and whose other lanes are `value`'s.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $replace<const INDEX: u32>(value: $vector, lane: $lane) -> $vector {
                unsafe { simd_insert(value, INDEX, lane) }
            }
        )+};
    }

    vector_lanes! {
        I8x16, i8, u16,
            bitmask_i8x16, extract_i8x16, replace_i8x16;
        U8x16, u8, u16,
            bitmask_u8x16, extract_u8x16, replace_u8x16;
        I16x8, i16, u8,
            bitmask_i16x8, extract_i16x8, replace_i16x8;
        U16x8, u16, u8,
            bitmask_u16x8, extract_u16x8, replace_u16x8;
        I32x4, i32, u8,
            bitmask_i32x4, extract_i32x4, replace_i32x4;
        U32x4, u32, u8,
            bitmask_u32x4, extract_u32x4, replace_u32x4;
        I64x2, i64, u8,
            bitmask_i64x2, extract_i64x2, replace_i64x2;
        U64x2, u64, u8,
            bitmask_u64x2, extract_u64x2, replace_u64x2;
        I8x32, i8, u32,
            bitmask_i8x32, extract_i8x32, replace_i8x32;
        U8x32, u8, u32,
            bitmask_u8x32, extract_u8x32, replace_u8x32;
        I16x16, i16, u16,
            bitmask_i16x16, extract_i16x16, replace_i16x16;
        U16x16, u16, u16,
            bitmask_u16x16, extract_u16x16, replace_u16x16;
        I32x8, i32, u8,
            bitmask_i32x8, extract_i32x8, replace_i32x8;
        U32x8, u32, u8,
            bitmask_u32x8, extract_u32x8, replace_u32x8;
        I64x4, i64, u8,
            bitmask_i64x4, extract_i64x4, replace_i64x4;
        U64x4, u64, u8,
            bitmask_u64x4, extract_u64x4, replace_u64x4;
        I8x64, i8, u64,
            bitmask_i8x64, extract_i8x64, replace_i8x64;
        U8x64, u8, u64,
            bitmask_u8x64, extract_u8x64, replace_u8x64;
        I16x32, i16, u32,
            bitmask_i16x32, extract_i16x32, replace_i16x32;
        U16x32, u16, u32,
            bitmask_u16x32, extract_u16x32, replace_u16x32;
        I32x16, i32, u16,
            bitmask_i32x16, extract_i32x16, replace_i32x16;
        U32x16, u32, u16,
            bitmask_u32x16, extract_u32x16, replace_u32x16;
        I64x8, i64, u8,
            bitmask_i64x8, extract_i64x8, replace_i64x8;
        U64x8, u64, u8,
            bitmask_u64x8, extract_u64x8, replace_u64x8;
    }

    macro_rules! vector_signed {
        ($($vector:ident, $unsigned:ident, $lane:ty, $bits:ty, $lanes:literal,
            $neg:ident, $abs:ident, $shr_logical:ident, $all_true:ident,
            $any_true:ident;)+) => {$(
            #[doc = concat!("Answers the wrapping per-lane negation of one `",
                stringify!($vector), "`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $neg(value: $vector) -> $vector {
                unsafe { simd_neg(value) }
            }

            #[doc = concat!("Answers the per-lane absolute value of one `",
                stringify!($vector), "`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $abs(value: $vector) -> $vector {
                unsafe {
                    let negative: $vector = simd_lt(value, $vector([0; $lanes]));
                    simd_select(negative, simd_neg(value), value)
                }
            }

            #[doc = concat!("Shifts every lane of a `", stringify!($vector),
                "` right by `bits` without carrying the sign in.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            #[cfg(target_abi = "polyasm")]
            pub fn $shr_logical(value: $vector, bits: u32) -> $vector {
                unsafe {
                    let unsigned: $unsigned = crate::mem::transmute(value);
                    crate::mem::transmute(simd_shr_scalar(unsigned, bits))
                }
            }

            #[doc = concat!("Shifts every lane of a `", stringify!($vector),
                "` right by `bits` without carrying the sign in.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            #[cfg(not(target_abi = "polyasm"))]
            pub fn $shr_logical(value: $vector, bits: u32) -> $vector {
                unsafe {
                    let shifted = simd_shr(value, $vector([bits as $lane; $lanes]));
                    let keep = (!(0 as $bits) >> bits) as $lane;
                    simd_and(shifted, $vector([keep; $lanes]))
                }
            }

            #[doc = concat!("Answers whether every lane of a `", stringify!($vector),
                "` mask is set.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $all_true(value: $vector) -> bool {
                unsafe { simd_reduce_all(value) }
            }

            #[doc = concat!("Answers whether any lane of a `", stringify!($vector),
                "` mask is set.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $any_true(value: $vector) -> bool {
                unsafe { simd_reduce_any(value) }
            }
        )+};
    }

    vector_signed! {
        I8x16, U8x16, i8, u8, 16,
            neg_i8x16, abs_i8x16, shr_logical_i8x16, all_true_i8x16, any_true_i8x16;
        I16x8, U16x8, i16, u16, 8,
            neg_i16x8, abs_i16x8, shr_logical_i16x8, all_true_i16x8, any_true_i16x8;
        I32x4, U32x4, i32, u32, 4,
            neg_i32x4, abs_i32x4, shr_logical_i32x4, all_true_i32x4, any_true_i32x4;
        I64x2, U64x2, i64, u64, 2,
            neg_i64x2, abs_i64x2, shr_logical_i64x2, all_true_i64x2, any_true_i64x2;
        I8x32, U8x32, i8, u8, 32,
            neg_i8x32, abs_i8x32, shr_logical_i8x32, all_true_i8x32, any_true_i8x32;
        I16x16, U16x16, i16, u16, 16,
            neg_i16x16, abs_i16x16, shr_logical_i16x16, all_true_i16x16, any_true_i16x16;
        I32x8, U32x8, i32, u32, 8,
            neg_i32x8, abs_i32x8, shr_logical_i32x8, all_true_i32x8, any_true_i32x8;
        I64x4, U64x4, i64, u64, 4,
            neg_i64x4, abs_i64x4, shr_logical_i64x4, all_true_i64x4, any_true_i64x4;
        I8x64, U8x64, i8, u8, 64,
            neg_i8x64, abs_i8x64, shr_logical_i8x64, all_true_i8x64, any_true_i8x64;
        I16x32, U16x32, i16, u16, 32,
            neg_i16x32, abs_i16x32, shr_logical_i16x32, all_true_i16x32, any_true_i16x32;
        I32x16, U32x16, i32, u32, 16,
            neg_i32x16, abs_i32x16, shr_logical_i32x16, all_true_i32x16, any_true_i32x16;
        I64x8, U64x8, i64, u64, 8,
            neg_i64x8, abs_i64x8, shr_logical_i64x8, all_true_i64x8, any_true_i64x8;
    }

    macro_rules! vector_unsigned {
        ($($vector:ident, $lane:ty, $lanes:literal, $avg:ident;)+) => {$(
            #[doc = concat!("Answers the per-lane rounding average of two `",
                stringify!($vector), "` values without overflowing a lane.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $avg(left: $vector, right: $vector) -> $vector {
                unsafe {
                    let carry = simd_shr(simd_xor(left, right), $vector([1; $lanes]));
                    simd_sub(simd_or(left, right), carry)
                }
            }
        )+};
    }

    vector_unsigned! {
        U8x16, u8, 16,
            avg_u8x16;
        U16x8, u16, 8,
            avg_u16x8;
        U32x4, u32, 4,
            avg_u32x4;
        U64x2, u64, 2,
            avg_u64x2;
        U8x32, u8, 32,
            avg_u8x32;
        U16x16, u16, 16,
            avg_u16x16;
        U32x8, u32, 8,
            avg_u32x8;
        U64x4, u64, 4,
            avg_u64x4;
        U8x64, u8, 64,
            avg_u8x64;
        U16x32, u16, 32,
            avg_u16x32;
        U32x16, u32, 16,
            avg_u32x16;
        U64x8, u64, 8,
            avg_u64x8;
    }

    macro_rules! vector_bytes {
        ($($vector:ident, $lane:ty, $lanes:literal,
            $swizzle:ident, $shift_in_one:ident, $shift_in_two:ident, $shift_in_three:ident,
            $half_shift_in_one:ident, $half_shift_in_two:ident, $half_shift_in_three:ident,
            $interleave_low:ident, $interleave_high:ident;)+) => {$(
            #[doc = concat!("Answers a `", stringify!($vector),
                "` whose lane `i` is `table[indices[i]]` inside `i`'s own sixteen-byte lane, ",
                "and zero where that index byte is sixteen or more read as unsigned.")]
            #[cfg(target_abi = "polyasm")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $swizzle(table: $vector, indices: $vector) -> $vector {
                unsafe { simd_swizzle_dyn(table, indices) }
            }

            #[doc = concat!("Answers a `", stringify!($vector),
                "` whose lane `i` is `table[indices[i]]` inside `i`'s own sixteen-byte lane, ",
                "and zero where that index byte is sixteen or more read as unsigned.")]
            #[cfg(not(target_abi = "polyasm"))]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $swizzle(table: $vector, indices: $vector) -> $vector {
                let mut result = $vector([0; $lanes]);
                let mut lane = 0usize;
                while lane < $lanes {
                    let index = unsafe {
                        simd_extract_dyn::<$vector, $lane>(indices, lane as u32)
                    } as u8;
                    let value = if index >= 16 {
                        0 as $lane
                    } else {
                        let base = (lane / 16 * 16) as u32 + u32::from(index);
                        unsafe { simd_extract_dyn::<$vector, $lane>(table, base) }
                    };
                    result = unsafe { simd_insert_dyn(result, lane as u32, value) };
                    lane += 1;
                }
                result
            }

            #[doc = concat!("Answers the whole-vector window of `other` below `value` that ",
                "starts one byte before `value`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $shift_in_one(value: $vector, other: $vector) -> $vector {
                unsafe {
                    simd_shuffle(other, value, const { ShuffleIndex(align_right::<$lanes>(1)) })
                }
            }

            #[doc = concat!("Answers the whole-vector window of `other` below `value` that ",
                "starts two bytes before `value`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $shift_in_two(value: $vector, other: $vector) -> $vector {
                unsafe {
                    simd_shuffle(other, value, const { ShuffleIndex(align_right::<$lanes>(2)) })
                }
            }

            #[doc = concat!("Answers the whole-vector window of `other` below `value` that ",
                "starts three bytes before `value`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $shift_in_three(value: $vector, other: $vector) -> $vector {
                unsafe {
                    simd_shuffle(other, value, const { ShuffleIndex(align_right::<$lanes>(3)) })
                }
            }

            #[doc = concat!("Answers the same window as the whole-vector shift, taken inside ",
                "every sixteen-byte lane on its own.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $half_shift_in_one(value: $vector, other: $vector) -> $vector {
                unsafe {
                    simd_shuffle(
                        other,
                        value,
                        const { ShuffleIndex(align_right_in_lane::<$lanes>(1)) },
                    )
                }
            }

            #[doc = concat!("Answers the same window as the whole-vector shift, taken inside ",
                "every sixteen-byte lane on its own.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $half_shift_in_two(value: $vector, other: $vector) -> $vector {
                unsafe {
                    simd_shuffle(
                        other,
                        value,
                        const { ShuffleIndex(align_right_in_lane::<$lanes>(2)) },
                    )
                }
            }

            #[doc = concat!("Answers the same window as the whole-vector shift, taken inside ",
                "every sixteen-byte lane on its own.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $half_shift_in_three(value: $vector, other: $vector) -> $vector {
                unsafe {
                    simd_shuffle(
                        other,
                        value,
                        const { ShuffleIndex(align_right_in_lane::<$lanes>(3)) },
                    )
                }
            }

            #[doc = concat!("Interleaves the low eight bytes of each sixteen-byte lane of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $interleave_low(left: $vector, right: $vector) -> $vector {
                unsafe {
                    simd_shuffle(left, right, const { ShuffleIndex(interleave::<$lanes>(0)) })
                }
            }

            #[doc = concat!("Interleaves the high eight bytes of each sixteen-byte lane of two `",
                stringify!($vector), "` values.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $interleave_high(left: $vector, right: $vector) -> $vector {
                unsafe {
                    simd_shuffle(left, right, const { ShuffleIndex(interleave::<$lanes>(8)) })
                }
            }
        )+};
    }

    vector_bytes! {
        I8x16, i8, 16,
            swizzle_i8x16, shift_in_one_byte_i8x16, shift_in_two_bytes_i8x16, shift_in_three_bytes_i8x16, half_shift_in_one_byte_i8x16, half_shift_in_two_bytes_i8x16, half_shift_in_three_bytes_i8x16, interleave_low_bytes_i8x16, interleave_high_bytes_i8x16;
        U8x16, u8, 16,
            swizzle_u8x16, shift_in_one_byte_u8x16, shift_in_two_bytes_u8x16, shift_in_three_bytes_u8x16, half_shift_in_one_byte_u8x16, half_shift_in_two_bytes_u8x16, half_shift_in_three_bytes_u8x16, interleave_low_bytes_u8x16, interleave_high_bytes_u8x16;
        I8x32, i8, 32,
            swizzle_i8x32, shift_in_one_byte_i8x32, shift_in_two_bytes_i8x32, shift_in_three_bytes_i8x32, half_shift_in_one_byte_i8x32, half_shift_in_two_bytes_i8x32, half_shift_in_three_bytes_i8x32, interleave_low_bytes_i8x32, interleave_high_bytes_i8x32;
        U8x32, u8, 32,
            swizzle_u8x32, shift_in_one_byte_u8x32, shift_in_two_bytes_u8x32, shift_in_three_bytes_u8x32, half_shift_in_one_byte_u8x32, half_shift_in_two_bytes_u8x32, half_shift_in_three_bytes_u8x32, interleave_low_bytes_u8x32, interleave_high_bytes_u8x32;
        I8x64, i8, 64,
            swizzle_i8x64, shift_in_one_byte_i8x64, shift_in_two_bytes_i8x64, shift_in_three_bytes_i8x64, half_shift_in_one_byte_i8x64, half_shift_in_two_bytes_i8x64, half_shift_in_three_bytes_i8x64, interleave_low_bytes_i8x64, interleave_high_bytes_i8x64;
        U8x64, u8, 64,
            swizzle_u8x64, shift_in_one_byte_u8x64, shift_in_two_bytes_u8x64, shift_in_three_bytes_u8x64, half_shift_in_one_byte_u8x64, half_shift_in_two_bytes_u8x64, half_shift_in_three_bytes_u8x64, interleave_low_bytes_u8x64, interleave_high_bytes_u8x64;
    }

    macro_rules! vector_wide_bytes {
        ($($vector:ident, $half:ident, $lanes:literal,
            $swap:ident, $broadcast:ident;)+) => {$(
            #[doc = concat!("Swaps the two sixteen-byte lanes of every thirty-two-byte lane of a `",
                stringify!($vector), "`.")]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub fn $swap(value: $vector) -> $vector {
                unsafe {
                    simd_shuffle(value, value, const { ShuffleIndex(swap_lanes::<$lanes>()) })
                }
            }

            #[doc = concat!("Reads sixteen unaligned bytes and repeats them across a `",
                stringify!($vector), "`.")]
            #[doc = ""]
            #[doc = "# Safety"]
            #[doc = ""]
            #[doc = "`data` must be readable for sixteen bytes."]
            #[inline]
            #[must_use]
            #[stable(feature = "polyasm_vector", since = "CURRENT_RUSTC_VERSION")]
            pub unsafe fn $broadcast(data: *const u8) -> $vector {
                let half = unsafe { data.cast::<$half>().read_unaligned() };
                unsafe {
                    simd_shuffle(half, half, const { ShuffleIndex(broadcast_lane::<$lanes>()) })
                }
            }
        )+};
    }

    vector_wide_bytes! {
        I8x32, I8x16, 32,
            swap_128bit_lanes_i8x32, load_broadcast_128_i8x32;
        U8x32, U8x16, 32,
            swap_128bit_lanes_u8x32, load_broadcast_128_u8x32;
        I8x64, I8x16, 64,
            swap_128bit_lanes_i8x64, load_broadcast_128_i8x64;
        U8x64, U8x16, 64,
            swap_128bit_lanes_u8x64, load_broadcast_128_u8x64;
    }
}

// ---------------------------------------------------------------------------
// Packet rows, over a runtime-lent context (`super::packet`).
// ---------------------------------------------------------------------------

#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
use super::packet::{
    PacketDataEnd, PacketDataLoad8Abs, PacketDataLoad8Ind, PacketDataLoad16BeAbs,
    PacketDataLoad16BeInd, PacketDataLoad32BeAbs, PacketDataLoad32BeInd, PacketDataRange,
    PacketDataStart, PacketMacLoad8Abs, PacketMacLoad8Ind, PacketMacLoad16BeAbs,
    PacketMacLoad16BeInd, PacketMacLoad32BeAbs, PacketMacLoad32BeInd, PacketNetworkLoad8Abs,
    PacketNetworkLoad8Ind, PacketNetworkLoad16BeAbs, PacketNetworkLoad16BeInd,
    PacketNetworkLoad32BeAbs, PacketNetworkLoad32BeInd,
};

/// Checks a constant prefix with exactly the `PacketDataRange` instruction.
///
/// The result is `true` exactly when the original start plus unsigned LENGTH
/// stays in range and is at or before its matching end. Failure is `false`,
/// distinct from the whole-invocation zero termination of a packet load. The
/// check reads zero bytes and keeps every pointer. The full-width machine
/// answer is 0 or 1.
///
/// This intrinsic carries the checked range as well as its boolean result.
/// Lowerings preserve its boundary operands and the checked range on the true
/// path, including through scalar spills and calls. They keep the predicate
/// itself, its original start pointer and the order of the loads it guards,
/// and keep the predicate where its answer is read by this check alone.
/// LENGTH values greater than u32::MAX are compile-time errors, also when
/// this intrinsic is called directly instead of through `prefix`.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_data_range<const LENGTH: usize>(operands: PacketDataRange<'_, LENGTH>) -> bool;

/// Executes exactly the `PacketDataLoad16BeAbs` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_data_load16be_abs<const DISPLACEMENT: i32>(
    operands: PacketDataLoad16BeAbs<'_, DISPLACEMENT>,
) -> u64;

/// Executes exactly the `PacketDataLoad16BeInd` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_data_load16be_ind(operands: PacketDataLoad16BeInd<'_>) -> u64;

/// Executes exactly the `PacketDataLoad32BeAbs` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_data_load32be_abs<const DISPLACEMENT: i32>(
    operands: PacketDataLoad32BeAbs<'_, DISPLACEMENT>,
) -> u64;

/// Executes exactly the `PacketDataLoad32BeInd` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_data_load32be_ind(operands: PacketDataLoad32BeInd<'_>) -> u64;

/// Executes exactly the `PacketDataLoad8Abs` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_data_load8_abs<const DISPLACEMENT: i32>(
    operands: PacketDataLoad8Abs<'_, DISPLACEMENT>,
) -> u64;

/// Executes exactly the `PacketDataLoad8Ind` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_data_load8_ind(operands: PacketDataLoad8Ind<'_>) -> u64;

/// Executes exactly the `PacketMacLoad16BeAbs` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_mac_load16be_abs<const DISPLACEMENT: i32>(
    operands: PacketMacLoad16BeAbs<'_, DISPLACEMENT>,
) -> u64;

/// Executes exactly the `PacketMacLoad16BeInd` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_mac_load16be_ind(operands: PacketMacLoad16BeInd<'_>) -> u64;

/// Executes exactly the `PacketMacLoad32BeAbs` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_mac_load32be_abs<const DISPLACEMENT: i32>(
    operands: PacketMacLoad32BeAbs<'_, DISPLACEMENT>,
) -> u64;

/// Executes exactly the `PacketMacLoad32BeInd` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_mac_load32be_ind(operands: PacketMacLoad32BeInd<'_>) -> u64;

/// Executes exactly the `PacketMacLoad8Abs` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_mac_load8_abs<const DISPLACEMENT: i32>(
    operands: PacketMacLoad8Abs<'_, DISPLACEMENT>,
) -> u64;

/// Executes exactly the `PacketMacLoad8Ind` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_mac_load8_ind(operands: PacketMacLoad8Ind<'_>) -> u64;

/// Executes exactly the `PacketNetworkLoad16BeAbs` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_network_load16be_abs<const DISPLACEMENT: i32>(
    operands: PacketNetworkLoad16BeAbs<'_, DISPLACEMENT>,
) -> u64;

/// Executes exactly the `PacketNetworkLoad16BeInd` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_network_load16be_ind(operands: PacketNetworkLoad16BeInd<'_>) -> u64;

/// Executes exactly the `PacketNetworkLoad32BeAbs` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_network_load32be_abs<const DISPLACEMENT: i32>(
    operands: PacketNetworkLoad32BeAbs<'_, DISPLACEMENT>,
) -> u64;

/// Executes exactly the `PacketNetworkLoad32BeInd` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_network_load32be_ind(operands: PacketNetworkLoad32BeInd<'_>) -> u64;

/// Executes exactly the `PacketNetworkLoad8Abs` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_network_load8_abs<const DISPLACEMENT: i32>(
    operands: PacketNetworkLoad8Abs<'_, DISPLACEMENT>,
) -> u64;

/// Executes exactly the `PacketNetworkLoad8Ind` packet instruction.
///
/// This intrinsic is safe to call with a borrowed context. Invalid packet
/// ranges end the guest invocation with result zero, so every Rust reference
/// stays valid. The answer is zero-extended to the complete register.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_network_load8_ind(operands: PacketNetworkLoad8Ind<'_>) -> u64;

/// Executes exactly the `PacketDataStart` packet boundary instruction.
///
/// The pointer retains this context's allocation provenance and full address
/// width. The paired start and end bound its current contiguous readable window.
/// It is safe to obtain this raw pointer; dereferencing it still requires the
/// caller to preserve the context's borrow and prove the accessed range.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_data_start(operands: PacketDataStart<'_>) -> *const u8;

/// Executes exactly the `PacketDataEnd` packet boundary instruction.
///
/// The pointer retains this context's allocation provenance and full address
/// width. The paired start and end bound its current contiguous readable window.
/// It is safe to obtain this raw pointer; dereferencing it still requires the
/// caller to preserve the context's borrow and prove the accessed range.
#[inline]
#[must_use]
#[cfg(all(target_abi = "polyasm", target_pointer_width = "64"))]
#[rustc_intrinsic]
#[rustc_nounwind]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub fn packet_data_end(operands: PacketDataEnd<'_>) -> *const u8;
