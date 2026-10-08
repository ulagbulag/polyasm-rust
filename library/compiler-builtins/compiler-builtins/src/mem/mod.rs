// Trying to satisfy clippy here is hopeless
#![allow(clippy::style)]
// FIXME(e2024): this eventually needs to be removed.
#![allow(unsafe_op_in_unsafe_fn)]
// FIXME(bench): remove when the benchmark rustc version is updated
#![allow(unknown_lints)]

// memcpy/memmove/memset have optimized implementations on some architectures
#[cfg_attr(all(feature = "arch", target_arch = "x86_64"), path = "x86_64.rs")]
mod impls;

/// The functions of the memory extension instruction set of PolyASM that
/// `memcpy`, `memmove`, `memset`, `memcmp` and `bcmp` call on PolyASM targets.
/// Each one carries the namespace, the name and the parameters of its function
/// in the `polyasm-plugin-memory` set, and its body stands as the fallback a
/// machine runs where it spells zero engines for the function. Each one stays
/// a function of its own through the LTO of an image, so the image keeps its
/// namespace and its name.
#[cfg(target_arch = "polyasm")]
mod polyasm {
    /// Copies the `length` bytes from `from` on to `to`, two regions apart.
    #[core::polyasm::extension(memory, copy)]
    #[inline(never)]
    pub(super) unsafe extern "polyasm" fn __copy(to: *mut u8, from: *const u8, length: usize) {
        core::intrinsics::copy_nonoverlapping(from, to, length)
    }

    /// Copies the `length` bytes from `from` on to `to`, the two regions free to overlap.
    #[core::polyasm::extension(memory, copy_within)]
    #[inline(never)]
    pub(super) unsafe extern "polyasm" fn __copy_within(
        to: *mut u8,
        from: *const u8,
        length: usize,
    ) {
        core::intrinsics::copy(from, to, length)
    }

    /// Writes `value` into the `length` bytes from `to` on.
    #[core::polyasm::extension(memory, fill)]
    #[inline(never)]
    pub(super) unsafe extern "polyasm" fn __fill(to: *mut u8, value: u8, length: usize) {
        core::intrinsics::write_bytes(to, value, length)
    }

    /// Answers zero where the `length` bytes from `a` on match those from `b` on, and otherwise
    /// the first byte of `a` that differs less the byte of `b` at its place.
    #[core::polyasm::extension(memory, compare)]
    #[inline(never)]
    pub(super) unsafe extern "polyasm" fn __compare(
        a: *const u8,
        b: *const u8,
        length: usize,
    ) -> i32 {
        super::impls::compare_bytes(a, b, length)
    }
}

#[cfg(any(windows, target_os = "uefi"))]
#[expect(non_camel_case_types)]
pub type wchar_t = u16;
#[cfg(not(any(windows, target_os = "uefi")))]
#[expect(non_camel_case_types)]
pub type wchar_t = i32;

intrinsics! {
    #[mem_builtin]
    #[allow(suspicious_runtime_symbol_definitions)]
    pub unsafe extern "C" fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
        core::cfg_select! {
            target_arch = "polyasm" => polyasm::__copy(dest, src, n),
            _ => impls::copy_forward(dest, src, n),
        }
        dest
    }

    #[mem_builtin]
    #[allow(suspicious_runtime_symbol_definitions)]
    pub unsafe extern "C" fn memmove(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
        core::cfg_select! {
            target_arch = "polyasm" => polyasm::__copy_within(dest, src, n),
            _ => {
                let delta = (dest as usize).wrapping_sub(src as usize);
                if delta >= n {
                    // We can copy forwards because either dest is far enough ahead of src,
                    // or src is ahead of dest (and delta overflowed).
                    impls::copy_forward(dest, src, n);
                } else {
                    impls::copy_backward(dest, src, n);
                }
            }
        }
        dest
    }

    #[mem_builtin]
    #[allow(suspicious_runtime_symbol_definitions)]
    pub unsafe extern "C" fn memset(s: *mut u8, c: core::ffi::c_int, n: usize) -> *mut u8 {
        core::cfg_select! {
            target_arch = "polyasm" => polyasm::__fill(s, c as u8, n),
            _ => impls::set_bytes(s, c as u8, n),
        }
        s
    }

    #[mem_builtin]
    #[allow(suspicious_runtime_symbol_definitions)]
    pub unsafe extern "C" fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> core::ffi::c_int {
        core::cfg_select! {
            target_arch = "polyasm" => polyasm::__compare(s1, s2, n),
            _ => impls::compare_bytes(s1, s2, n),
        }
    }

    #[mem_builtin]
    #[allow(suspicious_runtime_symbol_definitions)]
    pub unsafe extern "C" fn bcmp(s1: *const u8, s2: *const u8, n: usize) -> core::ffi::c_int {
        core::cfg_select! {
            target_arch = "polyasm" => polyasm::__compare(s1, s2, n),
            _ => memcmp(s1, s2, n),
        }
    }

    #[mem_builtin]
    pub unsafe extern "C" fn strlen(s: *const core::ffi::c_char) -> usize {
        impls::c_string_length(s)
    }

    #[mem_builtin]
    pub unsafe extern "C" fn wcslen(s: *const crate::mem::wchar_t) -> usize {
        let mut s = s;
        let mut n = 0;
        while *s != 0 {
            n += 1;
            s = s.add(1);
        }
        n
    }
}
