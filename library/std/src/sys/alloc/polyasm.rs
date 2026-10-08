//! This is an implementation of a global allocator on PolyASM targets. The
//! image itself is the whole runtime, so we provide our own allocator!
//!
//! PolyASM carries WebAssembly's family with a heap fixed at link time: the
//! linker decides the whole heap and names it by the two PolyASM boundary
//! symbols that `polyasm-format` spells, `polyasm::heap-base` and
//! `polyasm::heap-end`. That one heap is the foundation on which we build an
//! allocator, so we do so!
//!
//! The current allocator here is the `dlmalloc` crate over a system allocator
//! that donates the linked heap once. A PolyASM image runs one thread, so the
//! allocator runs unsynchronized.

use core::cell::{Cell, SyncUnsafeCell};
use core::ptr;

use crate::alloc::Layout;

unsafe extern "C" {
    #[link_name = "polyasm::heap-base"]
    static HEAP_BASE: u8;

    #[link_name = "polyasm::heap-end"]
    static HEAP_END: u8;
}

struct SyncDlmalloc(dlmalloc::Dlmalloc<System>);
unsafe impl Sync for SyncDlmalloc {}

static DLMALLOC: SyncUnsafeCell<SyncDlmalloc> =
    SyncUnsafeCell::new(SyncDlmalloc(dlmalloc::Dlmalloc::new_with_allocator(System::new())));

#[inline]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    // SAFETY: DLMALLOC access is guaranteed to be safe because the lock gives us unique and non-reentrant access.
    // Calling malloc() is safe because preconditions on this function match the trait method preconditions.
    let _lock = lock::lock();
    unsafe { (*DLMALLOC.get()).0.malloc(layout.size(), layout.align()) }
}

#[inline]
pub unsafe fn alloc_zeroed(layout: Layout) -> *mut u8 {
    // SAFETY: DLMALLOC access is guaranteed to be safe because the lock gives us unique and non-reentrant access.
    // Calling calloc() is safe because preconditions on this function match the trait method preconditions.
    let _lock = lock::lock();
    unsafe { (*DLMALLOC.get()).0.calloc(layout.size(), layout.align()) }
}

#[inline]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    // SAFETY: DLMALLOC access is guaranteed to be safe because the lock gives us unique and non-reentrant access.
    // Calling free() is safe because preconditions on this function match the trait method preconditions.
    let _lock = lock::lock();
    unsafe { (*DLMALLOC.get()).0.free(ptr, layout.size(), layout.align()) }
}

#[inline]
pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    // SAFETY: DLMALLOC access is guaranteed to be safe because the lock gives us unique and non-reentrant access.
    // Calling realloc() is safe because preconditions on this function match the trait method preconditions.
    let _lock = lock::lock();
    unsafe { (*DLMALLOC.get()).0.realloc(ptr, layout.size(), layout.align(), new_size) }
}

/// The system allocator PolyASM donates its linked heap through.
struct System {
    donated: Cell<bool>,
}

impl System {
    /// Answers a system allocator that holds its heap for donation.
    const fn new() -> System {
        System { donated: Cell::new(false) }
    }
}

/// Answers the one heap the linker fixed into this image.
///
/// The linker decides the whole heap and names it by the two PolyASM boundary
/// symbols. The allocator takes it once and answers `None` after, since a
/// second donation would hand out the same bytes twice.
fn linked_heap(size: usize) -> Option<(usize, usize)> {
    let base = (&raw const HEAP_BASE).expose_provenance();
    let end = (&raw const HEAP_END).expose_provenance();
    if base == 0 || end <= base {
        return None;
    }
    let len = end - base;
    if len < size { None } else { Some((base, len)) }
}

unsafe impl dlmalloc::Allocator for System {
    fn alloc(&self, size: usize) -> (*mut u8, usize, u32) {
        if size == 0 || self.donated.get() {
            return (ptr::null_mut(), 0, 0);
        }
        match linked_heap(size) {
            Some((base, len)) => {
                self.donated.set(true);
                (ptr::with_exposed_provenance_mut(base), len, 0)
            }
            None => (ptr::null_mut(), 0, 0),
        }
    }

    fn remap(&self, _ptr: *mut u8, _oldsize: usize, _newsize: usize, _can_move: bool) -> *mut u8 {
        ptr::null_mut()
    }

    fn free_part(&self, _ptr: *mut u8, _oldsize: usize, _newsize: usize) -> bool {
        false
    }

    fn free(&self, _ptr: *mut u8, _size: usize) -> bool {
        false
    }

    fn can_release_part(&self, _flags: u32) -> bool {
        false
    }

    fn allocates_zeros(&self) -> bool {
        false
    }

    /// Answers the page the linked heap is measured in, 64 KiB.
    ///
    /// dlmalloc reads this on the paths that size a segment, so it answers a
    /// page; the malloc alignment serves a different purpose.
    fn page_size(&self) -> usize {
        64 * 1024
    }
}

mod lock {
    #[inline]
    pub fn lock() {} // one thread, an empty lock
}
