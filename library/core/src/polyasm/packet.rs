//! Packet instructions for a runtime-lent context.
//!
//! Packet storage is separate from guest linear memory. This API serves
//! 64-bit PolyASM targets so a context or boundary keeps its full width beyond
//! a 32-bit guest offset. A machine preserves each pointer's packet identity.
//! The machine alone owns the context layout and the executable behavior.
//!
//! Each load names one fixed origin, width and byte order. A failed load ends
//! the entire guest invocation with result zero, including when called from a
//! nested function, with zero unwinding and zero destructors run. Missing MAC
//! storage, arithmetic overflow and a range outside logical packet storage are
//! failures. Successful loads zero-extend their answer to `u64`. Loads cross
//! runtime fragments, whereas [`PacketWindow`] addresses only the contiguous
//! window. A load stays in place even when its answer stays unread.
//!
//! A range predicate preserves the original start and end pointers. Unlike a
//! scalar slice length, this pair lets a machine establish a subsequent byte
//! read as safe. Failure is an ordinary `None`, so the caller chooses its own
//! result:
//!
//! ```no_run
//! use core::polyasm::PacketContext;
//! fn filter(header: &[u8; 14]) -> u32 {
//!     if u16::from_be_bytes([header[12], header[13]]) == 0x0800 { 1 } else { 2 }
//! }
//! fn polyasm_entry(packet: &PacketContext) -> u32 {
//!     let Some(header) = packet.window().prefix::<14>() else { return 1 };
//!     filter(header)
//! }
//! ```
//!
//! An immediate is part of the named operand type:
//!
//! ```no_run
//! use core::polyasm::packet::{packet_data_load16be_abs, PacketDataLoad16BeAbs, PacketContext};
//! fn ether_type(packet: &PacketContext) -> u64 {
//!     packet_data_load16be_abs(PacketDataLoad16BeAbs::<12> { context: packet })
//! }
//! ```
//!
//! A borrow stays tied to the invocation's lifetime:
//!
//! ```compile_fail
//! use core::polyasm::PacketContext;
//! fn escape(packet: &PacketContext) -> &'static [u8] {
//!     packet.window().prefix::<14>().unwrap()
//! }
//! ```
//!
//! Context fabrication is unavailable to safe Rust:
//!
//! ```compile_fail
//! use core::polyasm::PacketContext;
//! let packet = PacketContext {};
//! ```
//!
//! Each instruction takes its own opcode's operands alone:
//!
//! ```compile_fail
//! use core::polyasm::packet::{packet_data_start, PacketDataEnd, PacketContext};
//! fn mismatched(packet: &PacketContext) {
//!     packet_data_start(PacketDataEnd { context: packet });
//! }
//! ```

// The packet instructions are declared with every other PolyASM instruction
// in `super::intrinsics`; this module holds their operand types and names
// them here as well.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
pub use super::intrinsics::{
    packet_data_end, packet_data_load8_abs, packet_data_load8_ind, packet_data_load16be_abs,
    packet_data_load16be_ind, packet_data_load32be_abs, packet_data_load32be_ind,
    packet_data_range, packet_data_start, packet_mac_load8_abs, packet_mac_load8_ind,
    packet_mac_load16be_abs, packet_mac_load16be_ind, packet_mac_load32be_abs,
    packet_mac_load32be_ind, packet_network_load8_abs, packet_network_load8_ind,
    packet_network_load16be_abs, packet_network_load16be_ind, packet_network_load32be_abs,
    packet_network_load32be_ind,
};
use crate::fmt;
use crate::marker::PhantomData;

mod context {
    unsafe extern "C" {
        #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
        pub type PacketContext;
    }
}

/// An opaque packet capability borrowed from the executing machine.
///
/// The machine alone defines this type's layout; Rust code holds a borrowed
/// reference alone, with zero constructors, mutable accessors or
/// conversions from an integer or a Linux context structure. The machine lends
/// its reference to a guest entry function for one invocation. A context stays
/// on its invocation's thread, unowned by safe Rust.
///
/// # Runtime obligations
///
/// Before forming a reference, the machine establishes a live context
/// with a stable identity and correctly initialized packet storage. The
/// current contiguous window has one allocation and a non-null start,
/// including for an empty window. Its end is at or after its start, and its
/// byte length is at most `isize::MAX`. Both boundary instructions preserve
/// that allocation's provenance and the complete pointer width.
///
/// The context, window and contents remain valid and unchanged throughout
/// every shared borrow. An operation that changes packet storage or its
/// origins requires exclusive access to the context, after all borrowed
/// slices have expired. This also applies to external helpers and devices.
/// Packet fragments are separate allocations at times and stay outside the
/// contiguous window even when their logical offsets follow it.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_context"]
pub type PacketContext = context::PacketContext;

#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
impl fmt::Debug for PacketContext {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PacketContext")
    }
}

#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
impl PacketContext {
    /// Borrows the current contiguous window as its original pointer pair.
    ///
    /// Separate fragments remain accessible through dedicated packet loads.
    /// The window retains this borrow and carries this context's pointers
    /// alone. Its checked prefixes preserve pointer identity across calls.
    #[inline]
    #[must_use]
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub fn window(&self) -> PacketWindow<'_> {
        PacketWindow {
            start: packet_data_start(PacketDataStart { context: self }),
            end: packet_data_end(PacketDataEnd { context: self }),
            context: PhantomData,
        }
    }
}

/// One stable contiguous packet window, retaining its originating borrow.
///
/// The pointers are private: safe code keeps contexts apart, keeps each packet
/// boundary a pointer, and keeps separate fragments outside a window. Copies
/// preserve the same original pair, and every operation keeps the pointers it
/// checked a range against. The checked prefixes are the only slices a window
/// hands out.
#[derive(Clone, Copy, Debug)]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_window"]
pub struct PacketWindow<'packet> {
    start: *const u8,
    // The packet_data_range intrinsic reads this field by compiler identity.
    #[allow(dead_code)]
    end: *const u8,
    context: PhantomData<&'packet PacketContext>,
}

#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
impl<'packet> PacketWindow<'packet> {
    /// Borrows exactly `LENGTH` leading bytes if they are all contiguous.
    ///
    /// `None` leaves the invocation running and selects zero packet actions.
    /// Zero bytes succeeds even for an empty window. The length is an unsigned
    /// 32-bit instruction immediate, checked at compile time at its full width.
    /// The returned array retains the packet borrow even if this window value
    /// is moved or dropped, and an ordinary Rust function takes it as is.
    #[inline]
    #[must_use]
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub fn prefix<const LENGTH: usize>(self) -> Option<&'packet [u8; LENGTH]> {
        const { assert!(LENGTH <= u32::MAX as usize, "packet prefix exceeds the instruction length") };
        if packet_data_range(PacketDataRange::<LENGTH> { window: self }) {
            // SAFETY: the native range predicate proved exactly these bytes
            // in this original window. The context guarantees initialized,
            // stable storage with a non-null start, including for LENGTH=0.
            Some(unsafe { &*self.start.cast::<[u8; LENGTH]>() })
        } else {
            None
        }
    }
}

/// Named operands of `PacketDataRange`, with its unsigned constant length.
///
/// The original boundary pair is inside `window`; its private fields stay
/// closed to safe Rust. `LENGTH` fits in `u32`.
#[derive(Debug)]
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_data_range_operands"]
pub struct PacketDataRange<'packet, const LENGTH: usize> {
    /// The unchanged borrowed pointer pair that subsequent accesses use.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub window: PacketWindow<'packet>,
}

/// Named operands of `PacketDataLoad16BeAbs`.
///
/// `DISPLACEMENT` is a signed byte displacement from the opcode's fixed
/// origin. Carrying it in the type makes the instruction immediate a compile-
/// time value; a runtime displacement uses the corresponding `Ind` instruction.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_data_load16be_abs_operands"]
#[derive(Debug)]
pub struct PacketDataLoad16BeAbs<'packet, const DISPLACEMENT: i32> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
}

/// Named operands of `PacketDataLoad16BeInd`.
///
/// `offset` is the already-computed signed displacement from the opcode's
/// fixed origin. The instruction keeps that origin and leaves the displacement
/// unwrapped.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_data_load16be_ind_operands"]
#[derive(Debug)]
pub struct PacketDataLoad16BeInd<'packet> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
    /// Signed byte displacement from this opcode's fixed packet origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub offset: i64,
}

/// Named operands of `PacketDataLoad32BeAbs`.
///
/// `DISPLACEMENT` is a signed byte displacement from the opcode's fixed
/// origin. Carrying it in the type makes the instruction immediate a compile-
/// time value; a runtime displacement uses the corresponding `Ind` instruction.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_data_load32be_abs_operands"]
#[derive(Debug)]
pub struct PacketDataLoad32BeAbs<'packet, const DISPLACEMENT: i32> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
}

/// Named operands of `PacketDataLoad32BeInd`.
///
/// `offset` is the already-computed signed displacement from the opcode's
/// fixed origin. The instruction keeps that origin and leaves the displacement
/// unwrapped.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_data_load32be_ind_operands"]
#[derive(Debug)]
pub struct PacketDataLoad32BeInd<'packet> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
    /// Signed byte displacement from this opcode's fixed packet origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub offset: i64,
}

/// Named operands of `PacketDataLoad8Abs`.
///
/// `DISPLACEMENT` is a signed byte displacement from the opcode's fixed
/// origin. Carrying it in the type makes the instruction immediate a compile-
/// time value; a runtime displacement uses the corresponding `Ind` instruction.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_data_load8_abs_operands"]
#[derive(Debug)]
pub struct PacketDataLoad8Abs<'packet, const DISPLACEMENT: i32> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
}

/// Named operands of `PacketDataLoad8Ind`.
///
/// `offset` is the already-computed signed displacement from the opcode's
/// fixed origin. The instruction keeps that origin and leaves the displacement
/// unwrapped.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_data_load8_ind_operands"]
#[derive(Debug)]
pub struct PacketDataLoad8Ind<'packet> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
    /// Signed byte displacement from this opcode's fixed packet origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub offset: i64,
}

/// Named operands of `PacketMacLoad16BeAbs`.
///
/// `DISPLACEMENT` is a signed byte displacement from the opcode's fixed
/// origin. Carrying it in the type makes the instruction immediate a compile-
/// time value; a runtime displacement uses the corresponding `Ind` instruction.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_mac_load16be_abs_operands"]
#[derive(Debug)]
pub struct PacketMacLoad16BeAbs<'packet, const DISPLACEMENT: i32> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
}

/// Named operands of `PacketMacLoad16BeInd`.
///
/// `offset` is the already-computed signed displacement from the opcode's
/// fixed origin. The instruction keeps that origin and leaves the displacement
/// unwrapped.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_mac_load16be_ind_operands"]
#[derive(Debug)]
pub struct PacketMacLoad16BeInd<'packet> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
    /// Signed byte displacement from this opcode's fixed packet origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub offset: i64,
}

/// Named operands of `PacketMacLoad32BeAbs`.
///
/// `DISPLACEMENT` is a signed byte displacement from the opcode's fixed
/// origin. Carrying it in the type makes the instruction immediate a compile-
/// time value; a runtime displacement uses the corresponding `Ind` instruction.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_mac_load32be_abs_operands"]
#[derive(Debug)]
pub struct PacketMacLoad32BeAbs<'packet, const DISPLACEMENT: i32> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
}

/// Named operands of `PacketMacLoad32BeInd`.
///
/// `offset` is the already-computed signed displacement from the opcode's
/// fixed origin. The instruction keeps that origin and leaves the displacement
/// unwrapped.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_mac_load32be_ind_operands"]
#[derive(Debug)]
pub struct PacketMacLoad32BeInd<'packet> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
    /// Signed byte displacement from this opcode's fixed packet origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub offset: i64,
}

/// Named operands of `PacketMacLoad8Abs`.
///
/// `DISPLACEMENT` is a signed byte displacement from the opcode's fixed
/// origin. Carrying it in the type makes the instruction immediate a compile-
/// time value; a runtime displacement uses the corresponding `Ind` instruction.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_mac_load8_abs_operands"]
#[derive(Debug)]
pub struct PacketMacLoad8Abs<'packet, const DISPLACEMENT: i32> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
}

/// Named operands of `PacketMacLoad8Ind`.
///
/// `offset` is the already-computed signed displacement from the opcode's
/// fixed origin. The instruction keeps that origin and leaves the displacement
/// unwrapped.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_mac_load8_ind_operands"]
#[derive(Debug)]
pub struct PacketMacLoad8Ind<'packet> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
    /// Signed byte displacement from this opcode's fixed packet origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub offset: i64,
}

/// Named operands of `PacketNetworkLoad16BeAbs`.
///
/// `DISPLACEMENT` is a signed byte displacement from the opcode's fixed
/// origin. Carrying it in the type makes the instruction immediate a compile-
/// time value; a runtime displacement uses the corresponding `Ind` instruction.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_network_load16be_abs_operands"]
#[derive(Debug)]
pub struct PacketNetworkLoad16BeAbs<'packet, const DISPLACEMENT: i32> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
}

/// Named operands of `PacketNetworkLoad16BeInd`.
///
/// `offset` is the already-computed signed displacement from the opcode's
/// fixed origin. The instruction keeps that origin and leaves the displacement
/// unwrapped.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_network_load16be_ind_operands"]
#[derive(Debug)]
pub struct PacketNetworkLoad16BeInd<'packet> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
    /// Signed byte displacement from this opcode's fixed packet origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub offset: i64,
}

/// Named operands of `PacketNetworkLoad32BeAbs`.
///
/// `DISPLACEMENT` is a signed byte displacement from the opcode's fixed
/// origin. Carrying it in the type makes the instruction immediate a compile-
/// time value; a runtime displacement uses the corresponding `Ind` instruction.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_network_load32be_abs_operands"]
#[derive(Debug)]
pub struct PacketNetworkLoad32BeAbs<'packet, const DISPLACEMENT: i32> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
}

/// Named operands of `PacketNetworkLoad32BeInd`.
///
/// `offset` is the already-computed signed displacement from the opcode's
/// fixed origin. The instruction keeps that origin and leaves the displacement
/// unwrapped.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_network_load32be_ind_operands"]
#[derive(Debug)]
pub struct PacketNetworkLoad32BeInd<'packet> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
    /// Signed byte displacement from this opcode's fixed packet origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub offset: i64,
}

/// Named operands of `PacketNetworkLoad8Abs`.
///
/// `DISPLACEMENT` is a signed byte displacement from the opcode's fixed
/// origin. Carrying it in the type makes the instruction immediate a compile-
/// time value; a runtime displacement uses the corresponding `Ind` instruction.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_network_load8_abs_operands"]
#[derive(Debug)]
pub struct PacketNetworkLoad8Abs<'packet, const DISPLACEMENT: i32> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
}

/// Named operands of `PacketNetworkLoad8Ind`.
///
/// `offset` is the already-computed signed displacement from the opcode's
/// fixed origin. The instruction keeps that origin and leaves the displacement
/// unwrapped.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_network_load8_ind_operands"]
#[derive(Debug)]
pub struct PacketNetworkLoad8Ind<'packet> {
    /// The runtime-lent context defining packet storage and this origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
    /// Signed byte displacement from this opcode's fixed packet origin.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub offset: i64,
}

/// Named operands of `PacketDataStart`.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_data_start_operands"]
#[derive(Debug)]
pub struct PacketDataStart<'packet> {
    /// The runtime-lent context whose contiguous window is addressed.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
}

/// Named operands of `PacketDataEnd`.
#[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
#[rustc_diagnostic_item = "polyasm_packet_data_end_operands"]
#[derive(Debug)]
pub struct PacketDataEnd<'packet> {
    /// The runtime-lent context whose contiguous window is addressed.
    #[stable(feature = "polyasm_packet", since = "CURRENT_RUSTC_VERSION")]
    pub context: &'packet PacketContext,
}
