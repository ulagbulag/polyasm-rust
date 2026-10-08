//! Various optimizations specific to cg_clif

use std::sync::OnceLock;

pub(crate) mod peephole;
pub(crate) mod slot_promote;

/// How many cells a function carries in values at one and the same place.
///
/// The number is measured. Against the standing image on
/// `malloc-0-uninit-1000-x`, three alternating pairs answer 11.402 / 11.968 /
/// 11.393 µs unpromoted and 7.8754 / 7.8833 / 7.8761 µs at this bound, and the
/// two sets of bands stay apart. Unbounded promotion of the same tree exceeds
/// what the register allocator seats: it runs out of registers and reaches for
/// the frame again, and it measures at the level of leaving the cells alone.
const LIVE_CELLS: usize = 16;

/// How `staging_mem2reg` is asked to promote the stack slot cells this backend
/// hands it.
///
/// The pass itself belongs to Cranelift and stays apart from `rustc`; what
/// is decided here is only how far this backend lets it go.
///
/// The first lever is a bound on how many cells a function carries in values
/// at once, which is the count that decides whether the register allocator has
/// room: what exhausts an allocator is how many values want a register at one
/// and the same place. `CG_CLIF_MEM2REG_LIVE` names that bound when it reads as
/// a count, and [`LIVE_CELLS`] is the bound this backend measured otherwise.
///
/// The second is what happens to a cell some call has live across it. A frame
/// cell is free across a call, because the callee stays away from it; a value
/// costs whatever it takes to be somewhere the callee leaves alone, which on
/// this backend is a write to the register file and a read back out of it on
/// either side of the call. `CG_CLIF_MEM2REG_NO_CALL_CELLS` leaves those cells
/// in the frame and promotes the rest of the function as before — measured at
/// 11.387 µs beside the 7.8833 µs the same tree answers with those cells
/// promoted, so it is a lever and stays off by default.
pub(crate) fn mem2reg_options() -> staging_mem2reg::Options {
    static OPTIONS: OnceLock<staging_mem2reg::Options> = OnceLock::new();
    OPTIONS
        .get_or_init(|| staging_mem2reg::Options {
            live_cells_limit: Some(
                std::env::var("CG_CLIF_MEM2REG_LIVE")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(LIVE_CELLS),
            ),
            across_calls: match std::env::var_os("CG_CLIF_MEM2REG_NO_CALL_CELLS") {
                Some(_) => staging_mem2reg::AcrossCalls::Decline,
                None => staging_mem2reg::AcrossCalls::Promote,
            },
        })
        .clone()
}
