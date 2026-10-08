// FIXME(#142296): this hack is because there is no reliable way (yet) to determine whether a given
// target supports std. In the long-term, we should try to implement a way to *reliably* determine
// target (std) metadata.
//
// NOTE: this is pulled out to `build_helpers` to share this hack between `bootstrap` and
// `compiletest`.

/// The built-in PolyASM target tuples, one architecture at three pointer
/// widths.
///
/// Every PolyASM rule the build system carries — zero native C ABI, permanently
/// `no_std`, linked by Cranelift — is a rule about the whole architecture. A site
/// that named a single tuple would silently exempt the other two, so the tuples
/// are named once, here, where both `bootstrap` and `compiletest` read them.
///
/// `rustc_target::spec::POLYASM_TARGET_TUPLES` is the same list on the
/// compiler side. The build system and the compiler it builds stay unlinked, so
/// the two exist separately and change together.
pub const POLYASM_TARGET_TUPLES: &[&str] =
    &["polyasm-unknown-unknown", "polyasm32-unknown-unknown", "polyasm64-unknown-unknown"];

/// Whether `target_tuple` is one of the built-in PolyASM tuples.
pub fn is_polyasm_target(target_tuple: &str) -> bool {
    POLYASM_TARGET_TUPLES.contains(&target_tuple)
}

pub fn target_supports_std(target_tuple: &str) -> bool {
    !is_polyasm_target(target_tuple)
        && !(target_tuple.contains("-none")
            || target_tuple.contains("nvptx")
            || target_tuple.contains("switch"))
}
