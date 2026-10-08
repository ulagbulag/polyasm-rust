//! The crate image the interchange frontend answers and crate-visible
//! interfaces.

use std::collections::BTreeMap;

use polyasm::object::InterchangeFragment;
pub(crate) use polyasm::object::{CallableKind, FunctionSignature};
use rustc_hir::def::DefKind;

/// Reads the source callable identity of one monomorphized definition.
pub(crate) const fn callable_kind(kind: DefKind) -> CallableKind {
    match kind {
        DefKind::Fn | DefKind::AssocFn | DefKind::Ctor(..) => CallableKind::Function,
        DefKind::Closure | DefKind::SyntheticCoroutineBody => CallableKind::Closure,
        _ => CallableKind::Unknown,
    }
}

/// One crate's object records, as `polytime-frontend-cranelift` lowered them,
/// and the native relocation offset each MIR direct call owns, keyed by
/// caller, callee and MIR block.
pub(crate) struct PortableImage {
    pub(super) fragment: InterchangeFragment,
    pub(super) calls: BTreeMap<(String, String, u32), u32>,
}

pub(crate) struct CompiledImages {
    pub(crate) allocator: Option<PortableImage>,
    pub(crate) regular: PortableImage,
}

impl PortableImage {
    /// Returns the unique native relocation owned by one MIR direct call.
    /// Ambiguous or eliminated calls answer `None`, since they lack an executable address.
    pub(crate) fn direct_call_offset(
        &self,
        caller: &str,
        callee: &str,
        source_block: u32,
    ) -> Option<u32> {
        self.calls.get(&(caller.to_owned(), callee.to_owned(), source_block)).copied()
    }
}
