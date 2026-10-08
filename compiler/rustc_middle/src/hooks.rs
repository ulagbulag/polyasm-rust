//! "Hooks" let you write `tcx` methods in downstream crates and call them in this crate, reducing
//! the amount of code that needs to be in this crate (which is already very big). This is somewhat
//! similar to queries, but queries come with a lot of machinery for caching and incremental
//! compilation, whereas hooks are just plain function pointers without any of the query magic.

use std::io;

use rustc_hir::def_id::{DefId, DefPathHash};
use rustc_index::bit_set::DenseBitSet;
use rustc_session::StableCrateId;
use rustc_span::def_id::{CrateNum, LocalDefId};
use rustc_span::{ExpnHash, ExpnId, bug};

use crate::mir;
use crate::query::on_disk_cache::CacheEncoder;
use crate::ty::{self, Ty, TyCtxt};

macro_rules! declare_hooks {
    (@default $name:ident ($($arg:ident),*) = $default:expr) => { $default };
    (@default $name:ident ($($arg:ident),*)) => {
        |_, $($arg,)*| default_hook(stringify!($name))
    };

    (
        $(
            $(#[$attr:meta])*
            hook $name:ident(
                $( $arg:ident: $K:ty ),* $(,)?
            ) -> $V:ty $(= $default:expr)?;
        )*
    ) => {

        impl<'tcx> TyCtxt<'tcx> {
            $(
            $(#[$attr])*
            #[inline(always)]
            pub fn $name(self, $($arg: $K,)*) -> $V
            {
                (self.hooks.$name)(self, $($arg,)*)
            }
            )*
        }

        pub struct Providers {
            $(pub $name: for<'tcx> fn(
                TyCtxt<'tcx>,
                $($arg: $K,)*
            ) -> $V,)*
        }

        impl Default for Providers {
            fn default() -> Self {
                #[allow(unused)]
                Providers {
                    $($name:
                        declare_hooks!(@default $name ($($arg),*) $(= $default)?)),*
                }
            }
        }

        impl Copy for Providers {}
        impl Clone for Providers {
            fn clone(&self) -> Self { *self }
        }
    };
}

declare_hooks! {
    hook prepare_external_shim(shim: ty::ShimKind<'tcx>, body: &mut mir::Body<'tcx>) -> () = |_, _, _| {};
    hook retain_external_goal(predicate: ty::Predicate<'tcx>, span: rustc_span::Span, body: LocalDefId, seen: &mut rustc_data_structures::fx::FxHashSet<(ty::Predicate<'tcx>, rustc_span::Span, LocalDefId)>) -> bool = |_, _, _, _, _| true;
    hook external_trait_decision(trait_ref: ty::TraitRef<'tcx>) -> Option<Result<bool, ()>> = |_, _| None;
    hook is_external_builtin_trait(def_id: DefId) -> bool = |_, _| false;
    hook external_trait_rejection(trait_ref: ty::TraitRef<'tcx>, span: rustc_span::Span) -> Option<(rustc_span::Span, String)> = |_, _, _| None;
    hook check_external_instance(instance: ty::Instance<'tcx>) -> () = |_, _| {};
    hook external_inline_policy(caller: DefId, callee: DefId, forced: bool) -> Result<(), &'static str> = |_, _, _, _| Ok(());
    hook external_mono_roots() -> Vec<crate::mono::MonoItem<'tcx>> = |_| Vec::new();
    hook external_instance_uses(instance: ty::Instance<'tcx>, span: rustc_span::Span) -> Vec<ty::Instance<'tcx>> = |_, _, _| Vec::new();
    hook allow_instance_reification(instance: ty::Instance<'tcx>, span: rustc_span::Span) -> bool = |_, _, _| true;
    hook external_mono_linkage(item: crate::mono::MonoItem<'tcx>) -> Option<(rustc_attr_ir::Linkage, crate::mono::Visibility)> = |_, _| None;
    hook adjust_external_abi(cx: &crate::ty::layout::LayoutCx<'tcx>, abi: &mut rustc_target::callconv::FnAbi<'tcx, Ty<'tcx>>, convention: rustc_abi::ExternAbi) -> bool = |_, _, _, _| false;
    hook external_trait_impl(impl_id: LocalDefId, trait_id: DefId) -> Result<(), rustc_span::ErrorGuaranteed> = |_, _, _| Ok(());
    hook skip_mono_item_body(instance: ty::Instance<'tcx>) -> bool = |_, _| false;
    hook external_intrinsic_signature(def_id: LocalDefId) -> Option<(usize, usize, usize, ty::PolyFnSig<'tcx>)> = |_, _| None;
    /// Tries to destructure an `mir::Const` ADT or array into its variant index
    /// and its field values. This should only be used for pretty printing.
    hook try_destructure_mir_constant_for_user_output(val: mir::ConstValue, ty: Ty<'tcx>) -> Option<mir::DestructuredConstant<'tcx>>;

    /// Getting a &core::panic::Location referring to a span.
    hook const_caller_location(file: rustc_span::Symbol, line: u32, col: u32) -> mir::ConstValue;

    /// Imports all `SourceFile`s from the given crate into the current session.
    /// This normally happens automatically when we decode a `Span` from
    /// that crate's metadata - however, the incr comp cache needs
    /// to trigger this manually when decoding a foreign `Span`
    hook import_source_files(key: CrateNum) -> ();

    hook expn_hash_to_expn_id(
        cnum: CrateNum,
        index_guess: u32,
        hash: ExpnHash
    ) -> ExpnId;

    /// Converts a `DefPathHash` to its corresponding `DefId` in the current compilation
    /// session, if it still exists. This is used during incremental compilation to
    /// turn a deserialized `DefPathHash` into its current `DefId`.
    /// Will fetch a DefId from a DefPathHash for a foreign crate.
    hook def_path_hash_to_def_id_extern(hash: DefPathHash, stable_crate_id: StableCrateId) -> Option<DefId>;

    /// Returns `true` if we should codegen an instance in the local crate, or returns `false` if we
    /// can just link to the upstream crate and therefore don't need a mono item.
    ///
    /// Note: this hook isn't called within `rustc_middle` but #127779 suggests it's a hook instead
    /// of a normal function because external tools might want to override it.
    hook should_codegen_locally(instance: crate::ty::Instance<'tcx>) -> bool;

    hook alloc_self_profile_query_strings() -> ();

    /// Saves and writes the DepGraph to the file system.
    ///
    /// This function saves both the dep-graph and the query result cache,
    /// and drops the result cache.
    ///
    /// This function should only run after all queries have completed.
    /// Trying to execute a query afterwards would attempt to read the result cache we just dropped.
    hook save_dep_graph() -> ();

    hook verify_query_key_hashes() -> ();

    /// Ensure the given scalar is valid for the given type.
    /// This checks non-recursive runtime validity.
    hook validate_scalar_in_layout(scalar: crate::ty::ScalarInt, ty: Ty<'tcx>) -> bool;

    /// **Do not call this directly; call the `mir_built` query instead.**
    ///
    /// Creates the MIR for a given `DefId`, including unreachable code.
    hook build_mir_inner_impl(def: LocalDefId) -> mir::Body<'tcx>;

    /// Serializes all eligible query return values into the on-disk cache.
    hook encode_query_values(encoder: &mut CacheEncoder<'tcx>) -> ();

    /// Identifies landing pads that don't do anything, allowing some post-monomorphization
    /// simplifications during codegen.
    hook find_noop_landing_pads_for_instance(
        body: &mir::Body<'tcx>,
        instance: ty::Instance<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> DenseBitSet<mir::BasicBlock>;

    /// Configures extra data to be included when pretty printing MIR
    hook mir_pretty_extra_data(
        body: &mir::Body<'tcx>,
    ) -> Option<Box<dyn Fn(mir::PassWhere, &mut dyn io::Write) -> io::Result<()>>>;

}

#[cold]
fn default_hook(name: &str) -> ! {
    bug!("`tcx.{name}` cannot be called as `{name}` was never assigned to a provider function")
}
