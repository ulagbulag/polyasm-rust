//! Packet failure leaves the entire guest invocation, including its callers.
//!
//! LLVM treats that departure like an unwind: memory visible to a caller
//! remains observable even when the callee stays away from its return. This
//! emits zero Rust unwinding or cleanup code. The distinct packet opcode
//! performs the departure. In particular, an opaque memory effect alone leaves
//! a store through an independent `noalias` reference unprotected before that
//! departure. Rust cleanup edges represent a different departure: a packet
//! exit skips those cleanups too. Such graphs therefore require the target's
//! default `panic=abort` strategy until the IR has a distinct guest-exit
//! effect.

use rustc_data_structures::fx::FxIndexSet;
use rustc_middle::mir::TerminatorKind;
use rustc_middle::ty::{self, EarlyBinder, Instance};
use rustc_span::sym;

use crate::context::CodegenCx;

impl<'ll, 'tcx> CodegenCx<'ll, 'tcx> {
    /// Whether this callee leaves the invocation through a packet opcode at
    /// times. Unknown calls remain conservative; closed, packet-free graphs
    /// retain their ordinary Rust `nounwind` attributes. The instance cache
    /// crosses function boundaries within a codegen unit, and MIR crosses crate
    /// and codegen-unit boundaries with LLVM LTO left optional.
    pub(crate) fn may_exit_packet(&self, instance: Option<Instance<'tcx>>) -> bool {
        if !self.tcx.sess.is_polyasm_target() {
            return false;
        }
        let effect = instance.is_none_or(|root| self.packet_exit_graph(root));
        if effect && self.tcx.sess.panic_strategy().unwinds() {
            self.tcx.sess.dcx().fatal(
                "a PolyASM packet-capable call graph requires `-Cpanic=abort`; \
                 Rust cleanup edges cannot represent whole-invocation packet exit",
            );
        }
        effect
    }

    fn packet_exit_graph(&self, root: Instance<'tcx>) -> bool {
        if let Some(&effect) = self.packet_exits.borrow().get(&root) {
            return effect;
        }

        let mut pending = vec![root];
        let mut visited = FxIndexSet::default();
        let effect = 'graph: loop {
            let Some(mut instance) = pending.pop() else {
                break false;
            };
            if !visited.insert(instance) {
                continue;
            }
            if let Some(&effect) = self.packet_exits.borrow().get(&instance) {
                if effect {
                    break true;
                }
                continue;
            }
            if let Some(intrinsic) = self.tcx.intrinsic(instance.def_id()) {
                if matches!(
                    intrinsic.name,
                    sym::packet_data_load8_abs
                        | sym::packet_data_load8_ind
                        | sym::packet_data_load16be_abs
                        | sym::packet_data_load16be_ind
                        | sym::packet_data_load32be_abs
                        | sym::packet_data_load32be_ind
                        | sym::packet_mac_load8_abs
                        | sym::packet_mac_load8_ind
                        | sym::packet_mac_load16be_abs
                        | sym::packet_mac_load16be_ind
                        | sym::packet_mac_load32be_abs
                        | sym::packet_mac_load32be_ind
                        | sym::packet_network_load8_abs
                        | sym::packet_network_load8_ind
                        | sym::packet_network_load16be_abs
                        | sym::packet_network_load16be_ind
                        | sym::packet_network_load32be_abs
                        | sym::packet_network_load32be_ind
                        | sym::packet_data_start
                        | sym::packet_data_end
                        | sym::autodiff
                        | sym::catch_unwind
                        | sym::const_eval_select
                ) {
                    break true;
                }
                // Every other PolyASM instruction answers or faults where it
                // stands, and its portable body is the row's record and stays
                // inside the invocation.
                if intrinsic.must_be_overridden
                    || rustc_codegen_ssa::polyasm::instruction(self.tcx, instance.def_id())
                {
                    continue;
                }
                // A fallback has the ordinary item's MIR. The intrinsic
                // instance itself carries zero MIR to query.
                instance = Instance {
                    def: ty::InstanceKind::Item(instance.def_id()),
                    args: instance.args,
                };
            }
            match instance.def {
                ty::InstanceKind::Virtual(..) | ty::InstanceKind::LlvmIntrinsic(..) => {
                    break true;
                }
                ty::InstanceKind::Item(definition) if !self.tcx.is_mir_available(definition) => {
                    break true;
                }
                _ => {}
            }
            let body = self.tcx.instance_mir(instance.def);
            for block in body.basic_blocks.iter() {
                match &block.terminator().kind {
                    TerminatorKind::Call { func, .. } | TerminatorKind::TailCall { func, .. } => {
                        let function = instance.instantiate_mir_and_normalize_erasing_regions(
                            self.tcx,
                            ty::TypingEnv::fully_monomorphized(),
                            EarlyBinder::bind(self.tcx, func.ty(&body.local_decls, self.tcx)),
                        );
                        let ty::FnDef(definition, arguments) = *function.kind() else {
                            break 'graph true;
                        };
                        let Some(arguments) = arguments.no_bound_vars() else {
                            break 'graph true;
                        };
                        let Ok(Some(callee)) = Instance::try_resolve(
                            self.tcx,
                            ty::TypingEnv::fully_monomorphized(),
                            definition,
                            arguments,
                        ) else {
                            break 'graph true;
                        };
                        pending.push(callee);
                    }
                    TerminatorKind::Drop { place, .. } => {
                        let dropped = instance.instantiate_mir_and_normalize_erasing_regions(
                            self.tcx,
                            ty::TypingEnv::fully_monomorphized(),
                            EarlyBinder::bind(self.tcx, place.ty(&body.local_decls, self.tcx).ty),
                        );
                        pending.push(Instance::resolve_drop_glue(self.tcx, dropped));
                    }
                    TerminatorKind::InlineAsm { .. } => break 'graph true,
                    _ => {}
                }
            }
        };
        let mut cache = self.packet_exits.borrow_mut();
        cache.insert(root, effect);
        if !effect {
            // A closed graph whose nodes all stay inside marks every visited node safe.
            // A positive result marks only its root: sibling nodes reach the
            // exit or stay away from it, and recursive components keep a
            // provisional negative result out of the cache.
            cache.extend(visited.into_iter().map(|instance| (instance, false)));
        }
        effect
    }
}
