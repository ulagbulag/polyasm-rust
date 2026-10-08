//! Rechecks source-level PolyASM markers after monomorphization.
//!
//! Trait selection uses a conservative HIR reading to avoid querying MIR while
//! the callable owner is under type checking. Every concrete code-generation
//! instance carrying a positive, negative, or exact-clock marker is checked
//! here against normalized MIR, including predicates inherited from a parent
//! item and markers implied by a supertrait. This makes an ordinary
//! `F: Always<..>` bound as strict as an explicit compiler marker.

use rustc_attr_ir::lang_items::LangItem;
use rustc_middle::mir::{Location, Operand, TerminatorKind};
use rustc_middle::mono::resolve_polyasm_callable;
use rustc_middle::ty::consts::ConstExt;
use rustc_middle::ty::{
    self, EarlyBinder, Instance, InstanceKind, Ty, TyCtxt, TypeVisitableExt, Unnormalized,
};
use rustc_span::Span;
use rustc_trait_selection::traits::{
    elaborate, explain_normalized_polyasm_rejection,
    explain_normalized_polyasm_statement_rejection, normalized_polyasm_property_holds,
    normalized_polyasm_statement_property_holds, normalized_polyasm_static_schedule,
    polyasm_statement_selection,
};

#[derive(Copy, Clone)]
enum WitnessKind {
    NegativeProperty,
    PositiveProperty,
    StaticClock,
}

pub(super) fn check_always_predicates<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    if !tcx.sess.is_polyasm_target() {
        return;
    }

    check_statement_markers(tcx, instance);
    if !matches!(instance.def, InstanceKind::Item(_)) {
        return;
    }

    let typing_env = ty::TypingEnv::fully_monomorphized();
    if (tcx.is_diagnostic_item(rustc_span::sym::polyasm_require_always, instance.def_id())
        || tcx.is_diagnostic_item(rustc_span::sym::polyasm_require_not_always, instance.def_id()))
        && instance
            .args
            .get(2)
            .and_then(|argument| argument.as_type())
            .is_some_and(|callable| callable.needs_drop(tcx, typing_env))
    {
        reject(
            tcx,
            tcx.def_span(instance.def_id()),
            "PolyASM compiler witness cannot erase a callable environment that requires drop",
        );
    }
    let predicates = tcx
        .clauses_of(instance.def_id())
        .instantiate(tcx, instance.args)
        .into_iter()
        .map(|(predicate, span)| (predicate.skip_norm_wip(), span));
    for (predicate, span) in elaborate(tcx, predicates) {
        let Some(trait_predicate) = predicate.as_trait_clause() else {
            continue;
        };
        let witness = match tcx.as_lang_item(trait_predicate.def_id()) {
            Some(LangItem::PolyasmAlways | LangItem::PolyasmCompilerCertificate) => {
                WitnessKind::PositiveProperty
            }
            Some(LangItem::PolyasmCompilerCounterexample) => WitnessKind::NegativeProperty,
            Some(LangItem::PolyasmCompilerStaticClockCertificate) => WitnessKind::StaticClock,
            _ => continue,
        };
        if trait_predicate.polarity() != ty::ClausePolarity::Positive {
            continue;
        }

        let Ok(predicate) =
            tcx.try_normalize_erasing_regions(typing_env, Unnormalized::new_wip(predicate))
        else {
            reject(
                tcx,
                span,
                "this `Always` bound could not be normalized for its concrete PolyASM instance",
            );
            continue;
        };
        let Some(trait_predicate) =
            predicate.as_trait_clause().and_then(|predicate| predicate.no_bound_vars())
        else {
            reject(
                tcx,
                span,
                "this `Always` bound is not fully monomorphized for PolyASM code generation",
            );
            continue;
        };
        let trait_ref = trait_predicate.trait_ref;
        if trait_ref.args.has_infer() || trait_ref.args.has_non_region_param() {
            reject(
                tcx,
                span,
                "this `Always` bound still contains unresolved generic arguments during PolyASM code generation",
            );
            continue;
        }

        if !matches!(trait_ref.self_ty().kind(), ty::FnDef(..) | ty::Closure(..)) {
            // The defining crate also implements compiler marker traits for
            // sealed checker tokens. Only concrete callable
            // bodies are derived by rustc and therefore require a MIR recheck.
            continue;
        }

        let Some(callable) = resolve_polyasm_callable(tcx, trait_ref.self_ty(), span) else {
            reject(
                tcx,
                span,
                "the concrete self type of this `Always` bound is not a resolvable function item or closure",
            );
            continue;
        };
        match witness {
            WitnessKind::PositiveProperty => check_property(
                tcx,
                callable,
                trait_ref.args.type_at(1),
                trait_ref.args.type_at(2),
                true,
                span,
            ),
            WitnessKind::NegativeProperty => check_property(
                tcx,
                callable,
                trait_ref.args.type_at(1),
                trait_ref.args.type_at(2),
                false,
                span,
            ),
            WitnessKind::StaticClock => {
                check_static_clock(tcx, callable, trait_ref, span);
            }
        }
    }
}

fn check_statement_markers<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    let body = tcx.instance_mir(instance.def);
    for (block, data) in body.basic_blocks.iter_enumerated() {
        let terminator = data.terminator();
        let marker = Location { block, statement_index: data.statements.len() };
        let TerminatorKind::Call { func, args, .. } = &terminator.kind else {
            continue;
        };
        let Some((def_id, generic_args)) = func.const_fn_def() else { continue };
        if !tcx.is_diagnostic_item(rustc_span::sym::polyasm_require_statement, def_id) {
            continue;
        }
        let [argument] = &args[..] else {
            reject(
                tcx,
                terminator.source_info.span,
                "PolyASM statement warrant marker has an invalid ABI",
            );
            continue;
        };
        let scope_span = argument.span;
        if matches!(argument.node, Operand::RuntimeChecks(_)) || scope_span.is_dummy() {
            reject(
                tcx,
                terminator.source_info.span,
                "PolyASM statement warrant lost the exact argument source scope before monomorphization",
            );
            continue;
        }
        let Some(selection) = polyasm_statement_selection(tcx, instance, marker) else {
            reject(
                tcx,
                terminator.source_info.span,
                "PolyASM statement warrant lost its exact normalized-MIR dependency slice",
            );
            continue;
        };
        let generic_args = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            ty::TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, generic_args),
        );
        if generic_args.has_non_region_param() || generic_args.has_infer() {
            reject(
                tcx,
                terminator.source_info.span,
                "PolyASM statement warrant still has unresolved property or architecture arguments after monomorphization",
            );
            continue;
        }
        let property = generic_args.type_at(0);
        let architecture = generic_args.type_at(1);
        match normalized_polyasm_statement_property_holds(
            tcx,
            instance,
            &selection,
            property,
            architecture,
        ) {
            Some(true) => {}
            Some(false) | None => {
                let mut diagnostic = tcx.dcx().struct_span_err(
                    terminator.source_info.span,
                    format!(
                        "PolyASM cannot prove `Always<{property}, {architecture}>` for this statement expression"
                    ),
                );
                diagnostic.span_label(
                    scope_span,
                    "this exact expression does not satisfy the requested PolyASM property",
                );
                if let Some((failure_span, reason)) = explain_normalized_polyasm_statement_rejection(
                    tcx,
                    instance,
                    &selection,
                    property,
                    architecture,
                ) {
                    diagnostic.span_note(failure_span, reason);
                } else {
                    diagnostic.note("the normalized statement scope did not provide conclusive compiler witness");
                }
                diagnostic.emit();
            }
        }
    }
}

fn check_property<'tcx>(
    tcx: TyCtxt<'tcx>,
    callable: Instance<'tcx>,
    property: Ty<'tcx>,
    architecture: Ty<'tcx>,
    expected: bool,
    span: Span,
) {
    match normalized_polyasm_property_holds(tcx, callable, property, architecture) {
        Some(actual) if actual == expected => {}
        actual => {
            let witness = if expected { "Always" } else { "NotAlways" };
            let callable_name = tcx.def_path_str(callable.def_id());
            let mut diagnostic = tcx.dcx().struct_span_err(
                span,
                format!(
                    "PolyASM cannot prove `{witness}<{property}, {architecture}>` for normalized callable `{callable_name}`"
                ),
            );
            diagnostic.span_label(
                span,
                if expected {
                    "this positive compiler warrant is not satisfied after monomorphization"
                } else {
                    "this negative compiler warrant is not satisfied after monomorphization"
                },
            );
            if let Some((failure_span, reason)) =
                explain_normalized_polyasm_rejection(tcx, callable, property, architecture)
            {
                diagnostic.span_note(failure_span, reason);
            } else if actual == Some(true) && !expected {
                diagnostic.span_note(
                    tcx.def_span(callable.def_id()),
                    "the normalized callable body satisfies the requested property, so negative witness would be unsound",
                );
            } else {
                diagnostic.note(
                    "the normalized callable body did not provide conclusive compiler witness",
                );
            }
            diagnostic.emit();
        }
    }
}

fn check_static_clock<'tcx>(
    tcx: TyCtxt<'tcx>,
    callable: Instance<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    span: Span,
) {
    let Some(cycles) = u64_const(tcx, trait_ref.args.const_at(2)) else {
        reject(
            tcx,
            span,
            "the PolyASM static-clock cycle count is not a concrete `u64` after monomorphization",
        );
        return;
    };
    let architecture = trait_ref.args.type_at(1);
    let actual = normalized_polyasm_static_schedule(tcx, callable, architecture);
    if actual != Some(cycles) {
        let callable_name = tcx.def_path_str(callable.def_id());
        let mut diagnostic = tcx.dcx().struct_span_err(
            span,
            format!(
                "PolyASM cannot prove `Always<StaticClock, {architecture}>` with exactly `{cycles}` cycles for normalized callable `{callable_name}`"
            ),
        );
        diagnostic.span_label(
            span,
            "this exact board-specific static-clock warrant does not match the normalized body",
        );
        match actual {
            Some(actual) => diagnostic.span_note(
                tcx.def_span(callable.def_id()),
                format!(
                    "the normalized body has an exact schedule of `{actual}` cycles on `{architecture}`"
                ),
            ),
            None => diagnostic.note(
                "the normalized callable body did not provide one exact static schedule for this board",
            ),
        };
        diagnostic.emit();
    }
}

fn u64_const<'tcx>(tcx: TyCtxt<'tcx>, constant: ty::Const<'tcx>) -> Option<u64> {
    let value = constant.try_to_value()?;
    if value.ty != tcx.types.u64 {
        return None;
    }
    value.try_to_bits(tcx, ty::TypingEnv::fully_monomorphized())?.try_into().ok()
}

fn reject(tcx: TyCtxt<'_>, span: Span, message: &'static str) {
    tcx.dcx().span_err(span, message);
}
