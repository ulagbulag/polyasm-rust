//! Concrete field layouts used by the HIR and normalized-MIR readings.

use rustc_data_structures::fx::FxHashSet;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PointerLayout {
    Addressless,
    ContainsPointer,
    Unresolved,
}

impl PointerLayout {
    fn with(self, field: Self) -> Self {
        match (self, field) {
            (Self::ContainsPointer, _) | (_, Self::ContainsPointer) => Self::ContainsPointer,
            (Self::Unresolved, _) | (_, Self::Unresolved) => Self::Unresolved,
            (Self::Addressless, Self::Addressless) => Self::Addressless,
        }
    }
}

pub(super) struct TypeInspection<'tcx> {
    pub(super) tcx: TyCtxt<'tcx>,
    pub(super) ty: Ty<'tcx>,
}

impl<'tcx> TypeInspection<'tcx> {
    pub(super) fn pointer_layout(self) -> PointerLayout {
        FieldInspection { tcx: self.tcx, visiting: FxHashSet::default() }.pointer_layout(self.ty)
    }

    pub(super) fn is_drop_free(self) -> bool {
        FieldInspection { tcx: self.tcx, visiting: FxHashSet::default() }.is_drop_free(self.ty)
    }
}

struct FieldInspection<'tcx> {
    tcx: TyCtxt<'tcx>,
    visiting: FxHashSet<Ty<'tcx>>,
}

impl<'tcx> FieldInspection<'tcx> {
    fn pointer_layout(&mut self, ty: Ty<'tcx>) -> PointerLayout {
        match ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Never
            | ty::FnDef(..) => PointerLayout::Addressless,
            ty::Ref(..) | ty::RawPtr(..) | ty::FnPtr(..) => PointerLayout::ContainsPointer,
            ty::Tuple(fields) => fields.iter().fold(PointerLayout::Addressless, |layout, field| {
                layout.with(self.pointer_layout(field))
            }),
            ty::Array(element, _) | ty::Slice(element) | ty::Pat(element, _) => {
                self.pointer_layout(*element)
            }
            ty::Adt(definition, args) if !definition.is_union() && self.visiting.insert(ty) => {
                let mut layout = PointerLayout::Addressless;
                for variant in definition.variants() {
                    for field in &variant.fields {
                        layout = layout
                            .with(self.pointer_layout(field.ty(self.tcx, args).skip_norm_wip()));
                    }
                }
                self.visiting.remove(&ty);
                layout
            }
            // Opaque, recursive, and union storage lacks a closed field-by-field
            // device representation. Unknown layout stays neutral: normalization
            // exposes an ordinary addressless value later at times.
            _ => PointerLayout::Unresolved,
        }
    }

    fn is_drop_free(&mut self, ty: Ty<'tcx>) -> bool {
        match ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Never
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::RawPtr(..)
            | ty::Ref(..) => true,
            ty::Tuple(fields) => fields.iter().all(|field| self.is_drop_free(field)),
            ty::Array(element, _) | ty::Pat(element, _) => self.is_drop_free(*element),
            ty::Adt(definition, args)
                if !definition.is_union()
                    && !ty.has_non_region_param()
                    && !ty.has_infer()
                    && !ty.has_escaping_bound_vars()
                    && self.visiting.insert(ty) =>
            {
                // Inspect fields before querying drop glue. Copy alone leaves
                // hidden pointer storage and opaque types open.
                // Key recursion by instantiated type: Wrapper<Wrapper<f32>> is
                // a nested instantiation, distinct from a recursive field layout.
                let fields_are_closed = definition.variants().iter().all(|variant| {
                    variant
                        .fields
                        .iter()
                        .all(|field| self.is_drop_free(field.ty(self.tcx, args).skip_norm_wip()))
                });
                self.visiting.remove(&ty);
                fields_are_closed && !ty.needs_drop(self.tcx, ty::TypingEnv::fully_monomorphized())
            }
            _ => false,
        }
    }
}
