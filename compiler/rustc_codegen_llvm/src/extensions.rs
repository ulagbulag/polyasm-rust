use rustc_codegen_ssa::mir::{
    FunctionCx, IntrinsicResult,
    block::{CallKind, MergingSucc, TerminatorCodegenHelper},
    operand::OperandRef,
    place::PlaceValue,
};
use rustc_middle::{
    mir,
    ty::{Instance, layout::TyAndLayout},
};
use rustc_span::{Span, Spanned};

use crate::{builder::Builder, context::CodegenCx, llvm::Value};

pub struct Providers {
    pub intrinsic: for<'a, 'll, 'tcx> fn(
        &mut Builder<'a, 'll, 'tcx>,
        Instance<'tcx>,
        &[OperandRef<'tcx, &'ll Value>],
        TyAndLayout<'tcx>,
        Option<PlaceValue<&'ll Value>>,
        Span,
    ) -> Option<IntrinsicResult<'tcx, &'ll Value>>,
    pub call: for<'a, 'll, 'tcx> fn(
        &mut FunctionCx<'a, 'tcx, Builder<'a, 'll, 'tcx>>,
        &TerminatorCodegenHelper<'tcx>,
        &mut Builder<'a, 'll, 'tcx>,
        &mir::Operand<'tcx>,
        &[Spanned<mir::Operand<'tcx>>],
        mir::Place<'tcx>,
        Option<mir::BasicBlock>,
        mir::UnwindAction,
        mir::SourceInfo,
        &CallKind,
        bool,
    ) -> Option<MergingSucc>,
    pub call_attrs: for<'a, 'll, 'tcx> fn(
        &mut Builder<'a, 'll, 'tcx>,
        &FunctionCx<'a, 'tcx, Builder<'a, 'll, 'tcx>>,
        &TerminatorCodegenHelper<'tcx>,
        Option<Instance<'tcx>>,
        &'ll Value,
        &'ll Value,
    ),
    pub function_attrs:
        for<'ll, 'tcx> fn(&CodegenCx<'ll, 'tcx>, Option<Instance<'tcx>>, &'ll Value),
    pub callsite_attrs:
        for<'a, 'll, 'tcx> fn(&mut Builder<'a, 'll, 'tcx>, Option<Instance<'tcx>>, &'ll Value),
}
