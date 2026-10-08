use rustc_middle::mir::ConstValue;
use rustc_session::Session;
use rustc_span::Spanned;

use crate::prelude::*;

#[derive(Default)]
pub struct Providers {
    pub target_triple: Option<fn(&Session) -> Option<target_lexicon::Triple>>,
    pub optimize: Option<for<'tcx> fn(TyCtxt<'tcx>, &mut Function)>,
    pub block: Option<for<'m, 'clif, 'tcx> fn(&mut FunctionCx<'m, 'clif, 'tcx>, BasicBlock, bool)>,
    pub statement:
        Option<for<'m, 'clif, 'tcx> fn(&mut FunctionCx<'m, 'clif, 'tcx>, &Statement<'tcx>) -> bool>,
    pub call: Option<
        for<'m, 'clif, 'tcx> fn(
            &mut FunctionCx<'m, 'clif, 'tcx>,
            SourceInfo,
            &Operand<'tcx>,
            &[Spanned<Operand<'tcx>>],
            Place<'tcx>,
            Option<BasicBlock>,
            UnwindAction,
        ) -> bool,
    >,
    pub intrinsic: Option<
        for<'m, 'clif, 'tcx> fn(
            &mut FunctionCx<'m, 'clif, 'tcx>,
            Instance<'tcx>,
            &[Spanned<Operand<'tcx>>],
            CPlace<'tcx>,
            Option<BasicBlock>,
            SourceInfo,
        ) -> Option<Result<(), Instance<'tcx>>>,
    >,
    pub constant: Option<
        for<'m, 'clif, 'tcx> fn(
            &mut FunctionCx<'m, 'clif, 'tcx>,
            ConstValue,
            Ty<'tcx>,
        ) -> Option<CValue<'tcx>>,
    >,
    pub copy_value: Option<
        for<'m, 'clif, 'tcx> fn(
            &mut FunctionCx<'m, 'clif, 'tcx>,
            Pointer,
            CValue<'tcx>,
            MemFlagsData,
        ) -> bool,
    >,
    pub requires_address: Option<for<'tcx> fn(&Rvalue<'tcx>) -> bool>,
    pub skip_drop: Option<for<'tcx> fn(TyCtxt<'tcx>, Instance<'tcx>) -> bool>,
    pub libcall: Option<
        for<'m, 'clif, 'tcx> fn(
            &mut FunctionCx<'m, 'clif, 'tcx>,
            &str,
            &Signature,
        ) -> Option<FuncRef>,
    >,
    pub indirect_libcall_return: Option<fn(&Session, Type) -> bool>,
    pub comparison_type: Option<fn(&Session) -> Type>,
}
