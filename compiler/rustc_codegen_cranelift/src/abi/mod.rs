//! Handling of everything related to the calling convention. Also fills `fx.local_map`.

mod comments;
mod pass_mode;
mod returning;

use std::borrow::Cow;
use std::mem;

use cranelift_codegen::ir::{
    ArgumentPurpose, BlockArg, ExceptionTableData, ExceptionTableItem, ExceptionTag, SigRef,
};
use cranelift_codegen::isa::CallConv;
use cranelift_module::ModuleError;
use rustc_abi::{CanonAbi, ExternAbi, X86Call};
use rustc_codegen_ssa::base::is_call_from_compiler_builtins_to_upstream_monomorphization;
use rustc_codegen_ssa::diagnostics::CompilerBuiltinsCannotCall;
use rustc_codegen_ssa::polyasm::{
    normalized_polyasm_static_schedule, polyasm_static_callable_selection,
    polyasm_static_callable_selection_on, polyasm_static_clock_hz,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::DefKind;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mono::resolve_polyasm_callable;
use rustc_middle::ty::consts::ConstExt;
use rustc_middle::ty::layout::FnAbiOf;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{ShimKind, TypeVisitableExt};
use rustc_session::Session;
use rustc_span::{Spanned, sym};
use rustc_target::callconv::{FnAbi, PassMode};
use rustc_target::spec::Arch;
use smallvec::{SmallVec, smallvec};

use self::pass_mode::*;
pub(crate) use self::returning::codegen_return;
use crate::base::codegen_unwind_terminate;
use crate::debuginfo::EXCEPTION_HANDLER_CLEANUP;
use crate::prelude::*;

struct ArgValue<'tcx> {
    value: CValue<'tcx>,
    is_underaligned_pointee: bool,
}

fn clif_sig_from_fn_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    default_call_conv: CallConv,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> Signature {
    let call_conv = conv_to_call_conv(tcx.sess, fn_abi.conv, default_call_conv);

    let inputs = fn_abi.args.iter().flat_map(|arg_abi| arg_abi.get_abi_param(tcx).into_iter());

    let (return_ptr, returns) = fn_abi.ret.get_abi_return(tcx);
    // Sometimes the first param is a pointer to the place where the return value needs to be stored.
    let params: Vec<_> = return_ptr.into_iter().chain(inputs).collect();

    Signature { params, returns, call_conv }
}

pub(crate) fn conv_to_call_conv(
    sess: &Session,
    c: CanonAbi,
    default_call_conv: CallConv,
) -> CallConv {
    match c {
        CanonAbi::Rust | CanonAbi::RustCold | CanonAbi::C => default_call_conv,

        CanonAbi::RustPreserveNone | CanonAbi::RustTail => {
            sess.dcx().fatal(format!("call conv {c:?} is LLVM-specific"))
        }

        // Functions with this calling convention can only be called from assembly, but it is
        // possible to declare an `extern "custom"` block, so the backend still needs a calling
        // convention for declaring foreign functions.
        CanonAbi::Custom => default_call_conv,

        CanonAbi::X86(x86_call) => match x86_call {
            X86Call::SysV64 => CallConv::SystemV,
            X86Call::Win64 => CallConv::WindowsFastcall,
            // Should already get a back compat warning
            _ => default_call_conv,
        },

        CanonAbi::Interrupt(_) | CanonAbi::Arm(_) | CanonAbi::Swift => {
            sess.dcx().fatal(format!("call conv {c:?} is not yet implemented"))
        }
        CanonAbi::GpuKernel => {
            unreachable!("tried to use {c:?} call conv which only exists on an unsupported target")
        }
    }
}

pub(crate) fn get_function_sig<'tcx>(
    tcx: TyCtxt<'tcx>,
    default_call_conv: CallConv,
    inst: Instance<'tcx>,
) -> Signature {
    assert!(!inst.args.has_infer());
    clif_sig_from_fn_abi(
        tcx,
        default_call_conv,
        FullyMonomorphizedLayoutCx(tcx).fn_abi_of_instance(inst, ty::List::empty()),
    )
}

fn declare_import_function(
    tcx: TyCtxt<'_>,
    module: &mut dyn Module,
    name: &str,
    sig: &Signature,
) -> FuncId {
    match module.declare_function(name, Linkage::Import, sig) {
        Ok(func_id) => func_id,
        Err(ModuleError::IncompatibleDeclaration(_)) => tcx.dcx().fatal(format!(
            "attempt to declare `{name}` as function, but it was already declared as static"
        )),
        Err(ModuleError::IncompatibleSignature(_, prev_sig, new_sig)) => tcx.dcx().fatal(format!(
            "attempt to declare `{name}` with signature {new_sig:?}, \
             but it was already declared with signature {prev_sig:?}"
        )),
        Err(err) => Err::<_, _>(err).unwrap(),
    }
}

/// Instance must be monomorphized
pub(crate) fn import_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    module: &mut dyn Module,
    inst: Instance<'tcx>,
) -> FuncId {
    let name = tcx.symbol_name(inst).name;
    let sig = get_function_sig(tcx, module.target_config().default_call_conv, inst);
    declare_import_function(tcx, module, name, &sig)
}

impl<'tcx> FunctionCx<'_, '_, 'tcx> {
    /// Instance must be monomorphized
    pub(crate) fn get_function_ref(&mut self, inst: Instance<'tcx>) -> FuncRef {
        let func_id = import_function(self.tcx, self.module, inst);
        let func_ref = self.module.declare_func_in_func(func_id, self.bcx.func);

        if self.clif_comments.enabled() {
            self.add_comment(func_ref, format!("{:?}", inst));
        }

        func_ref
    }

    pub(crate) fn lib_call(
        &mut self,
        name: &str,
        params: Vec<AbiParam>,
        mut returns: Vec<AbiParam>,
        args: &[Value],
    ) -> Cow<'_, [Value]> {
        // FIXME any way to reuse the abi adjustment code in rustc_target?

        // Pass and return f128 indirectly on s390x and x86_64 Windows.
        let indirect_f128 = self.tcx.sess.target.arch == Arch::S390x
            || (self.tcx.sess.target.is_like_windows && self.tcx.sess.target.arch == Arch::X86_64);

        // Pass i128 arguments by-ref on s390x and Windows.
        let (params, args): (Vec<_>, Cow<'_, [_]>) =
            if self.tcx.sess.target.is_like_windows || self.tcx.sess.target.arch == Arch::S390x {
                let (params, args): (Vec<_>, Vec<_>) = params
                    .into_iter()
                    .zip(args)
                    .map(|(param, &arg)| {
                        if param.value_type == types::I128
                            || (indirect_f128 && param.value_type == types::F128)
                        {
                            let arg_ptr = self.create_stack_slot(16, 16);
                            arg_ptr.store(self, arg, MemFlagsData::trusted());
                            (AbiParam::new(self.pointer_type), arg_ptr.get_addr(self))
                        } else {
                            (param, arg)
                        }
                    })
                    .unzip();

                (params, args.into())
            } else {
                (params, args.into())
            };

        let ret_single_i128 = matches!(*returns, [AbiParam { value_type: types::I128, .. }]);
        let ret_single_f128 = matches!(*returns, [AbiParam { value_type: types::F128, .. }]);
        if ret_single_i128 && self.tcx.sess.target.is_like_windows {
            // Return i128 using the vector ABI on Windows
            returns[0].value_type = types::I64X2;

            let ret = self.lib_call_unadjusted(name, params, returns, &args)[0];

            Cow::Owned(vec![codegen_bitcast(self, types::I128, ret)])
        } else if (ret_single_i128 && self.tcx.sess.target.arch == Arch::S390x)
            || (ret_single_f128 && indirect_f128)
            || (self.tcx.sess.target.arch == Arch::Wasm32 && (ret_single_i128 || ret_single_f128))
        {
            // Return x86_64 Windows f128, s390x i128, and wasm32 i128 and f128 indirectly (sret
            // in LLVM terminology).
            let ret_ty = returns[0].value_type;
            let mut params = params;
            let mut args = args.to_vec();

            params.insert(0, AbiParam::special(self.pointer_type, ArgumentPurpose::StructReturn));
            let ret_ptr = self.create_stack_slot(16, 16);
            args.insert(0, ret_ptr.get_addr(self));

            self.lib_call_unadjusted(name, params, vec![], &args);

            Cow::Owned(vec![ret_ptr.load(self, ret_ty, MemFlagsData::trusted())])
        } else {
            Cow::Borrowed(self.lib_call_unadjusted(name, params, returns, &args))
        }
    }

    fn lib_call_unadjusted(
        &mut self,
        name: &str,
        params: Vec<AbiParam>,
        returns: Vec<AbiParam>,
        args: &[Value],
    ) -> &[Value] {
        let sig = Signature { params, returns, call_conv: self.target_config.default_call_conv };
        let func_ref = if crate::driver::polyasm::is_target(self.tcx.sess) && name == "memcmp" {
            // Preserve the compiler-generated comparison as a typed libcall
            // so executable translation emits MemoryCompare.
            let signature = self.bcx.import_signature(sig);
            self.bcx.import_function(cranelift_codegen::ir::ExtFuncData {
                name: cranelift_codegen::ir::ExternalName::LibCall(
                    cranelift_codegen::ir::LibCall::Memcmp,
                ),
                signature,
                colocated: false,
                patchable: false,
            })
        } else {
            let func_id = declare_import_function(self.tcx, self.module, name, &sig);
            self.module.declare_func_in_func(func_id, self.bcx.func)
        };
        let call_inst = self.bcx.ins().call(func_ref, args);
        if self.clif_comments.enabled() {
            self.add_comment(func_ref, format!("{:?}", name));
            self.add_comment(call_inst, format!("lib_call {}", name));
        }
        let results = self.bcx.inst_results(call_inst);
        assert!(results.len() <= 2, "{}", results.len());
        results
    }
}

/// Make a [`CPlace`] capable of holding value of the specified type.
fn make_local_place<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    local: Local,
    layout: TyAndLayout<'tcx>,
    is_ssa: bool,
) -> CPlace<'tcx> {
    if layout.is_unsized() {
        fx.tcx.dcx().span_fatal(
            fx.mir.local_decls[local].source_info.span,
            "unsized locals are not yet supported",
        );
    }
    let place = if is_ssa {
        if let BackendRepr::ScalarPair { .. } = layout.backend_repr {
            CPlace::new_var_pair(fx, local, layout)
        } else {
            CPlace::new_var(fx, local, layout)
        }
    } else {
        CPlace::new_stack_slot(fx, layout)
    };

    self::comments::add_local_place_comments(fx, place, local);

    place
}

pub(crate) fn codegen_fn_prelude<'tcx>(fx: &mut FunctionCx<'_, '_, 'tcx>, start_block: Block) {
    fx.bcx.append_block_params_for_function_params(start_block);

    fx.bcx.switch_to_block(start_block);
    fx.bcx.ins().nop();

    let ssa_analyzed = crate::analyze::analyze(fx);

    self::comments::add_args_header_comment(fx);

    let mut block_params_iter = fx.bcx.func.dfg.block_params(start_block).to_vec().into_iter();
    let ret_place =
        self::returning::codegen_return_param(fx, &ssa_analyzed, &mut block_params_iter);
    assert_eq!(fx.local_map.push(ret_place), RETURN_PLACE);

    // None means pass_mode == NoPass
    enum ArgKind<'tcx> {
        Normal(Option<ArgValue<'tcx>>),
        Spread(Vec<Option<ArgValue<'tcx>>>),
    }

    // FIXME implement variadics in cranelift
    if fx.fn_abi.c_variadic {
        fx.tcx.dcx().span_fatal(
            fx.mir.span,
            "Defining variadic functions is not yet supported by Cranelift",
        );
    }

    let mut arg_abis_iter = fx.fn_abi.args.iter();

    let func_params = fx
        .mir
        .args_iter()
        .map(|local| {
            let arg_ty = fx.monomorphize(fx.mir.local_decls[local].ty);

            // FIXME(splat): un-tuple splatted arguments in codegen, for performance
            // Adapted from https://github.com/rust-lang/rust/blob/145155dc96757002c7b2e9de8489416e2fdbbd57/src/librustc_codegen_llvm/mir/mod.rs#L442-L482
            if Some(local) == fx.mir.spread_arg {
                // This argument (e.g. the last argument in the "rust-call" ABI)
                // is a tuple that was spread at the ABI level and now we have
                // to reconstruct it into a tuple local variable, from multiple
                // individual function arguments.

                let tupled_arg_tys = match arg_ty.kind() {
                    ty::Tuple(tys) => tys,
                    _ => bug!("spread argument isn't a tuple?! but {:?}", arg_ty),
                };

                let mut params = Vec::new();
                for (i, _arg_ty) in tupled_arg_tys.iter().enumerate() {
                    let arg_abi = arg_abis_iter.next().unwrap();
                    let param =
                        cvalue_for_param(fx, Some(local), Some(i), arg_abi, &mut block_params_iter);
                    params.push(param);
                }

                (local, ArgKind::Spread(params), arg_ty)
            } else {
                let arg_abi = arg_abis_iter.next().unwrap();
                let param =
                    cvalue_for_param(fx, Some(local), None, arg_abi, &mut block_params_iter);
                (local, ArgKind::Normal(param), arg_ty)
            }
        })
        .collect::<Vec<(Local, ArgKind<'tcx>, Ty<'tcx>)>>();

    assert!(fx.caller_location.is_none());
    if fx.instance.def.requires_caller_location(fx.tcx) {
        // Store caller location for `#[track_caller]`.
        let arg_abi = arg_abis_iter.next().unwrap();
        let param = cvalue_for_param(fx, None, None, arg_abi, &mut block_params_iter).unwrap();
        assert!(
            !param.is_underaligned_pointee,
            "caller location argument should not be underaligned",
        );
        fx.caller_location = Some(param.value);
    }

    assert_eq!(arg_abis_iter.next(), None, "ArgAbi left behind for {:?}", fx.fn_abi);
    assert!(block_params_iter.next().is_none(), "arg_value left behind");

    self::comments::add_locals_header_comment(fx);

    for (local, arg_kind, ty) in func_params {
        // While this is normally an optimization to prevent an unnecessary copy when an argument is
        // not mutated by the current function, this is necessary to support unsized arguments.
        if let ArgKind::Normal(Some(ArgValue { value: val, is_underaligned_pointee: false })) =
            arg_kind
            && let Some((addr, meta)) = val.try_to_ptr()
        {
            // Ownership of the value at the backing storage for an argument is passed to the
            // callee per the ABI, so it is fine to borrow the backing storage of this argument
            // to prevent a copy.

            let place = if let Some(meta) = meta {
                CPlace::for_ptr_with_extra(addr, meta, val.layout())
            } else {
                CPlace::for_ptr(addr, val.layout())
            };

            self::comments::add_local_place_comments(fx, place, local);

            assert_eq!(fx.local_map.push(place), local);
            continue;
        }

        let layout = fx.layout_of(ty);
        let is_ssa = ssa_analyzed[local].is_ssa(fx, ty);
        let place = make_local_place(fx, local, layout, is_ssa);
        assert_eq!(fx.local_map.push(place), local);

        match arg_kind {
            ArgKind::Normal(param) => {
                if let Some(param) = param {
                    if param.is_underaligned_pointee {
                        place.write_cvalue_transmute(fx, param.value);
                    } else {
                        place.write_cvalue(fx, param.value);
                    }
                }
            }
            ArgKind::Spread(params) => {
                for (i, param) in params.into_iter().enumerate() {
                    if let Some(param) = param {
                        let field_place = place.place_field(fx, FieldIdx::new(i));
                        if param.is_underaligned_pointee {
                            field_place.write_cvalue_transmute(fx, param.value);
                        } else {
                            field_place.write_cvalue(fx, param.value);
                        }
                    }
                }
            }
        }
    }

    for local in fx.mir.vars_and_temps_iter() {
        let ty = fx.monomorphize(fx.mir.local_decls[local].ty);
        let layout = fx.layout_of(ty);

        let is_ssa = ssa_analyzed[local].is_ssa(fx, ty);

        let place = make_local_place(fx, local, layout, is_ssa);
        assert_eq!(fx.local_map.push(place), local);
    }

    fx.bcx.ins().jump(*fx.block_map.get(START_BLOCK).unwrap(), &[]);
}

struct CallArgument<'tcx> {
    value: CValue<'tcx>,
    is_owned: bool,
}

// FIXME avoid intermediate `CValue` before calling `adjust_arg_for_abi`
fn codegen_call_argument_operand<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    operand: &Operand<'tcx>,
) -> CallArgument<'tcx> {
    CallArgument {
        value: codegen_operand(fx, operand),
        is_owned: matches!(operand, Operand::Move(_)),
    }
}

fn codegen_polyasm_static_invoke<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    source_info: mir::SourceInfo,
    func: &Operand<'tcx>,
    args: &[Spanned<Operand<'tcx>>],
    destination: Place<'tcx>,
    target: Option<BasicBlock>,
    unwind: UnwindAction,
) -> bool {
    let Some((def_id, generic_args)) = func.const_fn_def() else {
        return false;
    };
    let exact = fx.tcx.is_diagnostic_item(sym::polyasm_invoke_static_faster_exact, def_id);
    let on = fx.tcx.is_diagnostic_item(sym::polyasm_invoke_static_faster_on, def_id);
    if !exact && !on && !fx.tcx.is_diagnostic_item(sym::polyasm_invoke_static_faster, def_id) {
        return false;
    }
    let [lhs, rhs] = args else {
        fx.tcx
            .dcx()
            .span_err(source_info.span, "PolyASM static selection marker has an invalid ABI");
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    };
    let generic_args = fx.monomorphize(generic_args);
    if exact || on {
        let lhs_ty = fx.monomorphize(lhs.node.ty(&fx.mir.local_decls, fx.tcx));
        let rhs_ty = fx.monomorphize(rhs.node.ty(&fx.mir.local_decls, fx.tcx));
        if lhs_ty != generic_args.type_at(0) || rhs_ty != generic_args.type_at(1) {
            fx.tcx.dcx().span_err(
                source_info.span,
                "PolyASM exact static selection operands changed callable identity",
            );
            fx.bcx.ins().trap(TrapCode::user(2).unwrap());
            return true;
        }
    }
    let selection = if on {
        match polyasm_static_callable_selection_on(fx.tcx, generic_args, source_info.span) {
            Ok(selection) => selection,
            Err(_error) => {
                // The semantic-domain emitter owns this source-located
                // diagnostic. The executable lowering still terminates its
                // block and leaves the diagnostic to that emitter.
                fx.bcx.ins().trap(TrapCode::user(2).unwrap());
                return true;
            }
        }
    } else {
        let Some(selection) =
            polyasm_static_callable_selection(fx.tcx, generic_args, source_info.span)
        else {
            fx.tcx.dcx().span_err(
                source_info.span,
                "PolyASM static selection requires two closed compiler-checked schedules",
            );
            fx.bcx.ins().trap(TrapCode::user(2).unwrap());
            return true;
        };
        selection
    };
    let selected = selection.selected();
    let selected_ty = selection.selected_ty();
    let selected_layout = fx.layout_of(selected_ty);
    if selected_layout.size.bytes() != 0 {
        fx.tcx.dcx().span_err(
            source_info.span,
            "PolyASM static selection cannot bind a runtime callable environment",
        );
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    }
    let ret_place = codegen_place(fx, destination);
    let fn_abi = FullyMonomorphizedLayoutCx(fx.tcx).fn_abi_of_instance(selected, ty::List::empty());
    let mut call_args = if fx.tcx.def_kind(selected.def_id()) == DefKind::Closure {
        let Some(environment) = fn_abi.args.first() else {
            fx.tcx
                .dcx()
                .span_err(source_info.span, "PolyASM selected closure has no environment ABI");
            fx.bcx.ins().trap(TrapCode::user(2).unwrap());
            return true;
        };
        let (value, is_owned) = if environment.layout.ty == selected_ty {
            (CValue::zst(selected_layout), true)
        } else if let ty::Ref(_, pointee, _) = *environment.layout.ty.kind()
            && pointee == selected_ty
        {
            let address = Pointer::dangling(selected_layout.align.abi).get_addr(fx);
            (CValue::by_val(address, environment.layout), false)
        } else {
            fx.tcx.dcx().span_err(
                source_info.span,
                "PolyASM selected closure has an unsupported environment ABI",
            );
            fx.bcx.ins().trap(TrapCode::user(2).unwrap());
            return true;
        };
        vec![CallArgument { value, is_owned }]
    } else {
        Vec::new()
    };
    if selected.def.requires_caller_location(fx.tcx) {
        call_args
            .push(CallArgument { value: fx.get_caller_location(source_info), is_owned: false });
    }
    if fn_abi.args.len() != call_args.len() {
        fx.tcx
            .dcx()
            .span_err(source_info.span, "PolyASM selected callable has an unexpected compiler ABI");
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    }
    if fx.tcx.codegen_instance_attrs(selected.def).flags.contains(CodegenFnAttrFlags::COLD) {
        fx.bcx.set_cold_block(fx.bcx.current_block().unwrap());
        if let Some(destination_block) = target {
            fx.bcx.set_cold_block(fx.get_block(destination_block));
        }
    }
    let func_ref = CallTarget::Direct(fx.get_function_ref(selected));
    self::returning::codegen_with_call_return_arg(fx, &fn_abi.ret, ret_place, |fx, return_ptr| {
        let call_args = return_ptr
            .into_iter()
            .chain(call_args.into_iter().enumerate().flat_map(|(index, argument)| {
                adjust_arg_for_abi(fx, argument.value, &fn_abi.args[index], argument.is_owned)
                    .into_iter()
            }))
            .collect::<Vec<Value>>();
        codegen_call_with_unwind_action(fx, source_info.span, func_ref, unwind, &call_args, None)
    });
    if let Some(destination) = target {
        let destination = fx.get_block(destination);
        fx.bcx.ins().jump(destination, &[]);
    } else {
        fx.bcx.ins().trap(TrapCode::user(1).unwrap());
    }
    true
}

fn polyasm_u64_const<'tcx>(tcx: TyCtxt<'tcx>, constant: ty::Const<'tcx>) -> Option<u64> {
    let value = constant.try_to_value()?;
    if value.ty != tcx.types.u64 {
        return None;
    }
    value.try_to_bits(tcx, ty::TypingEnv::fully_monomorphized())?.try_into().ok()
}

fn codegen_polyasm_static_bind<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    source_info: mir::SourceInfo,
    func: &Operand<'tcx>,
    args: &[Spanned<Operand<'tcx>>],
    destination: Place<'tcx>,
    target: Option<BasicBlock>,
) -> bool {
    let Some((def_id, generic_args)) = func.const_fn_def() else {
        return false;
    };
    if !fx.tcx.is_diagnostic_item(sym::polyasm_bind_static_clock, def_id) {
        return false;
    }
    let [argument] = args else {
        fx.tcx
            .dcx()
            .span_err(source_info.span, "PolyASM static-clock binding marker has an invalid ABI");
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    };
    let generic_args = fx.monomorphize(generic_args);
    let callable_ty = generic_args.type_at(1);
    let argument_ty = fx.monomorphize(argument.node.ty(&fx.mir.local_decls, fx.tcx));
    let closed = match *callable_ty.kind() {
        ty::FnDef(..) => true,
        ty::Closure(_, args) => args.as_closure().tupled_upvars_ty().is_unit(),
        _ => false,
    };
    let Some(callable) = (argument_ty == callable_ty && closed)
        .then(|| resolve_polyasm_callable(fx.tcx, callable_ty, source_info.span))
        .flatten()
    else {
        fx.tcx.dcx().span_err(
            source_info.span,
            "PolyASM static-clock binding lost its closed callable identity",
        );
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    };
    let requested_cycles = polyasm_u64_const(fx.tcx, generic_args.const_at(2));
    let architecture = generic_args.type_at(0);
    let clock_hz = polyasm_static_clock_hz(fx.tcx, architecture);
    let (Some(requested_cycles), Some(clock_hz)) = (requested_cycles, clock_hz) else {
        fx.tcx.dcx().span_err(
            source_info.span,
            "PolyASM static-clock binding does not have a concrete registered schedule",
        );
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    };
    if clock_hz == 0
        || normalized_polyasm_static_schedule(fx.tcx, callable, architecture)
            != Some(requested_cycles)
    {
        fx.tcx.dcx().span_err(
            source_info.span,
            "PolyASM static-clock binding does not match the normalized callable schedule",
        );
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    }
    let destination_ty = fx.monomorphize(destination.ty(&fx.mir.local_decls, fx.tcx).ty);
    let ty::Adt(destination_definition, destination_args) = *destination_ty.kind() else {
        fx.tcx.dcx().span_err(
            source_info.span,
            "PolyASM static-clock binding has an invalid destination representation",
        );
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    };
    if !fx.tcx.is_diagnostic_item(sym::polyasm_static_callable, destination_definition.did())
        || destination_args.len() != 3
        || destination_args.type_at(0) != callable_ty
        || destination_args.type_at(1) != architecture
        || polyasm_u64_const(fx.tcx, destination_args.const_at(2)) != Some(requested_cycles)
    {
        fx.tcx.dcx().span_err(
            source_info.span,
            "PolyASM static-clock binding destination does not preserve its exact callable schedule",
        );
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    }
    let destination_layout = fx.layout_of(destination_ty);
    if fx.layout_of(callable_ty).size.bytes() != 0 || destination_layout.fields.count() != 3 {
        fx.tcx.dcx().span_err(
            source_info.span,
            "PolyASM static-clock binding cannot retain a runtime callable environment",
        );
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    }
    let callable_layout = destination_layout.field(&*fx, 0);
    let warrant_layout = destination_layout.field(&*fx, 1);
    let architecture_layout = destination_layout.field(&*fx, 2);
    if callable_layout.ty != callable_ty
        || callable_layout.size.bytes() != 0
        || architecture_layout.size.bytes() != 0
        || warrant_layout.fields.count() != 2
    {
        fx.tcx.dcx().span_err(
            source_info.span,
            "PolyASM static-clock binding has an incompatible witness layout",
        );
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    }
    let property_layout = warrant_layout.field(&*fx, 0);
    let schedule_layout = warrant_layout.field(&*fx, 1);
    if property_layout.size.bytes() != 0 || schedule_layout.fields.count() != 2 {
        fx.tcx.dcx().span_err(
            source_info.span,
            "PolyASM static-clock binding has an incompatible warrant layout",
        );
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    }
    let clock_layout = schedule_layout.field(&*fx, 0);
    let cycles_layout = schedule_layout.field(&*fx, 1);
    let clock_offset = destination_layout.fields.offset(1).bytes()
        + warrant_layout.fields.offset(1).bytes()
        + schedule_layout.fields.offset(0).bytes();
    let cycles_offset = destination_layout.fields.offset(1).bytes()
        + warrant_layout.fields.offset(1).bytes()
        + schedule_layout.fields.offset(1).bytes();
    let BackendRepr::ScalarPair { a, b, b_offset } = destination_layout.backend_repr else {
        fx.tcx.dcx().span_err(
            source_info.span,
            "PolyASM static-clock binding has a non-scalar schedule representation",
        );
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    };
    let b_offset = b_offset.bytes();
    let cycles_first = (cycles_offset, clock_offset) == (0, b_offset);
    let clock_first = (clock_offset, cycles_offset) == (0, b_offset);
    if cycles_layout.ty != fx.tcx.types.u64
        || clock_layout.size.bytes() != 8
        || !matches!(clock_layout.backend_repr, BackendRepr::Scalar(_))
        || scalar_to_clif_type(fx.tcx, a) != types::I64
        || scalar_to_clif_type(fx.tcx, b) != types::I64
        || (!cycles_first && !clock_first)
    {
        fx.tcx.dcx().span_err(
            source_info.span,
            "PolyASM static-clock binding cannot materialize its exact schedule payload",
        );
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        return true;
    }
    let cycles = fx.bcx.ins().iconst(types::I64, requested_cycles as i64);
    let clock_hz = fx.bcx.ins().iconst(types::I64, clock_hz as i64);
    let (first, second) = if cycles_first { (cycles, clock_hz) } else { (clock_hz, cycles) };
    codegen_place(fx, destination)
        .write_cvalue(fx, CValue::by_val_pair(first, second, destination_layout));
    if let Some(target) = target {
        let target = fx.get_block(target);
        fx.bcx.ins().jump(target, &[]);
    } else {
        fx.bcx.ins().trap(TrapCode::user(2).unwrap());
    }
    true
}

pub(crate) fn codegen_terminator_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    source_info: mir::SourceInfo,
    func: &Operand<'tcx>,
    args: &[Spanned<Operand<'tcx>>],
    destination: Place<'tcx>,
    target: Option<BasicBlock>,
    unwind: UnwindAction,
) {
    if fx.tcx.sess.is_polyasm_target()
        && codegen_polyasm_static_bind(fx, source_info, func, args, destination, target)
    {
        return;
    }
    if fx.tcx.sess.is_polyasm_target()
        && codegen_polyasm_static_invoke(fx, source_info, func, args, destination, target, unwind)
    {
        return;
    }
    if fx.tcx.sess.is_polyasm_target()
        && func.const_fn_def().is_some_and(|(def_id, _)| {
            fx.tcx.is_diagnostic_item(sym::polyasm_require_always, def_id)
                || fx.tcx.is_diagnostic_item(sym::polyasm_require_not_always, def_id)
        })
    {
        let [_argument] = args else {
            fx.tcx
                .dcx()
                .span_err(source_info.span, "PolyASM callable warrant marker has an invalid ABI");
            fx.bcx.ins().trap(TrapCode::user(2).unwrap());
            return;
        };
        if let Some(target) = target {
            let target = fx.get_block(target);
            fx.bcx.ins().jump(target, &[]);
        } else {
            fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        }
        return;
    }

    // An offload request answers its own argument back, exactly as a
    // statement marker does, so both close here by writing that argument
    // into the destination in place of emitting a call.
    if fx.tcx.sess.is_polyasm_target()
        && func.const_fn_def().is_some_and(|(def_id, _)| {
            fx.tcx.is_diagnostic_item(sym::polyasm_require_statement, def_id)
                || fx.tcx.is_diagnostic_item(sym::polyasm_offload, def_id)
                || fx.tcx.is_diagnostic_item(sym::polyasm_request_offload, def_id)
        })
    {
        let [argument] = args else {
            fx.tcx
                .dcx()
                .span_err(source_info.span, "PolyASM value-answering marker has an invalid ABI");
            fx.bcx.ins().trap(TrapCode::user(2).unwrap());
            return;
        };
        let value = codegen_operand(fx, &argument.node);
        let destination = codegen_place(fx, destination);
        destination.write_cvalue(fx, value);
        if let Some(target) = target {
            let target = fx.get_block(target);
            fx.bcx.ins().jump(target, &[]);
        } else {
            fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        }
        return;
    }

    if fx.tcx.sess.is_polyasm_target()
        && func
            .const_fn_def()
            .and_then(|(def_id, args)| {
                ty::Instance::try_resolve(
                    fx.tcx,
                    ty::TypingEnv::fully_monomorphized(),
                    def_id,
                    fx.monomorphize(args),
                )
                .ok()
                .flatten()
            })
            .is_some_and(|instance| fx.tcx.polyasm_witness_only_wrapper(instance))
    {
        if let Some(target) = target {
            let target = fx.get_block(target);
            fx.bcx.ins().jump(target, &[]);
        } else {
            fx.bcx.ins().trap(TrapCode::user(2).unwrap());
        }
        return;
    }

    let func = codegen_operand(fx, func);
    let fn_sig = func.layout().ty.fn_sig(fx.tcx);

    let ret_place = codegen_place(fx, destination);

    // Handle special calls like intrinsics and empty drop glue.
    let instance = if let ty::FnDef(def_id, fn_args) = *func.layout().ty.kind() {
        let instance = ty::Instance::expect_resolve(
            fx.tcx,
            ty::TypingEnv::fully_monomorphized(),
            def_id,
            fn_args.no_bound_vars().unwrap(),
            source_info.span,
        );

        if is_call_from_compiler_builtins_to_upstream_monomorphization(fx.tcx, instance) {
            if target.is_some() {
                let caller_def = fx.instance.def_id();
                let e = CompilerBuiltinsCannotCall {
                    span: fx.tcx.def_span(caller_def),
                    caller: with_no_trimmed_paths!(fx.tcx.def_path_str(caller_def)),
                    callee: with_no_trimmed_paths!(fx.tcx.def_path_str(def_id)),
                };
                fx.tcx.dcx().emit_err(e);
            } else {
                fx.bcx.ins().trap(TrapCode::user(2).unwrap());
                return;
            }
        }

        match instance.def {
            InstanceKind::Intrinsic(_) => {
                match crate::intrinsics::codegen_intrinsic_call(
                    fx,
                    instance,
                    args,
                    ret_place,
                    target,
                    source_info,
                ) {
                    Ok(()) => return,
                    Err(instance) => Some(instance),
                }
            }
            InstanceKind::LlvmIntrinsic(_) => {
                crate::intrinsics::codegen_llvm_intrinsic_call(
                    fx,
                    fx.tcx.symbol_name(instance).name,
                    args,
                    ret_place,
                    target,
                    source_info.span,
                );
                return;
            }
            // We don't need AsyncDropGlueCtorShim here because it is not `noop func`,
            // it is `func returning noop future`
            InstanceKind::Shim(ShimKind::DropGlue(_, None)) => {
                // empty drop glue - a nop.
                let dest = target.expect("Non terminating drop_in_place_real???");
                let ret_block = fx.get_block(dest);
                fx.bcx.ins().jump(ret_block, &[]);
                return;
            }
            _ => Some(instance),
        }
    } else {
        None
    };

    let extra_args = &args[fn_sig.inputs().skip_binder().len()..];
    let extra_args = fx.tcx.mk_type_list_from_iter(
        extra_args.iter().map(|op_arg| fx.monomorphize(op_arg.node.ty(fx.mir, fx.tcx))),
    );
    let fn_abi = if let Some(instance) = instance {
        FullyMonomorphizedLayoutCx(fx.tcx).fn_abi_of_instance(instance, extra_args)
    } else {
        FullyMonomorphizedLayoutCx(fx.tcx).fn_abi_of_fn_ptr(fn_sig, extra_args)
    };

    let is_cold = if fn_sig.abi() == ExternAbi::RustCold {
        true
    } else {
        instance.is_some_and(|inst| {
            fx.tcx.codegen_instance_attrs(inst.def).flags.contains(CodegenFnAttrFlags::COLD)
        })
    };
    if is_cold {
        fx.bcx.set_cold_block(fx.bcx.current_block().unwrap());
        if let Some(destination_block) = target {
            fx.bcx.set_cold_block(fx.get_block(destination_block));
        }
    }

    // Unpack arguments tuple for closures
    let mut args = if fn_sig.abi() == ExternAbi::RustCall {
        let (self_arg, pack_arg) = match args {
            [pack_arg] => (None, codegen_call_argument_operand(fx, &pack_arg.node)),
            [self_arg, pack_arg] => (
                Some(codegen_call_argument_operand(fx, &self_arg.node)),
                codegen_call_argument_operand(fx, &pack_arg.node),
            ),
            _ => panic!("rust-call abi requires one or two arguments"),
        };

        let tupled_arguments = match pack_arg.value.layout().ty.kind() {
            ty::Tuple(tupled_arguments) => tupled_arguments,
            _ => bug!("argument to function with \"rust-call\" ABI is not a tuple"),
        };

        let mut args = Vec::with_capacity(1 + tupled_arguments.len());
        args.extend(self_arg);
        for i in 0..tupled_arguments.len() {
            args.push(CallArgument {
                value: pack_arg.value.value_field(fx, FieldIdx::new(i)),
                is_owned: pack_arg.is_owned,
            });
        }
        args
    } else {
        args.iter().map(|arg| codegen_call_argument_operand(fx, &arg.node)).collect::<Vec<_>>()
    };

    // Pass the caller location for `#[track_caller]`.
    if instance.is_some_and(|inst| inst.def.requires_caller_location(fx.tcx)) {
        let caller_location = fx.get_caller_location(source_info);
        args.push(CallArgument { value: caller_location, is_owned: false });
    }

    let args = args;
    assert_eq!(fn_abi.args.len(), args.len());

    let (func_ref, first_arg_override) = match instance {
        // Trait object call
        Some(Instance { def: InstanceKind::Virtual(_, idx), .. }) => {
            if fx.clif_comments.enabled() {
                let nop_inst = fx.bcx.ins().nop();
                fx.add_post_comment(
                    nop_inst,
                    with_no_trimmed_paths!(format!(
                        "virtual call; self arg pass mode: {:?}",
                        fn_abi.args[0]
                    )),
                );
            }

            let (ptr, method) = crate::vtable::get_ptr_and_method_ref(fx, args[0].value, idx);
            let sig = clif_sig_from_fn_abi(fx.tcx, fx.target_config.default_call_conv, fn_abi);
            let sig = fx.bcx.import_signature(sig);

            (CallTarget::Indirect(sig, method), Some(ptr.get_addr(fx)))
        }

        // Normal call
        Some(instance) => {
            let func_ref = fx.get_function_ref(instance);
            (CallTarget::Direct(func_ref), None)
        }

        // Indirect call
        None => {
            if fx.clif_comments.enabled() {
                let nop_inst = fx.bcx.ins().nop();
                fx.add_post_comment(nop_inst, "indirect call");
            }

            let func = func.load_scalar(fx);
            let sig = clif_sig_from_fn_abi(fx.tcx, fx.target_config.default_call_conv, fn_abi);
            let sig = fx.bcx.import_signature(sig);

            (CallTarget::Indirect(sig, func), None)
        }
    };

    self::returning::codegen_with_call_return_arg(fx, &fn_abi.ret, ret_place, |fx, return_ptr| {
        let mut call_args = return_ptr
            .into_iter()
            .chain(first_arg_override)
            .chain(
                args.into_iter()
                    .enumerate()
                    .skip(if first_arg_override.is_some() { 1 } else { 0 })
                    .flat_map(|(i, arg)| {
                        adjust_arg_for_abi(fx, arg.value, &fn_abi.args[i], arg.is_owned).into_iter()
                    }),
            )
            .collect::<Vec<Value>>();

        // FIXME: Find a cleaner way to support varargs.
        if fn_abi.c_variadic {
            adjust_call_for_c_variadic(fx, fn_abi, source_info, func_ref, &mut call_args);
        }

        if fx.clif_comments.enabled() {
            let nop_inst = fx.bcx.ins().nop();
            with_no_trimmed_paths!(fx.add_post_comment(nop_inst, format!("abi: {:?}", fn_abi)));
        }

        codegen_call_with_unwind_action(fx, source_info.span, func_ref, unwind, &call_args, None)
    });

    if let Some(dest) = target {
        let ret_block = fx.get_block(dest);
        fx.bcx.ins().jump(ret_block, &[]);
    } else {
        fx.bcx.ins().trap(TrapCode::user(1 /* unreachable */).unwrap());
    }

    fn adjust_call_for_c_variadic<'tcx>(
        fx: &mut FunctionCx<'_, '_, 'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        source_info: mir::SourceInfo,
        target: CallTarget,
        call_args: &mut Vec<Value>,
    ) {
        if fn_abi.conv != CanonAbi::C {
            fx.tcx.dcx().span_fatal(
                source_info.span,
                format!("Variadic call for non-C abi {:?}", fn_abi.conv),
            );
        }
        let sig_ref = match target {
            CallTarget::Direct(func_ref) => fx.bcx.func.dfg.ext_funcs[func_ref].signature,
            CallTarget::Indirect(sig_ref, _) => sig_ref,
        };
        // `mem::take()` the `params` so that `fx.bcx` can be used below.
        let mut abi_params = mem::take(&mut fx.bcx.func.dfg.signatures[sig_ref].params);

        // Recalculate the parameters in the signature to ensure the signature contains the variadic arguments.
        let has_return_arg = matches!(fn_abi.ret.mode, PassMode::Indirect { .. });
        // Drop everything except the return argument (if there is one).
        abi_params.truncate(if has_return_arg { 1 } else { 0 });
        // Add the fixed arguments.
        abi_params.extend(
            fn_abi.args[..fn_abi.fixed_count as usize]
                .iter()
                .flat_map(|arg_abi| arg_abi.get_abi_param(fx.tcx).into_iter()),
        );
        let fixed_arg_count = abi_params.len();
        // Add the variadic arguments.
        abi_params.extend(
            fn_abi.args[fn_abi.fixed_count as usize..]
                .iter()
                .flat_map(|arg_abi| arg_abi.get_abi_param(fx.tcx).into_iter()),
        );

        if fx.tcx.sess.target.is_like_darwin && fx.tcx.sess.target.arch == Arch::AArch64 {
            // Add any padding arguments needed for Apple AArch64.
            // There's no need to pad the argument list unless variadic arguments are actually being
            // passed.
            if abi_params.len() > fixed_arg_count {
                // 128-bit integers take 2 registers, and everything else takes 1.
                // FIXME: Add support for non-integer types
                // This relies on the checks below to ensure all arguments are integer types and
                // that the ABI is "C".
                // The return argument isn't counted as it goes in its own dedicated register.
                let integer_registers_used: usize = abi_params
                    [if has_return_arg { 1 } else { 0 }..fixed_arg_count]
                    .iter()
                    .map(|arg| if arg.value_type.bits() == 128 { 2 } else { 1 })
                    .sum();
                // The ABI uses 8 registers before it starts pushing arguments to the stack. Pad out
                // the registers if needed to ensure the variadic arguments are passed on the stack.
                if integer_registers_used < 8 {
                    abi_params.splice(
                        fixed_arg_count..fixed_arg_count,
                        (integer_registers_used..8).map(|_| AbiParam::new(types::I64)),
                    );
                    call_args.splice(
                        fixed_arg_count..fixed_arg_count,
                        (integer_registers_used..8).map(|_| fx.bcx.ins().iconst(types::I64, 0)),
                    );
                }
            }

            // `StructArgument` is not currently used by the `aarch64` ABI, and is therefore not
            // handled when calculating how many padding arguments to use. Assert that this remains
            // the case.
            assert!(abi_params.iter().all(|param| matches!(
                param.purpose,
                // The only purposes used are `Normal` and `StructReturn`.
                ArgumentPurpose::Normal | ArgumentPurpose::StructReturn
            )));
        }

        // Check all parameters are integers.
        for param in abi_params.iter() {
            if !param.value_type.is_int() {
                // FIXME: Set %al to upperbound on float args once floats are supported.
                fx.tcx.dcx().span_fatal(
                    source_info.span,
                    format!("Non int ty {:?} for variadic call", param.value_type),
                );
            }
        }

        assert_eq!(abi_params.len(), call_args.len());

        // Put the `AbiParam`s back in the signature.
        fx.bcx.func.dfg.signatures[sig_ref].params = abi_params;
    }
}

/// Answers whether running one drop glue would leave the program as it was.
///
/// `ShimKind::DropGlue(_, None)` already states an empty drop, and
/// the arm above it answers that. This answers the one beside it: a glue that
/// *exists* and whose body, once the shim was built and elaborated, writes
/// zero bytes and asserts zero conditions, and whose every drop and every call names a
/// body that answers the same. `Box<T>` over a zero-sized `T` is what sits at
/// the bottom of it -- deallocating zero bytes is empty work, and what the glue is
/// left holding is a bare `return`.
///
/// The whole body is skipped in place of trimmed, so every read stays
/// unasked: a load stays unrun, and every program leaves a place being read
/// unobserved. A write is observed, and so are a call, an assertion, an unwind
/// edge and a discriminant store, and each of those stops the answer. A
/// projection on the left of an assignment is a write through a pointer and
/// stops it too, which is why the place names a bare local.
///
/// A missing `SwitchInt` arm in the terminator list would leave the `Option`
/// case unanswered, and the presence of `Assert` there would make the answer
/// wrong.
///
/// A `Drop` and a `Call` are answered by the body they name in place of their
/// own kind, because the chain this walks is four bodies deep and the
/// emptiness sits at the bottom of it: `Box<T>` over a zero-sized `T` holds a
/// bare `return`, `Option<Box<T>>` holds one `Drop` on that `Box`, `[T]` holds
/// one `Drop` on the element inside its walk, and `[T; N]` holds one `Call` to
/// the slice body. Reading one body alone answers `false` for the upper three
/// and leaves a thousand calls to a body that is `mov $0,%eax; ret`. The walk
/// skips the unwind edge on such a terminator, since only a running body
/// unwinds, and a `Call` that lacks a return target diverges and answers `false`.
///
/// `answered` is what stops a glue that reaches itself: a recursive type walks
/// back to a body already on the stack, which every finite reading leaves open,
/// and meeting one answers `false` in place of following it.
fn glue_leaves_the_program_unchanged<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    let mut answered = FxHashSet::default();
    drop_glue_body_leaves_the_program_unchanged(tcx, instance, &mut answered)
}

fn drop_glue_body_leaves_the_program_unchanged<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    answered: &mut FxHashSet<Instance<'tcx>>,
) -> bool {
    match instance.def {
        ty::InstanceKind::Shim(ty::ShimKind::DropGlue(_, None)) => return true,
        ty::InstanceKind::Shim(ty::ShimKind::DropGlue(_, Some(_))) => {}
        _ => return false,
    }
    if !answered.insert(instance) {
        return false;
    }
    let typing_env = ty::TypingEnv::fully_monomorphized();
    let body = tcx.instance_mir(instance.def);
    body.basic_blocks.iter().all(|block| {
        block.statements.iter().all(|statement| match &statement.kind {
            mir::StatementKind::StorageLive(_)
            | mir::StatementKind::StorageDead(_)
            | mir::StatementKind::FakeRead(_)
            | mir::StatementKind::PlaceMention(_)
            | mir::StatementKind::AscribeUserType(_, _)
            | mir::StatementKind::ConstEvalCounter
            | mir::StatementKind::Nop => true,
            mir::StatementKind::Assign(assignment) => assignment.0.projection.is_empty(),
            _ => false,
        }) && match &block.terminator().kind {
            mir::TerminatorKind::Return
            | mir::TerminatorKind::Goto { .. }
            | mir::TerminatorKind::SwitchInt { .. }
            | mir::TerminatorKind::Unreachable => true,
            mir::TerminatorKind::Drop { place, drop: None, .. } => {
                let dropped = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    ty::EarlyBinder::bind(tcx, place.ty(&body.local_decls, tcx).ty),
                );
                !dropped.is_trait()
                    && drop_glue_body_leaves_the_program_unchanged(
                        tcx,
                        Instance::resolve_drop_glue(tcx, dropped),
                        answered,
                    )
            }
            mir::TerminatorKind::Call { func, target: Some(_), .. } => {
                let called = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    ty::EarlyBinder::bind(tcx, func.ty(&body.local_decls, tcx)),
                );
                let ty::FnDef(def_id, bound_args) = *called.kind() else {
                    return false;
                };
                let Some(args) = bound_args.no_bound_vars() else {
                    return false;
                };
                let Ok(Some(callee)) = Instance::try_resolve(tcx, typing_env, def_id, args) else {
                    return false;
                };
                drop_glue_body_leaves_the_program_unchanged(tcx, callee, answered)
            }
            _ => false,
        }
    })
}

pub(crate) fn codegen_drop<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    source_info: mir::SourceInfo,
    drop_place: CPlace<'tcx>,
    target: BasicBlock,
    unwind: UnwindAction,
) {
    let ty = drop_place.layout().ty;
    let drop_instance = Instance::resolve_drop_glue(fx.tcx, ty);
    let ret_block = fx.get_block(target);

    // AsyncDropGlueCtorShim can't be here
    if let ty::InstanceKind::Shim(ty::ShimKind::DropGlue(_, None)) = drop_instance.def {
        // we don't actually need to drop anything
        fx.bcx.ins().jump(ret_block, &[]);
    } else if glue_leaves_the_program_unchanged(fx.tcx, drop_instance) {
        fx.bcx.ins().jump(ret_block, &[]);
    } else {
        match ty.kind() {
            ty::Dynamic(_, _) => {
                // IN THIS ARM, WE HAVE:
                // ty = *mut (dyn Trait)
                // which is: exists<T> ( *mut T,    Vtable<T: Trait> )
                //                       args[0]    args[1]
                //
                // args = ( Data, Vtable )
                //                  |
                //                  v
                //                /-------\
                //                | ...   |
                //                \-------/
                //
                let (ptr, vtable) = drop_place.to_ptr_unsized();
                let ptr = ptr.get_addr(fx);
                let drop_fn = crate::vtable::drop_fn_of_obj(fx, vtable);

                let is_null = fx.bcx.ins().icmp_imm_u(IntCC::Equal, drop_fn, 0);
                let target_block = fx.get_block(target);
                let continued = fx.bcx.create_block();
                fx.bcx.ins().brif(is_null, target_block, &[], continued, &[]);
                fx.bcx.switch_to_block(continued);

                // FIXME(eddyb) perhaps move some of this logic into
                // `Instance::resolve_drop_glue`?
                let virtual_drop = Instance {
                    def: ty::InstanceKind::Virtual(drop_instance.def_id(), 0),
                    args: drop_instance.args,
                };
                let fn_abi = FullyMonomorphizedLayoutCx(fx.tcx)
                    .fn_abi_of_instance(virtual_drop, ty::List::empty());

                let sig = clif_sig_from_fn_abi(fx.tcx, fx.target_config.default_call_conv, fn_abi);
                let sig = fx.bcx.import_signature(sig);
                codegen_call_with_unwind_action(
                    fx,
                    source_info.span,
                    CallTarget::Indirect(sig, drop_fn),
                    unwind,
                    &[ptr],
                    Some(ret_block),
                );
            }
            _ => {
                assert!(!matches!(drop_instance.def, InstanceKind::Virtual(_, _)));

                let fn_abi = FullyMonomorphizedLayoutCx(fx.tcx)
                    .fn_abi_of_instance(drop_instance, ty::List::empty());

                let arg_value = drop_place.place_ref(
                    fx,
                    fx.layout_of(Ty::new_mut_ref(fx.tcx, fx.tcx.lifetimes.re_erased, ty)),
                );
                let arg_value = adjust_arg_for_abi(fx, arg_value, &fn_abi.args[0], true);

                let mut call_args: Vec<Value> = arg_value.into_iter().collect::<Vec<_>>();

                if drop_instance.def.requires_caller_location(fx.tcx) {
                    // Pass the caller location for `#[track_caller]`.
                    let caller_location = fx.get_caller_location(source_info);
                    call_args.extend(adjust_arg_for_abi(
                        fx,
                        caller_location,
                        &fn_abi.args[1],
                        false,
                    ));
                }

                let func_ref = fx.get_function_ref(drop_instance);
                codegen_call_with_unwind_action(
                    fx,
                    source_info.span,
                    CallTarget::Direct(func_ref),
                    unwind,
                    &call_args,
                    Some(ret_block),
                );
            }
        }
    }
}

#[derive(Copy, Clone)]
pub(crate) enum CallTarget {
    Direct(FuncRef),
    Indirect(SigRef, Value),
}

pub(crate) fn codegen_call_with_unwind_action(
    fx: &mut FunctionCx<'_, '_, '_>,
    span: Span,
    func_ref: CallTarget,
    mut unwind: UnwindAction,
    call_args: &[Value],
    target_block: Option<Block>,
) -> SmallVec<[Value; 2]> {
    let sig_ref = match func_ref {
        CallTarget::Direct(func_ref) => fx.bcx.func.dfg.ext_funcs[func_ref].signature,
        CallTarget::Indirect(sig_ref, _func_ptr) => sig_ref,
    };

    if target_block.is_some() {
        assert!(fx.bcx.func.dfg.signatures[sig_ref].returns.is_empty());
    }

    if cfg!(not(feature = "unwinding")) {
        unwind = UnwindAction::Unreachable;
    }

    match unwind {
        UnwindAction::Continue | UnwindAction::Unreachable => {
            let call_inst = match func_ref {
                CallTarget::Direct(func_ref) => fx.bcx.ins().call(func_ref, call_args),
                CallTarget::Indirect(sig, func_ptr) => {
                    fx.bcx.ins().call_indirect(sig, func_ptr, call_args)
                }
            };

            if let Some(target_block) = target_block {
                fx.bcx.ins().jump(target_block, &[]);
                smallvec![]
            } else {
                fx.bcx
                    .func
                    .dfg
                    .inst_results(call_inst)
                    .iter()
                    .copied()
                    .collect::<SmallVec<[Value; 2]>>()
            }
        }
        UnwindAction::Cleanup(_) | UnwindAction::Terminate(_) => {
            let returns_types = fx.bcx.func.dfg.signatures[sig_ref]
                .returns
                .iter()
                .map(|return_param| return_param.value_type)
                .collect::<Vec<_>>();

            let fallthrough_block = fx.bcx.create_block();
            let fallthrough_block_call_args = returns_types
                .iter()
                .enumerate()
                .map(|(i, _)| BlockArg::TryCallRet(i.try_into().unwrap()))
                .collect::<Vec<_>>();
            let fallthrough_block_call = fx.bcx.func.dfg.block_call(
                target_block.unwrap_or(fallthrough_block),
                &fallthrough_block_call_args,
            );
            let pre_cleanup_block = fx.bcx.create_block();
            let pre_cleanup_block_call =
                fx.bcx.func.dfg.block_call(pre_cleanup_block, &[BlockArg::TryCallExn(0)]);
            let exception_table = fx.bcx.func.dfg.exception_tables.push(ExceptionTableData::new(
                sig_ref,
                fallthrough_block_call,
                [ExceptionTableItem::Tag(
                    ExceptionTag::with_number(EXCEPTION_HANDLER_CLEANUP).unwrap(),
                    pre_cleanup_block_call,
                )],
            ));

            match func_ref {
                CallTarget::Direct(func_ref) => {
                    fx.bcx.ins().try_call(func_ref, call_args, exception_table);
                }
                CallTarget::Indirect(_sig, func_ptr) => {
                    fx.bcx.ins().try_call_indirect(func_ptr, call_args, exception_table);
                }
            }

            fx.bcx.seal_block(pre_cleanup_block);
            fx.bcx.switch_to_block(pre_cleanup_block);
            fx.bcx.set_cold_block(pre_cleanup_block);
            match unwind {
                UnwindAction::Continue | UnwindAction::Unreachable => unreachable!(),
                UnwindAction::Cleanup(cleanup) => {
                    let exception_ptr =
                        fx.bcx.append_block_param(pre_cleanup_block, fx.pointer_type);
                    fx.bcx.def_var(fx.exception_slot, exception_ptr);
                    let cleanup_block = fx.get_block(cleanup);
                    fx.bcx.ins().jump(cleanup_block, &[]);
                }
                UnwindAction::Terminate(reason) => {
                    // FIXME dedup terminate blocks
                    fx.bcx.append_block_param(pre_cleanup_block, fx.pointer_type);

                    codegen_unwind_terminate(fx, span, reason);
                }
            }

            if target_block.is_none() {
                fx.bcx.seal_block(fallthrough_block);
                fx.bcx.switch_to_block(fallthrough_block);
                let returns = returns_types
                    .into_iter()
                    .map(|ty| fx.bcx.append_block_param(fallthrough_block, ty))
                    .collect();
                fx.bcx.ins().nop();
                returns
            } else {
                smallvec![]
            }
        }
    }
}

pub(crate) fn lib_call_arg_param(tcx: TyCtxt<'_>, ty: Type, is_signed: bool) -> AbiParam {
    let param = AbiParam::new(ty);
    if ty.is_int() && u64::from(ty.bits()) < tcx.data_layout.pointer_size().bits() {
        match (&tcx.sess.target.arch, tcx.sess.target.is_like_darwin) {
            (Arch::X86_64, _) | (Arch::AArch64, true) => match (ty, is_signed) {
                (types::I8 | types::I16, true) => param.sext(),
                (types::I8 | types::I16, false) => param.uext(),
                _ => param,
            },
            (Arch::AArch64, _) => param,
            (Arch::RiscV64, _) => match (ty, is_signed) {
                (types::I32, _) | (_, true) => param.sext(),
                _ => param.uext(),
            },
            (Arch::S390x, _) => {
                if is_signed {
                    param.sext()
                } else {
                    param.uext()
                }
            }
            (arch, _) => unimplemented!("{arch:?}"),
        }
    } else {
        param
    }
}
