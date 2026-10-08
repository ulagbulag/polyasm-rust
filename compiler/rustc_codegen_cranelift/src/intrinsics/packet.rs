//! Typed Rust packet intrinsics enter the interchange as closed instruction sites.
//!
//! The symbolic site has zero function bodies and stays apart from host
//! imports. Final executable translation consumes it and writes exactly one
//! packet opcode. Calls remain effectful in Cranelift, preserving a failed
//! load's whole-guest termination even when the caller leaves its answer
//! unread.

use cranelift_codegen::isa::CallConv;
use rustc_middle::ty::consts::ConstExt;
use rustc_span::{Spanned, Symbol, sym};

use crate::prelude::*;

pub(super) struct Call<'a, 'm, 'clif, 'tcx> {
    pub(super) fx: &'a mut FunctionCx<'m, 'clif, 'tcx>,
    pub(super) instance: Instance<'tcx>,
    pub(super) intrinsic: Symbol,
    pub(super) args: &'a [Spanned<mir::Operand<'tcx>>],
    pub(super) destination: CPlace<'tcx>,
}

pub(super) fn codegen(call: Call<'_, '_, '_, '_>) -> bool {
    let Call { fx, instance, intrinsic, args, destination } = call;
    match intrinsic {
        sym::packet_data_range => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            let length = instance.args.const_at(0).try_to_target_usize(fx.tcx).unwrap();
            let length = u32::try_from(length).unwrap_or_else(|_| {
                fx.tcx.dcx().span_fatal(
                    args[0].span,
                    "packet prefix exceeds the instruction length (u32::MAX)",
                )
            });
            intrinsic_args!(fx, args => (operands); intrinsic);
            // Scalar-pair ABI order and the named Rust field order differ at
            // times. Project from the evaluated aggregate layout even for a pair
            // passed in registers; the ordinary mem2reg pass forwards it.
            let layout = operands.layout();
            let (pointer, _) = operands.force_stack(fx);
            let operands = CValue::by_ref(pointer, layout);
            let ty::Adt(operand_definition, _) = operands.layout().ty.kind() else {
                bug!("PacketDataRange operands are not a named struct");
            };
            let window_field = operand_definition
                .non_enum_variant()
                .fields
                .iter_enumerated()
                .find(|(_, field)| field.name.as_str() == "window")
                .map(|(index, _)| index)
                .unwrap();
            let window = operands.value_field(fx, window_field);
            let ty::Adt(window_definition, _) = window.layout().ty.kind() else {
                bug!("PacketWindow is not a named struct");
            };
            let start_field = window_definition
                .non_enum_variant()
                .fields
                .iter_enumerated()
                .find(|(_, field)| field.name.as_str() == "start")
                .map(|(index, _)| index)
                .unwrap();
            let end_field = window_definition
                .non_enum_variant()
                .fields
                .iter_enumerated()
                .find(|(_, field)| field.name.as_str() == "end")
                .map(|(index, _)| index)
                .unwrap();
            let start = window.value_field(fx, start_field).load_scalar(fx);
            let end = window.value_field(fx, end_field).load_scalar(fx);
            let length = fx.bcx.ins().iconst(types::I32, i64::from(length));
            let signature = Signature {
                params: vec![
                    AbiParam::new(types::I64),
                    AbiParam::new(types::I64),
                    AbiParam::new(types::I32),
                ],
                returns: vec![AbiParam::new(types::I8)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx
                .module
                .declare_function("__polyasm_packet_data_range", Linkage::Import, &signature)
                .unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[start, end, length]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_data_load8_abs => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.load_scalar(fx);
            let displacement = instance.args.const_at(0).to_leaf().to_i32();
            let name = format!("__rustc_polyasm_packet_data_load8_abs.{displacement}");
            let signature = Signature {
                params: vec![AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_data_load8_ind => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.value_field(fx, FieldIdx::ZERO).load_scalar(fx);
            let offset = operands.value_field(fx, FieldIdx::new(1)).load_scalar(fx);
            let name = "__rustc_polyasm_packet_data_load8_ind";
            let signature = Signature {
                params: vec![AbiParam::new(types::I64), AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context, offset]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_data_load16be_abs => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.load_scalar(fx);
            let displacement = instance.args.const_at(0).to_leaf().to_i32();
            let name = format!("__rustc_polyasm_packet_data_load16be_abs.{displacement}");
            let signature = Signature {
                params: vec![AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_data_load16be_ind => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.value_field(fx, FieldIdx::ZERO).load_scalar(fx);
            let offset = operands.value_field(fx, FieldIdx::new(1)).load_scalar(fx);
            let name = "__rustc_polyasm_packet_data_load16be_ind";
            let signature = Signature {
                params: vec![AbiParam::new(types::I64), AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context, offset]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_data_load32be_abs => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.load_scalar(fx);
            let displacement = instance.args.const_at(0).to_leaf().to_i32();
            let name = format!("__rustc_polyasm_packet_data_load32be_abs.{displacement}");
            let signature = Signature {
                params: vec![AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_data_load32be_ind => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.value_field(fx, FieldIdx::ZERO).load_scalar(fx);
            let offset = operands.value_field(fx, FieldIdx::new(1)).load_scalar(fx);
            let name = "__rustc_polyasm_packet_data_load32be_ind";
            let signature = Signature {
                params: vec![AbiParam::new(types::I64), AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context, offset]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_mac_load8_abs => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.load_scalar(fx);
            let displacement = instance.args.const_at(0).to_leaf().to_i32();
            let name = format!("__rustc_polyasm_packet_mac_load8_abs.{displacement}");
            let signature = Signature {
                params: vec![AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_mac_load8_ind => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.value_field(fx, FieldIdx::ZERO).load_scalar(fx);
            let offset = operands.value_field(fx, FieldIdx::new(1)).load_scalar(fx);
            let name = "__rustc_polyasm_packet_mac_load8_ind";
            let signature = Signature {
                params: vec![AbiParam::new(types::I64), AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context, offset]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_mac_load16be_abs => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.load_scalar(fx);
            let displacement = instance.args.const_at(0).to_leaf().to_i32();
            let name = format!("__rustc_polyasm_packet_mac_load16be_abs.{displacement}");
            let signature = Signature {
                params: vec![AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_mac_load16be_ind => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.value_field(fx, FieldIdx::ZERO).load_scalar(fx);
            let offset = operands.value_field(fx, FieldIdx::new(1)).load_scalar(fx);
            let name = "__rustc_polyasm_packet_mac_load16be_ind";
            let signature = Signature {
                params: vec![AbiParam::new(types::I64), AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context, offset]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_mac_load32be_abs => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.load_scalar(fx);
            let displacement = instance.args.const_at(0).to_leaf().to_i32();
            let name = format!("__rustc_polyasm_packet_mac_load32be_abs.{displacement}");
            let signature = Signature {
                params: vec![AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_mac_load32be_ind => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.value_field(fx, FieldIdx::ZERO).load_scalar(fx);
            let offset = operands.value_field(fx, FieldIdx::new(1)).load_scalar(fx);
            let name = "__rustc_polyasm_packet_mac_load32be_ind";
            let signature = Signature {
                params: vec![AbiParam::new(types::I64), AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context, offset]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_network_load8_abs => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.load_scalar(fx);
            let displacement = instance.args.const_at(0).to_leaf().to_i32();
            let name = format!("__rustc_polyasm_packet_network_load8_abs.{displacement}");
            let signature = Signature {
                params: vec![AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_network_load8_ind => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.value_field(fx, FieldIdx::ZERO).load_scalar(fx);
            let offset = operands.value_field(fx, FieldIdx::new(1)).load_scalar(fx);
            let name = "__rustc_polyasm_packet_network_load8_ind";
            let signature = Signature {
                params: vec![AbiParam::new(types::I64), AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context, offset]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_network_load16be_abs => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.load_scalar(fx);
            let displacement = instance.args.const_at(0).to_leaf().to_i32();
            let name = format!("__rustc_polyasm_packet_network_load16be_abs.{displacement}");
            let signature = Signature {
                params: vec![AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_network_load16be_ind => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.value_field(fx, FieldIdx::ZERO).load_scalar(fx);
            let offset = operands.value_field(fx, FieldIdx::new(1)).load_scalar(fx);
            let name = "__rustc_polyasm_packet_network_load16be_ind";
            let signature = Signature {
                params: vec![AbiParam::new(types::I64), AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context, offset]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_network_load32be_abs => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.load_scalar(fx);
            let displacement = instance.args.const_at(0).to_leaf().to_i32();
            let name = format!("__rustc_polyasm_packet_network_load32be_abs.{displacement}");
            let signature = Signature {
                params: vec![AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_network_load32be_ind => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.value_field(fx, FieldIdx::ZERO).load_scalar(fx);
            let offset = operands.value_field(fx, FieldIdx::new(1)).load_scalar(fx);
            let name = "__rustc_polyasm_packet_network_load32be_ind";
            let signature = Signature {
                params: vec![AbiParam::new(types::I64), AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context, offset]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_data_start => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.load_scalar(fx);
            let name = "__rustc_polyasm_packet_data_start";
            let signature = Signature {
                params: vec![AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        sym::packet_data_end => {
            assert!(crate::driver::polyasm::is_target(fx.tcx.sess));
            assert_eq!(fx.pointer_type, types::I64);
            intrinsic_args!(fx, args => (operands); intrinsic);
            let context = operands.load_scalar(fx);
            let name = "__rustc_polyasm_packet_data_end";
            let signature = Signature {
                params: vec![AbiParam::new(types::I64)],
                returns: vec![AbiParam::new(types::I64)],
                call_conv: CallConv::PreserveAll,
            };
            let function = fx.module.declare_function(&name, Linkage::Import, &signature).unwrap();
            let callee = fx.module.declare_func_in_func(function, fx.bcx.func);
            let instruction = fx.bcx.ins().call(callee, &[context]);
            let answer = fx.bcx.inst_results(instruction)[0];
            destination.write_cvalue(fx, CValue::by_val(answer, destination.layout()));
        }
        _ => return false,
    }
    true
}
