//! Symbolic interchange-to-POBJ relocation and record conversion.

use cranelift_codegen::ir::LibCall;
use rustc_middle::ty::TyCtxt;

use super::super::model::{PortableImage, Relocation, RelocationTarget};
use super::abi;
use crate::driver::polyasm::object;

impl PortableImage {
    pub(crate) fn relocatable(&self, tcx: TyCtxt<'_>) -> object::InterchangeFragment {
        let functions = self
            .functions
            .iter()
            .map(|function| object::Function {
                alignment: function.alignment,
                call_conv: abi::call_conv(function.signature.call_conv)
                    .unwrap_or_else(|error| tcx.dcx().fatal(error)),
                callable_kind: function.callable_kind,
                code: function.code.clone(),
                compiler_instance: function.compiler_instance,
                linkage: abi::linkage(function.linkage),
                name: function.name.clone(),
                params: abi::abi_params(&function.signature.params)
                    .unwrap_or_else(|error| tcx.dcx().fatal(error)),
                relocations: function
                    .relocs
                    .iter()
                    .map(|relocation| object_relocation(tcx, relocation))
                    .collect(),
                returns: abi::abi_params(&function.signature.returns)
                    .unwrap_or_else(|error| tcx.dcx().fatal(error)),
            })
            .collect();
        let data = self
            .data
            .iter()
            .map(|data| object::Data {
                alignment: data.alignment,
                bytes: data.bytes.clone(),
                linkage: abi::linkage(data.linkage),
                name: data.name.clone(),
                relocations: data
                    .relocs
                    .iter()
                    .map(|relocation| object_relocation(tcx, relocation))
                    .collect(),
                tls: data.tls,
                writable: data.writable,
            })
            .collect();
        object::InterchangeFragment { data, functions }
    }
}

fn object_relocation(tcx: TyCtxt<'_>, relocation: &Relocation) -> object::Relocation {
    let kind = abi::relocation_kind(relocation.kind).unwrap_or_else(|error| tcx.dcx().fatal(error));
    let target = match &relocation.target {
        RelocationTarget::PacketDataRange { length } => {
            object::RelocationTarget::PacketDataRange { length: *length }
        }
        RelocationTarget::Function(name)
            if name == crate::driver::polyasm::MEMORY_COPY8_INTRINSIC =>
        {
            object::RelocationTarget::MemoryCopy8
        }
        RelocationTarget::Function(name)
            if name == crate::driver::polyasm::MEMORY_COPY64_INTRINSIC =>
        {
            object::RelocationTarget::MemoryCopy64
        }
        RelocationTarget::Function(name)
            if name.starts_with("__rustc_polyasm_packet_data_load8_abs.") =>
        {
            let displacement = name
                .strip_prefix("__rustc_polyasm_packet_data_load8_abs.")
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or_else(|| {
                    tcx.dcx().fatal("invalid PacketDataLoad8Abs instruction displacement")
                });
            object::RelocationTarget::PacketDataLoad8Abs { displacement }
        }
        RelocationTarget::Function(name) if name == "__rustc_polyasm_packet_data_load8_ind" => {
            object::RelocationTarget::PacketDataLoad8Ind
        }
        RelocationTarget::Function(name)
            if name.starts_with("__rustc_polyasm_packet_data_load16be_abs.") =>
        {
            let displacement = name
                .strip_prefix("__rustc_polyasm_packet_data_load16be_abs.")
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or_else(|| {
                    tcx.dcx().fatal("invalid PacketDataLoad16BeAbs instruction displacement")
                });
            object::RelocationTarget::PacketDataLoad16BeAbs { displacement }
        }
        RelocationTarget::Function(name) if name == "__rustc_polyasm_packet_data_load16be_ind" => {
            object::RelocationTarget::PacketDataLoad16BeInd
        }
        RelocationTarget::Function(name)
            if name.starts_with("__rustc_polyasm_packet_data_load32be_abs.") =>
        {
            let displacement = name
                .strip_prefix("__rustc_polyasm_packet_data_load32be_abs.")
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or_else(|| {
                    tcx.dcx().fatal("invalid PacketDataLoad32BeAbs instruction displacement")
                });
            object::RelocationTarget::PacketDataLoad32BeAbs { displacement }
        }
        RelocationTarget::Function(name) if name == "__rustc_polyasm_packet_data_load32be_ind" => {
            object::RelocationTarget::PacketDataLoad32BeInd
        }
        RelocationTarget::Function(name)
            if name.starts_with("__rustc_polyasm_packet_mac_load8_abs.") =>
        {
            let displacement = name
                .strip_prefix("__rustc_polyasm_packet_mac_load8_abs.")
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or_else(|| {
                    tcx.dcx().fatal("invalid PacketMacLoad8Abs instruction displacement")
                });
            object::RelocationTarget::PacketMacLoad8Abs { displacement }
        }
        RelocationTarget::Function(name) if name == "__rustc_polyasm_packet_mac_load8_ind" => {
            object::RelocationTarget::PacketMacLoad8Ind
        }
        RelocationTarget::Function(name)
            if name.starts_with("__rustc_polyasm_packet_mac_load16be_abs.") =>
        {
            let displacement = name
                .strip_prefix("__rustc_polyasm_packet_mac_load16be_abs.")
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or_else(|| {
                    tcx.dcx().fatal("invalid PacketMacLoad16BeAbs instruction displacement")
                });
            object::RelocationTarget::PacketMacLoad16BeAbs { displacement }
        }
        RelocationTarget::Function(name) if name == "__rustc_polyasm_packet_mac_load16be_ind" => {
            object::RelocationTarget::PacketMacLoad16BeInd
        }
        RelocationTarget::Function(name)
            if name.starts_with("__rustc_polyasm_packet_mac_load32be_abs.") =>
        {
            let displacement = name
                .strip_prefix("__rustc_polyasm_packet_mac_load32be_abs.")
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or_else(|| {
                    tcx.dcx().fatal("invalid PacketMacLoad32BeAbs instruction displacement")
                });
            object::RelocationTarget::PacketMacLoad32BeAbs { displacement }
        }
        RelocationTarget::Function(name) if name == "__rustc_polyasm_packet_mac_load32be_ind" => {
            object::RelocationTarget::PacketMacLoad32BeInd
        }
        RelocationTarget::Function(name)
            if name.starts_with("__rustc_polyasm_packet_network_load8_abs.") =>
        {
            let displacement = name
                .strip_prefix("__rustc_polyasm_packet_network_load8_abs.")
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or_else(|| {
                    tcx.dcx().fatal("invalid PacketNetworkLoad8Abs instruction displacement")
                });
            object::RelocationTarget::PacketNetworkLoad8Abs { displacement }
        }
        RelocationTarget::Function(name) if name == "__rustc_polyasm_packet_network_load8_ind" => {
            object::RelocationTarget::PacketNetworkLoad8Ind
        }
        RelocationTarget::Function(name)
            if name.starts_with("__rustc_polyasm_packet_network_load16be_abs.") =>
        {
            let displacement = name
                .strip_prefix("__rustc_polyasm_packet_network_load16be_abs.")
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or_else(|| {
                    tcx.dcx().fatal("invalid PacketNetworkLoad16BeAbs instruction displacement")
                });
            object::RelocationTarget::PacketNetworkLoad16BeAbs { displacement }
        }
        RelocationTarget::Function(name)
            if name == "__rustc_polyasm_packet_network_load16be_ind" =>
        {
            object::RelocationTarget::PacketNetworkLoad16BeInd
        }
        RelocationTarget::Function(name)
            if name.starts_with("__rustc_polyasm_packet_network_load32be_abs.") =>
        {
            let displacement = name
                .strip_prefix("__rustc_polyasm_packet_network_load32be_abs.")
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or_else(|| {
                    tcx.dcx().fatal("invalid PacketNetworkLoad32BeAbs instruction displacement")
                });
            object::RelocationTarget::PacketNetworkLoad32BeAbs { displacement }
        }
        RelocationTarget::Function(name)
            if name == "__rustc_polyasm_packet_network_load32be_ind" =>
        {
            object::RelocationTarget::PacketNetworkLoad32BeInd
        }
        RelocationTarget::Function(name) if name == "__rustc_polyasm_packet_data_start" => {
            object::RelocationTarget::PacketDataStart
        }
        RelocationTarget::Function(name) if name == "__rustc_polyasm_packet_data_end" => {
            object::RelocationTarget::PacketDataEnd
        }
        RelocationTarget::Function(name) => object::RelocationTarget::Function(name.clone()),
        RelocationTarget::Data(name) => object::RelocationTarget::Data(name.clone()),
        RelocationTarget::SelfFunctionOffset(offset) => {
            object::RelocationTarget::SelfFunctionOffset(*offset)
        }
        RelocationTarget::LibCall(LibCall::Memcmp) => object::RelocationTarget::MemoryCompare,
        RelocationTarget::LibCall(LibCall::Memcpy) => object::RelocationTarget::MemoryCopy,
        RelocationTarget::LibCall(LibCall::Memmove) => object::RelocationTarget::MemoryMove,
        RelocationTarget::LibCall(LibCall::Memset) => object::RelocationTarget::MemorySet,
        RelocationTarget::LibCall(libcall) => {
            object::RelocationTarget::Unresolved(format!("Cranelift libcall `{libcall}`"))
        }
        RelocationTarget::KnownSymbol(name) => {
            object::RelocationTarget::Unresolved(format!("Cranelift known symbol `{name}`"))
        }
    };
    object::Relocation { addend: relocation.addend, kind, offset: relocation.offset, target }
}
