//! Always-property capability derivation and closed-graph check.
//!
//! Every capability bit is `polyasm_format::record::always`'s, the one
//! vocabulary this compiler's certificate checks read as well, and the
//! derivation over a semantic body is `polyasm::object::domain::capability`'s.
//! This module reads the property a source marker names.

pub(super) use polyasm::object::domain::capability::derive_always_scopes;
pub(super) use polyasm_format::record::ALWAYS_CAPABILITY_MASK as CAPABILITY_MASK;
use polyasm_format::record::always::{
    ACCELERATOR, CUDA, DPA, EBPF, LINUX_SAFE, LLVM_BRIDGE, NATIVE, P4, PTX, RUST, STATIC_CLOCK,
    STATIC_MEMORY, STATIC_MEMORY_UPPER, VERILOG, WASM,
};
pub(super) use polyasm_format::record::always::{
    LINUX_SAFE as CAP_LINUX_SAFE, POINTER_ACCELERATORS as CAP_POINTER_ACCELERATORS, XDP as CAP_XDP,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Symbol;

pub(super) fn property_capability(tcx: TyCtxt<'_>, property: Ty<'_>) -> Option<u32> {
    let ty::Adt(definition, _) = property.kind() else {
        return None;
    };
    let def_id = definition.did();
    for (name, capability) in [
        ("polyasm_property_accelerator", ACCELERATOR),
        ("polyasm_property_cuda", CUDA),
        ("polyasm_property_dpa", DPA),
        ("polyasm_property_ebpf", EBPF),
        ("polyasm_property_llvm_bridge", LLVM_BRIDGE),
        ("polyasm_property_linux_safe", LINUX_SAFE),
        ("polyasm_property_native", NATIVE),
        ("polyasm_property_p4", P4),
        ("polyasm_property_ptx", PTX),
        ("polyasm_property_rust", RUST),
        ("polyasm_property_static_clock", STATIC_CLOCK),
        ("polyasm_property_static_memory", STATIC_MEMORY),
        ("polyasm_property_static_memory_upper", STATIC_MEMORY_UPPER),
        ("polyasm_property_verilog", VERILOG),
        ("polyasm_property_wasm", WASM),
        ("polyasm_property_xdp", CAP_XDP),
    ] {
        if tcx.is_diagnostic_item(Symbol::intern(name), def_id) {
            return Some(capability);
        }
    }
    None
}
