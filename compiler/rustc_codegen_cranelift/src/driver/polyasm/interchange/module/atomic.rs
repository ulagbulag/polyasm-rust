//! The Cranelift side of the closed atomic helpers.
//!
//! Cranelift's interchange backend selects zero atomic instructions, so every
//! atomic of a function becomes a call of the closed helper
//! `polyasm::object::vocabulary::ClosedAtomic` names, and the helper's body is
//! the one PolyASM atomic row that type assembles. This module reads the
//! Cranelift graph: which instructions are atomics, which helper each one
//! reaches, the signature a helper is declared with, and the call that takes
//! an atomic's place.

use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{
    AbiParam, AtomicRmwOp, Endianness, FuncRef, Function, Inst, InstBuilder, InstructionData,
    MemFlagsData, Opcode, Signature, Type, Value, types,
};
use cranelift_codegen::isa::CallConv;
use polyasm::object::vocabulary::ClosedAtomic;
use polyasm_format::executable::{AtomicOperation, AtomicSpelling};

/// The widest atomic access `polyasm-unknown-unknown` publishes, in bits.
///
/// The target takes `max_atomic_width: Some(64)` from the Wasm base options,
/// so the helpers spell one, two, four and eight byte accesses, eight bytes
/// being the widest.
const MAX_ATOMIC_BITS: u32 = 64;

/// Names the closed atomic of every Cranelift atomic, in layout order.
///
/// Selection sees zero atomics, so a width outside the namespace is reported
/// here in place of reaching the selector's capability miss.
pub(super) fn sites(function: &Function) -> Result<Vec<(Inst, ClosedAtomic)>, String> {
    function
        .layout
        .blocks()
        .flat_map(|block| function.layout.block_insts(block))
        .filter_map(|inst| Some((inst, closed_atomic(function, inst)?)))
        .map(|(inst, atomic)| Ok((inst, atomic?)))
        .collect()
}

/// Answers the signature one helper is declared and called with.
pub(super) fn signature(atomic: ClosedAtomic, call_conv: CallConv, pointer: Type) -> Signature {
    let mut signature = Signature::new(call_conv);
    let (operation, spelling) = match atomic {
        ClosedAtomic::Access(operation, spelling) => (operation, spelling),
        ClosedAtomic::Fence => return signature,
    };
    let value = accessed_type(spelling);
    match operation {
        AtomicOperation::CompareAndSwap => {
            signature.params.push(AbiParam::new(value));
            signature.params.push(AbiParam::new(pointer));
            signature.params.push(AbiParam::new(value));
        }
        AtomicOperation::Load => {
            signature.params.push(AbiParam::new(pointer));
        }
        _ => {
            signature.params.push(AbiParam::new(pointer));
            signature.params.push(AbiParam::new(value));
        }
    }
    if operation != AtomicOperation::Store {
        signature.returns.push(AbiParam::new(value));
    }
    signature
}

/// Replaces one atomic instruction with a call of its closed helper.
///
/// # Panics
///
/// Panics for an instruction [`sites`] names zero helpers for.
pub(super) fn call(function: &mut Function, inst: Inst, callee: FuncRef) {
    let arguments: Vec<Value> = match function.dfg.insts[inst] {
        InstructionData::AtomicCas { args, .. } => {
            vec![args[1], args[0], args[2]]
        }
        InstructionData::AtomicRmw { args, .. } => vec![args[0], args[1]],
        InstructionData::LoadNoOffset { arg, .. } => vec![arg],
        InstructionData::StoreNoOffset { args, .. } => vec![args[1], args[0]],
        InstructionData::NullAry { .. } => Vec::new(),
        // `sites` names exactly the five shapes above, and skipping one here
        // would leave an atomic in the graph for selection to miss.
        data => panic!(
            "the closed atomic rewrite calls zero helpers for a {} instruction",
            data.opcode()
        ),
    };
    let mut cursor = FuncCursor::new(function).at_inst(inst);
    cursor.use_srcloc(inst);
    let call = cursor.ins().call(callee, &arguments);
    function.dfg.replace_with_aliases(inst, call);
    function.layout.remove_inst(inst);
}

fn accessed_type(spelling: AtomicSpelling) -> Type {
    match spelling.bytes() {
        1 => types::I8,
        2 => types::I16,
        4 => types::I32,
        _ => types::I64,
    }
}

fn closed_atomic(function: &Function, inst: Inst) -> Option<Result<ClosedAtomic, String>> {
    let (flags, operation) = match function.dfg.insts[inst] {
        InstructionData::AtomicCas { flags, .. } => (flags, AtomicOperation::CompareAndSwap),
        InstructionData::AtomicRmw { flags, op, .. } => (flags, updated(op)),
        InstructionData::LoadNoOffset { flags, opcode: Opcode::AtomicLoad, .. } => {
            (flags, AtomicOperation::Load)
        }
        InstructionData::StoreNoOffset { flags, opcode: Opcode::AtomicStore, .. } => {
            (flags, AtomicOperation::Store)
        }
        InstructionData::NullAry { opcode: Opcode::Fence, .. } => {
            return Some(Ok(ClosedAtomic::Fence));
        }
        _ => return None,
    };
    // A store alone leaves its accessed value unnamed by a result, and every
    // other atomic returns exactly the value it accessed.
    let ty = match operation {
        AtomicOperation::Store => function.dfg.value_type(function.dfg.inst_args(inst)[0]),
        _ => function.dfg.value_type(function.dfg.first_result(inst)),
    };
    Some(match spelling(ty, function.dfg.mem_flags[flags]) {
        Some(spelling) => Ok(ClosedAtomic::Access(operation, spelling)),
        None => Err(format!(
            "PolyASM spells zero {ty} atomics; the widest atomic is {MAX_ATOMIC_BITS} bits"
        )),
    })
}

fn spelling(ty: Type, flags: MemFlagsData) -> Option<AtomicSpelling> {
    // A `.poly` image stores every wide value little-endian, so an access
    // that leaves the byte order open selects the little-endian instruction
    // and an explicitly big-endian one alone selects the big-endian one.
    match (ty, flags.endianness(Endianness::Little)) {
        (types::I8, _) => Some(AtomicSpelling::I8),
        (types::I16, Endianness::Big) => Some(AtomicSpelling::I16Big),
        (types::I16, Endianness::Little) => Some(AtomicSpelling::I16Little),
        (types::I32, Endianness::Big) => Some(AtomicSpelling::I32Big),
        (types::I32, Endianness::Little) => Some(AtomicSpelling::I32Little),
        (types::I64, Endianness::Big) => Some(AtomicSpelling::I64Big),
        (types::I64, Endianness::Little) => Some(AtomicSpelling::I64Little),
        _ => None,
    }
}

const fn updated(op: AtomicRmwOp) -> AtomicOperation {
    match op {
        AtomicRmwOp::Add => AtomicOperation::Add,
        AtomicRmwOp::And => AtomicOperation::And,
        AtomicRmwOp::Nand => AtomicOperation::Nand,
        AtomicRmwOp::Or => AtomicOperation::Or,
        AtomicRmwOp::Smax => AtomicOperation::SignedMax,
        AtomicRmwOp::Smin => AtomicOperation::SignedMin,
        AtomicRmwOp::Sub => AtomicOperation::Sub,
        AtomicRmwOp::Umax => AtomicOperation::UnsignedMax,
        AtomicRmwOp::Umin => AtomicOperation::UnsignedMin,
        AtomicRmwOp::Xchg => AtomicOperation::Exchange,
        AtomicRmwOp::Xor => AtomicOperation::Xor,
    }
}
