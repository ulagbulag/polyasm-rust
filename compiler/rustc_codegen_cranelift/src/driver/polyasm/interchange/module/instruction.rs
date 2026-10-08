//! Pre-ISLE normalization for wide operations the interchange backend represents as register pairs.

use cranelift_codegen::Context;
use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{InstBuilder, Opcode, types};

pub(super) fn normalize_wide_instructions(context: &mut Context) {
    normalize_wide_negations(context);
    normalize_wide_selects(context);
    normalize_wide_popcounts(context);
    normalize_wide_conditions(context);
    normalize_wide_shift_amounts(context);
}

/// Narrows an `i128` shift or rotation amount to its low `i64` half.
///
/// The interchange backend selects a shift of any width by a one-register
/// amount, and an `i128` amount occupies two registers, so the selector meets
/// a register pair where it reads one register. `u128 << u128`,
/// `i128 >> i128` and `u64 >> (u128 & 63)` all reach it in that form. A shift
/// or rotation reads its amount modulo the bit width of the shifted value,
/// at most 128, which divides `2^64`, so the low half names the same amount.
fn normalize_wide_shift_amounts(context: &mut Context) {
    let shifts = context
        .func
        .layout
        .blocks()
        .flat_map(|block| context.func.layout.block_insts(block))
        .filter(|&inst| {
            matches!(
                context.func.dfg.insts[inst].opcode(),
                Opcode::Ishl | Opcode::Ushr | Opcode::Sshr | Opcode::Rotl | Opcode::Rotr
            )
        })
        .filter_map(|inst| {
            let amount = *context.func.dfg.inst_args(inst).get(1)?;
            (context.func.dfg.value_type(amount) == types::I128).then_some((inst, amount))
        })
        .collect::<Vec<_>>();

    for (inst, amount) in shifts {
        let low = FuncCursor::new(&mut context.func).at_inst(inst).ins().isplit(amount).0;
        context.func.dfg.inst_args_mut(inst)[1] = low;
    }
}

/// Rewrites an `i128` negation as the subtraction of its operand from zero.
///
/// The interchange backend selects `isub` over an `i128` register pair and
/// holds its negation rule at `i64` and below, so `-x` on an `i128` reaches the
/// selector as an `ineg` it leaves unselected, and the relocatable backend
/// keeps only the declaration of the body that holds it. Two's complement
/// negation is the subtraction from zero bit for bit, so the rewrite keeps the
/// answer and selects through the wide subtraction rule.
fn normalize_wide_negations(context: &mut Context) {
    let negations = context
        .func
        .layout
        .blocks()
        .flat_map(|block| context.func.layout.block_insts(block))
        .filter(|&inst| context.func.dfg.insts[inst].opcode() == Opcode::Ineg)
        .filter_map(|inst| {
            let result = context.func.dfg.first_result(inst);
            (context.func.dfg.value_type(result) == types::I128).then(|| {
                let input = context.func.dfg.inst_args(inst)[0];
                (inst, input, result)
            })
        })
        .collect::<Vec<_>>();

    for (inst, input, result) in negations {
        let replacement = {
            let mut cursor = FuncCursor::new(&mut context.func).at_inst(inst);
            let zero = cursor.ins().iconst(types::I64, 0);
            let zero = cursor.ins().iconcat(zero, zero);
            cursor.ins().isub(zero, input)
        };
        context.func.dfg.clear_results(inst);
        context.func.dfg.change_to_alias(result, replacement);
        context.func.layout.remove_inst(inst);
    }
}

/// Rewrites an `i128` selection as one selection per `i64` half.
///
/// The interchange backend selects a choice between two values of one register
/// and holds its selection rule at `i64` and below, so a choice between two
/// `i128` pairs reaches the selector as a `select` it leaves unselected. Both
/// halves are chosen by the same condition, so choosing each half on its own
/// and joining the two answers is the same pair bit for bit.
fn normalize_wide_selects(context: &mut Context) {
    let selects = context
        .func
        .layout
        .blocks()
        .flat_map(|block| context.func.layout.block_insts(block))
        .filter(|&inst| context.func.dfg.insts[inst].opcode() == Opcode::Select)
        .filter_map(|inst| {
            let result = context.func.dfg.first_result(inst);
            (context.func.dfg.value_type(result) == types::I128).then(|| {
                let args = context.func.dfg.inst_args(inst);
                (inst, args[0], args[1], args[2], result)
            })
        })
        .collect::<Vec<_>>();

    for (inst, condition, taken, other, result) in selects {
        let replacement = {
            let mut cursor = FuncCursor::new(&mut context.func).at_inst(inst);
            let (taken_low, taken_high) = cursor.ins().isplit(taken);
            let (other_low, other_high) = cursor.ins().isplit(other);
            let low = cursor.ins().select(condition, taken_low, other_low);
            let high = cursor.ins().select(condition, taken_high, other_high);
            cursor.ins().iconcat(low, high)
        };
        context.func.dfg.clear_results(inst);
        context.func.dfg.change_to_alias(result, replacement);
        context.func.layout.remove_inst(inst);
    }
}

/// Lowers an `i128` population count into the sum of its two `i64` halves.
///
/// The interchange backend represents an `i128` with two integer registers and selects wide
/// arithmetic and comparisons directly, but its population-count rule accepts
/// at most `i64`. Cranelift reports that missing rule as `Unsupported`; the
/// relocatable backend would then retain only the compiler declaration and a
/// final image using `u128::checked_pow` would link a bodyless record. Splitting
/// the input is exact because the halves have disjoint bits, and zero-extension
/// restores the instruction's `i128` result type.
fn normalize_wide_popcounts(context: &mut Context) {
    let operations = context
        .func
        .layout
        .blocks()
        .flat_map(|block| context.func.layout.block_insts(block))
        .filter(|&inst| context.func.dfg.insts[inst].opcode() == Opcode::Popcnt)
        .filter_map(|inst| {
            let result = context.func.dfg.first_result(inst);
            (context.func.dfg.value_type(result) == types::I128).then(|| {
                let input = context.func.dfg.inst_args(inst)[0];
                (inst, input, result)
            })
        })
        .collect::<Vec<_>>();

    for (inst, input, result) in operations {
        let replacement = {
            let mut cursor = FuncCursor::new(&mut context.func).at_inst(inst);
            let (low, high) = cursor.ins().isplit(input);
            let low = cursor.ins().popcnt(low);
            let high = cursor.ins().popcnt(high);
            let count = cursor.ins().iadd(low, high);
            cursor.ins().uextend(types::I128, count)
        };
        context.func.dfg.clear_results(inst);
        context.func.dfg.change_to_alias(result, replacement);
        context.func.layout.remove_inst(inst);
    }
}

fn normalize_wide_conditions(context: &mut Context) {
    let conditions = context
        .func
        .layout
        .blocks()
        .flat_map(|block| context.func.layout.block_insts(block))
        .filter_map(|inst| {
            matches!(
                context.func.dfg.insts[inst].opcode(),
                Opcode::Brif | Opcode::Trapz | Opcode::Trapnz | Opcode::Select
            )
            .then(|| context.func.dfg.inst_args(inst).first().copied().map(|value| (inst, value)))
            .flatten()
        })
        .filter(|(_, value)| context.func.dfg.value_type(*value) == types::I128)
        .collect::<Vec<_>>();

    for (inst, value) in conditions {
        let condition = FuncCursor::new(&mut context.func).at_inst(inst).ins().icmp_imm_u(
            IntCC::NotEqual,
            value,
            0,
        );
        context.func.dfg.inst_args_mut(inst)[0] = condition;
    }
}

/// Lowers `i128` rotations and byte swaps into operations on the two `i64`
/// halves.
///
/// The interchange backend selects wide shifts, arithmetic and comparisons
/// directly, but its rotation and byte-swap rules accept at most `i64`.
/// Cranelift reports that missing rule as `Unsupported`, and its optimizer
/// itself forms an `i128` rotation out of a shift pair, so `u128::rotate_left`,
/// `(x << 64) | (x >> 64)` and `u128::to_be_bytes` all reach the selector as
/// such an instruction. This rewrite runs on the optimized function after the
/// selection miss: every shift pair it leaves joins two distinct halves, so a
/// second optimization keeps them. A rotation by `n` first swaps the
/// halves when `n & 64` is set, then shifts each half left by `n & 63` and
/// fills it from the other half shifted right by `64 - (n & 63)`, written as
/// a shift by one and then by `63 - (n & 63)` so a zero amount carries zero
/// bits. A right rotation by `n` is the left rotation by `-n`. A byte swap
/// swaps each half and exchanges them. Answers whether anything was rewritten.
pub(super) fn normalize_wide_permutations(context: &mut Context) -> bool {
    let operations = context
        .func
        .layout
        .blocks()
        .flat_map(|block| context.func.layout.block_insts(block))
        .filter(|&inst| {
            matches!(
                context.func.dfg.insts[inst].opcode(),
                Opcode::Rotl | Opcode::Rotr | Opcode::Bswap
            )
        })
        .filter(|&inst| {
            context.func.dfg.value_type(context.func.dfg.first_result(inst)) == types::I128
        })
        .collect::<Vec<_>>();

    for &inst in &operations {
        let opcode = context.func.dfg.insts[inst].opcode();
        let args = context.func.dfg.inst_args(inst).to_vec();
        let result = context.func.dfg.first_result(inst);
        let replacement = {
            let mut cursor = FuncCursor::new(&mut context.func).at_inst(inst);
            let (low, high) = cursor.ins().isplit(args[0]);
            if opcode == Opcode::Bswap {
                let swapped_low = cursor.ins().bswap(high);
                let swapped_high = cursor.ins().bswap(low);
                cursor.ins().iconcat(swapped_low, swapped_high)
            } else {
                let amount = args[1];
                let amount = match cursor.func.dfg.value_type(amount) {
                    types::I128 => cursor.ins().isplit(amount).0,
                    types::I64 => amount,
                    _ => cursor.ins().uextend(types::I64, amount),
                };
                let amount =
                    if opcode == Opcode::Rotr { cursor.ins().ineg(amount) } else { amount };
                let amount = cursor.ins().band_imm_u(amount, 127);
                let swap = cursor.ins().icmp_imm_u(IntCC::UnsignedGreaterThanOrEqual, amount, 64);
                let first = cursor.ins().select(swap, high, low);
                let second = cursor.ins().select(swap, low, high);
                let shift = cursor.ins().band_imm_u(amount, 63);
                let top = cursor.ins().iconst(types::I64, 63);
                let rest = cursor.ins().isub(top, shift);
                let first_shifted = cursor.ins().ishl(first, shift);
                let first_halved = cursor.ins().ushr_imm_u(first, 1);
                let first_carry = cursor.ins().ushr(first_halved, rest);
                let second_shifted = cursor.ins().ishl(second, shift);
                let second_halved = cursor.ins().ushr_imm_u(second, 1);
                let second_carry = cursor.ins().ushr(second_halved, rest);
                let rotated_low = cursor.ins().bor(first_shifted, second_carry);
                let rotated_high = cursor.ins().bor(second_shifted, first_carry);
                cursor.ins().iconcat(rotated_low, rotated_high)
            }
        };
        context.func.dfg.clear_results(inst);
        context.func.dfg.change_to_alias(result, replacement);
        context.func.layout.remove_inst(inst);
    }
    !operations.is_empty()
}
