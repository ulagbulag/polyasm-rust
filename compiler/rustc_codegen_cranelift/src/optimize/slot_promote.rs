//! Promotion of guest-local stack slots out of memory.
//!
//! MIR hands cg_clif scalar locals that `make_local_place` already keeps in
//! Cranelift variables, but every local whose layout is `BackendRepr::Memory`
//! -- a tuple, an `Option`, any tagged enum -- gets a stack slot instead. After
//! MIR inlining those are the values a loop carries: the `Option<T>` an
//! inlined `Iterator::next` builds and immediately destructures is written and
//! read back through the frame on every turn. LLVM's SROA plus mem2reg erase
//! that round trip, and this pass erases it for this backend.
//!
//! This pass answers one question per slot: does the slot's address stay inside
//! the function. A slot whose every `stack_addr` feeds only the address operand
//! of a plain `load` or `store` stays invisible to everything else in the
//! function -- calls, stores through unrelated pointers and unwind paths alike
//! -- so its cells are ordinary values and the memory traffic naming them is
//! redundant. A slot that leaks its address even once keeps all of its memory;
//! the decision is per slot, distinct from per function.
//!
//! What the pass then does with an eligible slot is deliberately small:
//!
//! * forward a load from a cell to the value the dominating store put there,
//!   within a block and down chains of single-predecessor blocks, where the
//!   stored value provably dominates the load ahead of any SSA construction;
//! * drop a store whose cell every surviving load leaves unread in the function,
//!   which is sound precisely because the slot stays inside the function;
//! * delete what those removals made unreachable, following the operands of
//!   every instruction it removed.
//!
//! Cells are matched on exact offset and exact type. A partial overlap kills
//! the cell in place of being reasoned about, so a slot accessed at mixed
//! widths simply keeps the accesses this pass leaves unresolved.

use std::sync::OnceLock;

use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::flowgraph::ControlFlowGraph;
use cranelift_codegen::ir::{
    Block, Function, Inst, InstructionData, Opcode, StackSlot, Value, ValueDef,
};

use crate::prelude::{FxHashMap, Type};

/// The most cells this pass lets one function hold forwarded values for.
///
/// Forwarding trades a frame slot for a value that stays in a register from the
/// store to the last load that reads it. That is free while the register file
/// has room and ruinous once it runs out: over the guest image, the functions
/// whose lowered code grows do so by three to eight times, and half of their
/// new memory traffic is host stack spill. The allocator breaks on how many
/// values want a register at one point, independent of how far a value
/// travels, so that count is what is bounded here. A function over the bound
/// keeps its memory in place of having some of its cells picked; the cells a
/// heuristic would drop first are the long-lived ones, which are the same cells
/// that remove the most instructions.
///
/// Unset means an unbounded pass, which wins 9.2% on copy-8-1000-x, 21.2% on
/// regex-stage-1 and 35.1% on epipe-stage-1.
fn live_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("CG_CLIF_SLOT_PROMOTE_LIVE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(usize::MAX)
    })
}

/// Where a value points, as far as this pass tells.
#[derive(Copy, Clone, PartialEq, Eq, Default)]
enum AddrOf {
    /// A value from outside `stack_addr`, so it names zero slots this pass owns.
    #[default]
    Nothing,
    /// The result of `stack_addr slot+offset`.
    Slot(StackSlot, i64),
}

/// A cell of a slot that holds a known value at this point.
#[derive(Copy, Clone)]
struct Cell {
    slot: StackSlot,
    offset: i64,
    ty: Type,
    value: Value,
}

impl Cell {
    fn overlaps(&self, slot: StackSlot, offset: i64, size: i64) -> bool {
        self.slot == slot
            && offset < self.offset + i64::from(self.ty.bytes())
            && self.offset < offset + size
    }
}

/// The set of cells whose contents are known at one point in the program.
///
/// Small enough everywhere it is used that a linear scan beats a keyed lookup,
/// and a vector keeps the pass deterministic.
type Known = Vec<Cell>;

fn kill(known: &mut Known, slot: StackSlot, offset: i64, size: i64) {
    known.retain(|cell| !cell.overlaps(slot, offset, size));
}

/// The slot and offset a load or store addresses, when it is a plain access to
/// a slot this pass still promotes.
fn access(
    func: &Function,
    addr_of: &SecondaryMap<Value, AddrOf>,
    escaped: &SecondaryMap<StackSlot, bool>,
    addr: Value,
    offset: i64,
) -> Option<(StackSlot, i64)> {
    match addr_of[func.dfg.resolve_aliases(addr)] {
        AddrOf::Slot(slot, base) if !escaped[slot] => Some((slot, base + offset)),
        _ => None,
    }
}

/// Whether an instruction goes once its results stay unread.
///
/// Loads are excluded in place of being reasoned about: this pass removes only
/// what its own removals orphaned, and a load from elsewhere stays outside its
/// judgement.
fn is_pure(func: &Function, inst: Inst) -> bool {
    let opcode = func.dfg.insts[inst].opcode();
    !opcode.can_load()
        && !opcode.can_store()
        && !opcode.can_trap()
        && !opcode.other_side_effects()
        && !opcode.is_terminator()
        && !opcode.is_branch()
        && !opcode.is_call()
        && !opcode.is_return()
        && func.dfg.has_results(inst)
}

pub(crate) fn run(func: &mut Function) {
    if func.sized_stack_slots.is_empty() {
        return;
    }

    // Every value a `stack_addr` produced, and the cell it names.
    let mut addr_of: SecondaryMap<Value, AddrOf> = SecondaryMap::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            if let InstructionData::StackAddr { stack_slot, offset, .. } = func.dfg.insts[inst] {
                let result = func.dfg.first_result(inst);
                addr_of[result] = AddrOf::Slot(stack_slot, i64::from(offset));
            }
        }
    }

    // A slot escapes as soon as one of its addresses reaches anything but the
    // address operand of a plain load or store. Extending loads and stores,
    // atomics and every other memory opcode land in the catch-all on purpose:
    // this pass reasons about whole cells of one exact width, and an access
    // of unknown size keeps its slot in memory.
    let mut escaped: SecondaryMap<StackSlot, bool> = SecondaryMap::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            match func.dfg.insts[inst] {
                InstructionData::Load { opcode: Opcode::Load, .. }
                | InstructionData::StackAddr { .. } => {}
                InstructionData::Store { opcode: Opcode::Store, args: [stored, _], .. } => {
                    if let AddrOf::Slot(slot, _) = addr_of[func.dfg.resolve_aliases(stored)] {
                        escaped[slot] = true;
                    }
                }
                _ => {
                    for value in func.dfg.inst_values(inst) {
                        if let AddrOf::Slot(slot, _) = addr_of[func.dfg.resolve_aliases(value)] {
                            escaped[slot] = true;
                        }
                    }
                }
            }
        }
    }

    if func.sized_stack_slots.keys().all(|slot| escaped[slot]) {
        return;
    }

    // Forward loads to the stores that dominate them. A block inherits what its
    // predecessor knew only when it has exactly one predecessor, which is what
    // makes the inherited definitions dominate the uses ahead of a dominator
    // tree; every other block starts empty.
    let cfg = ControlFlowGraph::with_function(func);
    let mut leaving: FxHashMap<Block, Known> = FxHashMap::default();
    let mut forwarded: Vec<(Inst, Value)> = Vec::new();
    // The largest number of cells that ever hold a value at once. Counted in
    // the walk that decides the forwarding, so the bound rides on that one pass.
    let mut peak = 0usize;
    for block in func.layout.blocks() {
        let mut sole = None;
        let mut count = 0usize;
        for pred in cfg.pred_iter(block) {
            if sole != Some(pred.block) {
                sole = Some(pred.block);
                count += 1;
            }
        }
        let mut known = match sole {
            Some(pred) if count == 1 => leaving.get(&pred).cloned().unwrap_or_default(),
            _ => Known::new(),
        };

        for inst in func.layout.block_insts(block) {
            match func.dfg.insts[inst] {
                InstructionData::Store {
                    opcode: Opcode::Store,
                    args: [stored, addr],
                    offset,
                    ..
                } => {
                    let Some((slot, offset)) =
                        access(func, &addr_of, &escaped, addr, i64::from(offset))
                    else {
                        continue;
                    };
                    let ty = func.dfg.value_type(func.dfg.resolve_aliases(stored));
                    kill(&mut known, slot, offset, i64::from(ty.bytes()));
                    known.push(Cell { slot, offset, ty, value: stored });
                    peak = peak.max(known.len());
                }
                InstructionData::Load { opcode: Opcode::Load, arg, offset, .. } => {
                    let Some((slot, offset)) =
                        access(func, &addr_of, &escaped, arg, i64::from(offset))
                    else {
                        continue;
                    };
                    let result = func.dfg.first_result(inst);
                    let ty = func.dfg.value_type(result);
                    match known
                        .iter()
                        .find(|cell| cell.slot == slot && cell.offset == offset && cell.ty == ty)
                    {
                        Some(cell) => forwarded.push((inst, cell.value)),
                        // Every overlapping cell is dead here, or the load would
                        // have matched; recording the loaded value makes a
                        // second read of the same cell redundant too.
                        None => {
                            known.push(Cell { slot, offset, ty, value: result });
                            peak = peak.max(known.len());
                        }
                    }
                }
                _ => {}
            }
        }

        leaving.insert(block, known);
    }

    // Over the bound the function keeps every load it had. Dead stores still go:
    // removing an unread write lengthens zero live ranges, so register pressure
    // stays as it is.
    if peak > live_limit() {
        forwarded.clear();
    }

    // Operands of everything this pass removes, to be swept afterwards.
    let mut orphaned: Vec<Value> = Vec::new();
    for &(inst, value) in &forwarded {
        orphaned.extend(func.dfg.inst_values(inst));
        let result = func.dfg.first_result(inst);
        func.dfg.clear_results(inst);
        func.dfg.change_to_alias(result, value);
        func.layout.remove_inst(inst);
    }

    // What every surviving load still reads. A store to a non-escaping slot
    // that zero such loads overlap writes memory that stays unobserved: the slot
    // dies with the frame and its address stays inside it.
    let mut read: FxHashMap<StackSlot, Vec<(i64, i64)>> = FxHashMap::default();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            let InstructionData::Load { opcode: Opcode::Load, arg, offset, .. } =
                func.dfg.insts[inst]
            else {
                continue;
            };
            let Some((slot, offset)) = access(func, &addr_of, &escaped, arg, i64::from(offset))
            else {
                continue;
            };
            let size = i64::from(func.dfg.value_type(func.dfg.first_result(inst)).bytes());
            read.entry(slot).or_default().push((offset, offset + size));
        }
    }

    let mut dead: Vec<Inst> = Vec::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            let InstructionData::Store {
                opcode: Opcode::Store, args: [stored, addr], offset, ..
            } = func.dfg.insts[inst]
            else {
                continue;
            };
            let Some((slot, offset)) = access(func, &addr_of, &escaped, addr, i64::from(offset))
            else {
                continue;
            };
            let size = i64::from(func.dfg.value_type(func.dfg.resolve_aliases(stored)).bytes());
            let observed = read
                .get(&slot)
                .is_some_and(|ranges| ranges.iter().any(|&(a, b)| offset < b && a < offset + size));
            if !observed {
                dead.push(inst);
            }
        }
    }
    for &inst in &dead {
        orphaned.extend(func.dfg.inst_values(inst));
        func.layout.remove_inst(inst);
    }

    if forwarded.is_empty() && dead.is_empty() {
        return;
    }

    // Sweep what the removals orphaned, following operands as they fall.
    let mut uses: SecondaryMap<Value, u32> = SecondaryMap::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            for value in func.dfg.inst_values(inst) {
                uses[func.dfg.resolve_aliases(value)] += 1;
            }
        }
    }
    while let Some(value) = orphaned.pop() {
        let value = func.dfg.resolve_aliases(value);
        if uses[value] != 0 {
            continue;
        }
        let ValueDef::Result(inst, _) = func.dfg.value_def(value) else { continue };
        if func.layout.inst_block(inst).is_none() || !is_pure(func, inst) {
            continue;
        }
        if func.dfg.inst_results(inst).iter().any(|&result| uses[result] != 0) {
            continue;
        }
        for operand in func.dfg.inst_values(inst) {
            let operand = func.dfg.resolve_aliases(operand);
            uses[operand] = uses[operand].saturating_sub(1);
            orphaned.push(operand);
        }
        func.layout.remove_inst(inst);
    }
}
