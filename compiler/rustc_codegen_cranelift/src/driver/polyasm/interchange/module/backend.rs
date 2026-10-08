//! Cranelift Module implementation for function, call, and data encoding.

use cranelift_codegen::binemit::Reloc;
use cranelift_codegen::control::ControlPlane;
use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::entity::{PrimaryMap, SecondaryMap};
use cranelift_codegen::ir::{
    Block, BlockCall, ExternalName, Function, GlobalValue, GlobalValueData, Inst, InstBuilder,
    InstructionData, MemFlagsData, Opcode, Signature, StackSlot, StackSlotKind, TrapCode,
    UserExternalName, Value, ValueDef,
};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::{CodegenError, Context};
use cranelift_module::{
    DataDescription, DataId, FuncId, Linkage, Module, ModuleDeclarations, ModuleError, ModuleReloc,
    ModuleResult,
};
use rustc_log::tracing::info;

use super::instruction::{normalize_wide_instructions, normalize_wide_permutations};
use super::state::InterchangeModule;
use crate::driver::polyasm::interchange::CallableKind;

impl Module for InterchangeModule {
    fn isa(&self) -> &dyn TargetIsa {
        &*self.isa
    }

    fn declarations(&self) -> &ModuleDeclarations {
        &self.declarations
    }

    fn declare_function(
        &mut self,
        name: &str,
        linkage: Linkage,
        signature: &Signature,
    ) -> ModuleResult<FuncId> {
        self.declarations.declare_function(name, linkage, signature).map(|item| item.0)
    }

    fn declare_anonymous_function(&mut self, signature: &Signature) -> ModuleResult<FuncId> {
        self.declarations.declare_anonymous_function(signature)
    }

    fn declare_func_in_func(
        &mut self,
        id: FuncId,
        function: &mut Function,
    ) -> cranelift_codegen::ir::FuncRef {
        let declaration = self.declarations.get_function_decl(id);
        let signature = function.import_signature(declaration.signature.clone());
        let symbol = self.function_name(id);
        let name = function
            .declare_imported_user_function(UserExternalName { namespace: 0, index: id.as_u32() });
        function.import_function(cranelift_codegen::ir::ExtFuncData {
            name: ExternalName::user(name),
            signature,
            colocated: !self.host_functions.contains(&symbol),
            patchable: false,
        })
    }

    fn declare_data(
        &mut self,
        name: &str,
        linkage: Linkage,
        writable: bool,
        tls: bool,
    ) -> ModuleResult<DataId> {
        self.declarations.declare_data(name, linkage, writable, tls).map(|item| item.0)
    }

    fn declare_anonymous_data(&mut self, writable: bool, tls: bool) -> ModuleResult<DataId> {
        self.declarations.declare_anonymous_data(writable, tls)
    }

    fn define_function_with_control_plane(
        &mut self,
        id: FuncId,
        context: &mut Context,
        control_plane: &mut ControlPlane,
    ) -> ModuleResult<()> {
        let declaration = self.declarations.get_function_decl(id);
        if !declaration.linkage.is_definable() {
            return Err(ModuleError::InvalidImportDefinition(self.function_name(id)));
        }
        let name = self.function_name(id);
        let kind = self.callable_kinds.entry(name.clone()).or_insert(CallableKind::Function);
        if *kind == CallableKind::Unknown {
            *kind = CallableKind::Function;
        }
        // LANG rule 12 makes atomics unconditional. A function the atomic
        // rewrite touched therefore reaches the image: leaving it out would
        // make an atomic conditional on Cranelift's interchange coverage, and
        // its caller would meet the missing body at final linkage, where a
        // Rust symbol has zero host descriptors to explain it.
        let held_atomic = self.close_atomic_operations(context)?;
        self.carry_packet_ranges(context)?;
        let index = id.as_u32() as usize;
        if self.functions.len() <= index {
            self.functions.resize_with(index + 1, || None);
        }
        if self.functions[index].is_some() {
            return Err(self.duplicate_function(id));
        }
        normalize_wide_instructions(context);
        read_constants_as_constant(&self.declarations, &mut context.func);
        fold_uniform_branches(&mut context.func);
        remove_unread_definitions(&mut context.func);
        remove_unaddressed_reservations(&mut context.func);
        // The interchange backend's total `lower_cond` selector represents
        // scalar integer conditions through 64 bits alone. For an i128
        // condition Cranelift otherwise reaches the selector's generated
        // missing-rule assertion in place of returning `CodegenError::Unsupported`.
        // Classify that capability miss before instruction selection, leaving
        // an honest undefined partial definition for the final linker. An
        // atomic makes this definition mandatory, so the miss is reported
        // here, where the function and the shape it missed are both known.
        if holds_unselectable_condition(&context.func) {
            if !held_atomic {
                info!(
                    "PolyASM leaves `{name}` to another object: it holds a condition wider than 64 bits"
                );
                return Ok(());
            }
            return Err(unselectable_condition(self.function_name(id)));
        }
        if let Err(error) = context.compile(&*self.isa, control_plane) {
            // A relocatable PolyASM object is a partial capability object. An
            // ISA limitation therefore leaves this declaration undefined so a
            // different object supplies it, or final reachability discards it.
            // Checker and allocator failures remain hard compiler errors:
            // they are bugs, distinct from capability misses. An atomic makes
            // this definition mandatory, so the miss is reported here.
            let unsupported = matches!(error.inner, CodegenError::Unsupported(_));
            let error = ModuleError::from(error);
            // The optimized function stays in the context.  An `i128` rotation
            // or byte swap it holds is the miss the halves of
            // `normalize_wide_permutations` select, so the function compiles
            // once more on those halves; every other function keeps the
            // first answer.
            if !unsupported || !normalize_wide_permutations(context) {
                if unsupported && !held_atomic {
                    info!("PolyASM leaves `{name}` to another object: {error}");
                    return Ok(());
                }
                return Err(error);
            }
            if let Err(error) = context.compile(&*self.isa, control_plane) {
                if matches!(error.inner, CodegenError::Unsupported(_)) && !held_atomic {
                    info!("PolyASM leaves `{name}` to another object: {}", error.inner);
                    return Ok(());
                }
                return Err(error.into());
            }
        }
        let buffer = &context.compiled_code().unwrap().buffer;
        let relocs = buffer
            .relocs()
            .iter()
            .map(|reloc| ModuleReloc::from_mach_reloc(reloc, &context.func, id))
            .collect::<Vec<_>>();
        let source_ranges = buffer.get_srclocs_sorted();
        let source_call_offsets = relocs
            .iter()
            .filter_map(|reloc| {
                let end = source_ranges.partition_point(|location| location.start <= reloc.offset);
                let location = source_ranges.get(end.checked_sub(1)?)?;
                (reloc.offset < location.end
                    && location.loc.bits() & 0x8000_0000 != 0
                    && !location.loc.is_default())
                .then_some((location.loc.bits() & 0x7fff_ffff, reloc.offset))
            })
            .collect();
        // The settled seatings this lane's lowering attaches to every address
        // it forms stay in place here because of the encoding:
        // `buffer.data()` is the *input ISA's* own byte stream -- one-byte
        // opcodes with a three-byte escape for the extended ones, which is
        // what `super::super::object::executable_translation::frame` walks --
        // while the format's framed record stream carries a sixteen-bit tag on
        // every row. The removal belongs in `frame`, where the rows already
        // carry their opcode, their operands and the source advance every
        // pc-relative target is measured against.
        self.functions[index] =
            Some((buffer.alignment, buffer.data().to_vec(), relocs, source_call_offsets));
        Ok(())
    }

    fn define_function_bytes(
        &mut self,
        id: FuncId,
        alignment: u64,
        bytes: &[u8],
        relocs: &[ModuleReloc],
    ) -> ModuleResult<()> {
        let name = self.function_name(id);
        let kind = self.callable_kinds.entry(name).or_insert(CallableKind::Function);
        if *kind == CallableKind::Unknown {
            *kind = CallableKind::Function;
        }
        let index = id.as_u32() as usize;
        if self.functions.len() <= index {
            self.functions.resize_with(index + 1, || None);
        }
        if self.functions[index].is_some() {
            return Err(self.duplicate_function(id));
        }
        let alignment = u32::try_from(alignment).map_err(|_| {
            ModuleError::Backend(
                std::io::Error::other(format!("function alignment {alignment} exceeds u32")).into(),
            )
        })?;
        let mut relocs = relocs.to_vec();
        relocs.sort_by_key(|relocation| relocation.offset);
        self.functions[index] = Some((alignment, bytes.to_vec(), relocs, Vec::new()));
        Ok(())
    }

    fn define_data(&mut self, id: DataId, description: &DataDescription) -> ModuleResult<()> {
        let declaration = self.declarations.get_data_decl(id);
        if !declaration.linkage.is_definable() {
            return Err(ModuleError::InvalidImportDefinition(self.data_name(id)));
        }
        let index = id.as_u32() as usize;
        if self.data.len() <= index {
            self.data.resize_with(index + 1, || None);
        }
        if self.data[index].is_some() {
            return Err(self.duplicate_data(id));
        }
        let bytes = match &description.init {
            cranelift_module::Init::Uninitialized => {
                return Err(ModuleError::Backend(
                    std::io::Error::other("uninitialized interchange data definition").into(),
                ));
            }
            cranelift_module::Init::Zeros { size } => vec![0; *size],
            cranelift_module::Init::Bytes { contents } => contents.to_vec(),
        };
        let alignment = u32::try_from(description.align.unwrap_or(1)).map_err(|_| {
            ModuleError::Backend(
                std::io::Error::other("interchange data alignment exceeds u32").into(),
            )
        })?;
        let mut relocs = description.all_relocs(Reloc::Abs4).collect::<Vec<_>>();
        relocs.sort_by_key(|relocation| relocation.offset);
        self.data[index] = Some((alignment, bytes, relocs));
        Ok(())
    }
}

/// How many forwarding blocks one branch destination is walked through.
///
/// A `jump` cycle among empty blocks is a body that stays inside them, so the
/// walk is bounded in place of running to a fixed point. The chains this
/// compiler emits are one block long -- a MIR arm the inliner emptied leaves a
/// single `goto` -- and the bound only makes the walk end.
const FORWARDED_BLOCK_LIMIT: usize = 16;

/// The user trap this compiler writes where MIR states a place is unreachable.
const UNREACHABLE_TRAP_CODE: u8 = 1;

/// Answers the one instruction a block's body is, where its body is one.
///
/// This compiler opens every block it lowers with a `nop` that anchors the
/// comments it writes beside the MIR the block came from. The `nop` is inert
/// and stays apart from what the block does, so a block whose body is
/// one instruction holds two and a reader counting instructions reads every
/// block as compound. Reading past it makes the two questions below
/// answerable.
fn sole_instruction(function: &Function, block: Block) -> Option<Inst> {
    let mut held: Option<Inst> = None;
    for inst in function.layout.block_insts(block) {
        if function.dfg.insts[inst].opcode() == Opcode::Nop {
            continue;
        }
        if held.is_some() {
            return None;
        }
        held = Some(inst);
    }
    held
}

/// Answers the block one branch destination lands in.
///
/// A block whose whole body is one argument-less `jump` is a pass-through: a
/// branch naming it and a branch naming its destination arrive at the same
/// instructions. Walking through it is what lets two arms of one switch that
/// name two such blocks be seen to arrive at one place; the walk is what turns
/// two names into one, and the switch otherwise stands.
///
/// A destination carrying arguments stays as it is. The arguments are
/// what that arm decides, so two arms handing one block two argument lists are
/// two arrivals however the block is reached.
fn landing_block(function: &Function, call: BlockCall) -> Option<Block> {
    let pool = &function.dfg.value_lists;
    if call.len(pool) != 0 {
        return None;
    }
    let mut landed = call.block(pool);
    for _ in 0..FORWARDED_BLOCK_LIMIT {
        let Some(inst) = sole_instruction(function, landed) else {
            return Some(landed);
        };
        let InstructionData::Jump { destination, .. } = function.dfg.insts[inst] else {
            return Some(landed);
        };
        if destination.len(pool) != 0 {
            return Some(landed);
        }
        landed = destination.block(pool);
    }
    Some(landed)
}

/// Answers whether a block states that arriving in it is a contradiction.
///
/// A bare user trap is what this compiler writes for
/// `TerminatorKind::Unreachable` alone, and MIR states that
/// where the program is already undefined. An arm arriving there therefore
/// leaves the choice free, so a switch whose every other arm arrives at one
/// block arrives at one block.
fn states_unreachable(function: &Function, block: Block) -> bool {
    let Some(inst) = sole_instruction(function, block) else {
        return false;
    };
    let InstructionData::Trap { opcode: Opcode::Trap, code } = function.dfg.insts[inst] else {
        return false;
    };
    TrapCode::user(UNREACHABLE_TRAP_CODE) == Some(code)
}

/// Replaces every switch whose reachable arms arrive at one block with a jump.
///
/// Cranelift reaches this shape only after lowering, where its branch
/// relaxation redirects the emitted branches at their common target and keeps
/// the comparison chain that chose between them: the image keeps a load, a
/// select and two compares in front of a single `ret` that both live arms
/// already reached. Folding before instruction selection takes the whole
/// switch instead, and every value it read stops being read, so
/// [`remove_unread_definitions`] carries the discriminant load and its address
/// arithmetic out with it.
///
/// A drop glue over a niche-optimised `Option<Box<T>>` whose `T` is
/// zero-sized is exactly this: MIR states the third arm unreachable, the drop
/// lowering empties the `Some` arm because deallocating zero bytes is empty work,
/// and what is left decides between two `goto`s that reach one `return`.
fn fold_uniform_branches(function: &mut Function) {
    let mut folds: Vec<(Inst, Block)> = Vec::new();
    for block in function.layout.blocks() {
        let Some(inst) = function.layout.last_inst(block) else {
            continue;
        };
        if !matches!(function.dfg.insts[inst].opcode(), Opcode::Brif | Opcode::BrTable) {
            continue;
        }
        let landings: Option<Vec<Block>> = function.dfg.insts[inst]
            .branch_destination(&function.dfg.jump_tables, &function.dfg.exception_tables)
            .iter()
            .map(|call| landing_block(function, *call))
            .collect();
        let Some(landings) = landings else {
            continue;
        };
        let mut reached =
            landings.iter().filter(|landed| !states_unreachable(function, **landed)).copied();
        let Some(first) = reached.next() else {
            continue;
        };
        if reached.any(|landed| landed != first) {
            continue;
        }
        folds.push((inst, first));
    }
    for (inst, destination) in folds {
        FuncCursor::new(function).at_inst(inst).ins().jump(destination, &[]);
        function.layout.remove_inst(inst);
    }
}

/// Answers whether one instruction leaves the body unchanged when removed.
///
/// This is Cranelift's own `has_side_effect` predicate, which the crate keeps
/// private: an instruction that stores, calls, branches, returns, terminates
/// or carries any other effect stays, and so does one that traps at times. A
/// load is the one exception the predicate makes and the one this pass exists
/// for -- a load whose own flags mark it trap-free has zero effects beyond the
/// value it produces, so a body that leaves that value unread is the same body
/// once the load goes.
fn leaves_body_unchanged(function: &Function, inst: Inst) -> bool {
    let data = &function.dfg.insts[inst];
    let opcode = data.opcode();
    if opcode.is_call()
        || opcode.is_branch()
        || opcode.is_terminator()
        || opcode.is_return()
        || opcode.other_side_effects()
        || opcode.can_store()
    {
        return false;
    }
    if opcode.can_load() {
        return match *data {
            InstructionData::Load { flags, .. } => function.dfg.mem_flags[flags].notrap(),
            _ => false,
        };
    }
    !opcode.can_trap()
}

/// Answers whether every value one instruction defines stays unread in the body.
fn defines_nothing_read(function: &Function, reads: &SecondaryMap<Value, u32>, inst: Inst) -> bool {
    leaves_body_unchanged(function, inst)
        && function.dfg.inst_results(inst).iter().all(|value| reads[*value] == 0)
}

/// Removes every definition the compiled body leaves unread.
///
/// Cranelift's e-graph pass elides a pure value that stays unconsumed, while a load
/// counts as impure there: only a load that is `readonly`, `notrap` and
/// movable joins the e-graph at all, and every other load stays in the
/// skeleton whatever reads what it loaded. That is free on a machine
/// whose registers are registers. It costs six memory operations per load on
/// this one, because a PolyASM machine keeps the guest's registers in a file
/// it addresses through memory, so an interchange load whose value stays unread is a
/// round trip through that file in place of a spare register.
///
/// The removal is a worklist in place of one sweep, so a definition that only
/// fed removed ones goes with them: the address arithmetic under a dead load
/// is pure and becomes unread the moment the load does.
fn remove_unread_definitions(function: &mut Function) {
    let mut reads: SecondaryMap<Value, u32> = SecondaryMap::new();
    for block in function.layout.blocks() {
        for inst in function.layout.block_insts(block) {
            for value in function.dfg.inst_values(inst) {
                let value = function.dfg.resolve_aliases(value);
                reads[value] = reads[value].saturating_add(1);
            }
        }
    }
    let mut worklist: Vec<Inst> = Vec::new();
    for block in function.layout.blocks() {
        for inst in function.layout.block_insts(block) {
            if defines_nothing_read(function, &reads, inst) {
                worklist.push(inst);
            }
        }
    }
    while let Some(inst) = worklist.pop() {
        if function.layout.inst_block(inst).is_none() {
            continue;
        }
        if !defines_nothing_read(function, &reads, inst) {
            continue;
        }
        let arguments: Vec<Value> = function
            .dfg
            .inst_values(inst)
            .map(|value| function.dfg.resolve_aliases(value))
            .collect();
        function.layout.remove_inst(inst);
        for argument in arguments {
            reads[argument] = reads[argument].saturating_sub(1);
            if reads[argument] != 0 {
                continue;
            }
            if let ValueDef::Result(definition, _) = function.dfg.value_def(argument) {
                if defines_nothing_read(function, &reads, definition) {
                    worklist.push(definition);
                }
            }
        }
    }
}

/// The namespace `cranelift-module` gives a data object in a user external name.
///
/// `Module::declare_data_in_func` writes this number and `DataId::from_name`
/// reads it back, so a symbol carrying any other namespace names a function
/// and the function table answers for it. Naming it here is what lets a
/// reader of a symbol decide which of the two tables to ask.
const DATA_NAMESPACE: u32 = 1;

/// How many constant offsets stand between a symbol and a load of it.
///
/// `iadd_imm` is the one shape this compiler puts there and it puts at most
/// one, so the bound only makes the walk end; a longer chain answers as a
/// non-constant in place of being followed.
const CONSTANT_ADDRESS_STEPS: u32 = 4;

/// Answers whether one global value names a constant this compiler wrote.
///
/// A *named* data object stays out of this answer even where its declaration says
/// it is read-only. This compiler declares every static it merely references
/// as `writable: false` and states the real mutability only where it defines
/// one, so a `static` holding a `Cell` reaches a reader through a declaration
/// that reads immutable and is written anyway. An anonymous data object has a
/// single spelling: it exists only where `Module::declare_anonymous_data` was
/// told the allocation's own mutability, so an absent name and a read-only
/// declaration together mark the constant, and each alone leaves it open.
fn names_constant_data(
    declarations: &ModuleDeclarations,
    function: &Function,
    global: GlobalValue,
) -> bool {
    let GlobalValueData::Symbol { name, tls, .. } = &function.global_values[global] else {
        return false;
    };
    if *tls {
        return false;
    }
    let ExternalName::User(reference) = name else {
        return false;
    };
    let Some(named) = function.params.user_named_funcs().get(*reference) else {
        return false;
    };
    if named.namespace != DATA_NAMESPACE {
        return false;
    }
    // The index reaches here from `Module::declare_data_in_func`, which is
    // handed a `DataId` this same declaration table answered with, so the
    // lookup is of a declaration that exists.
    let declared = declarations.get_data_decl(DataId::from_u32(named.index));
    declared.name.is_none() && !declared.writable && !declared.tls
}

/// Answers whether one value is the address of a constant this compiler wrote.
///
/// The walk is over the definitions rather than over the layout, so a symbol
/// defined in one block and offset in another is read the same as one whose
/// whole address is built in a single block.
fn addresses_constant_data(
    declarations: &ModuleDeclarations,
    function: &Function,
    address: Value,
    steps: u32,
) -> bool {
    let address = function.dfg.resolve_aliases(address);
    let ValueDef::Result(definition, _) = function.dfg.value_def(address) else {
        return false;
    };
    match function.dfg.insts[definition] {
        InstructionData::UnaryGlobalValue { opcode: Opcode::SymbolValue, global_value } => {
            names_constant_data(declarations, function, global_value)
        }
        InstructionData::Binary { opcode: Opcode::Iadd, args } => {
            let Some(remaining) = steps.checked_sub(1) else {
                return false;
            };
            let [left, right] = args;
            (states_constant_integer(function, right)
                && addresses_constant_data(declarations, function, left, remaining))
                || (states_constant_integer(function, left)
                    && addresses_constant_data(declarations, function, right, remaining))
        }
        _ => false,
    }
}

/// Answers whether one value is an integer the body states outright.
///
/// An offset into a constant is what this compiler adds to a symbol, and an
/// offset it wrote is an `iconst`. An addend from elsewhere carries the
/// address anywhere, so only the stated one is followed.
fn states_constant_integer(function: &Function, value: Value) -> bool {
    let value = function.dfg.resolve_aliases(value);
    let ValueDef::Result(definition, _) = function.dfg.value_def(value) else {
        return false;
    };
    matches!(
        function.dfg.insts[definition],
        InstructionData::UnaryImm { opcode: Opcode::Iconst, .. }
    )
}

/// States on every load of a constant that the memory it reads is one.
///
/// A `.rodata` allocation is read through a `symbol_value` and a `load`, and
/// the load this compiler writes carries `notrap` alone: the place that writes
/// it holds a `Pointer` whose base is a stack slot, a parameter or a constant
/// alike, and the pointer alone leaves which one open. Cranelift's e-graph admits a load
/// only when it is `readonly`, `notrap` and movable, so a load of a constant
/// stays in the skeleton -- read again by each reader of the same constant
/// and again on every turn of a loop that reads one.
///
/// That is free on a machine whose registers are registers. It costs an
/// address, a widening and a round trip through the register file on every
/// iteration on this one, because a PolyASM machine keeps the guest's
/// registers in a file it addresses through memory.
fn read_constants_as_constant(declarations: &ModuleDeclarations, function: &mut Function) {
    let mut upgraded: Vec<(Inst, MemFlagsData)> = Vec::new();
    for block in function.layout.blocks() {
        for inst in function.layout.block_insts(block) {
            let InstructionData::Load { arg, flags, .. } = function.dfg.insts[inst] else {
                continue;
            };
            let stated = function.dfg.mem_flags[flags];
            if !stated.notrap() || (stated.readonly() && stated.can_move()) {
                continue;
            }
            if !addresses_constant_data(declarations, function, arg, CONSTANT_ADDRESS_STEPS) {
                continue;
            }
            upgraded.push((inst, stated.with_readonly().with_can_move()));
        }
    }
    for (inst, stated) in upgraded {
        let Ok(flags) = function.dfg.mem_flags.insert(stated) else {
            continue;
        };
        if let InstructionData::Load { flags: held, .. } = &mut function.dfg.insts[inst] {
            *held = flags;
        }
    }
}

/// Drops every explicit stack reservation whose address every instruction leaves untaken.
///
/// A reservation is only reachable through `stack_addr`, so one that zero
/// surviving instructions name is a reservation the body leaves untouched. Leaving it declared costs a PolyASM machine: the frame's extent
/// decides whether the interchange ABI writes a frame at all, and a body whose
/// last reader [`remove_unread_definitions`] took away is otherwise still
/// charged an open, a reserve, a release and a close on every call to it.
///
/// The reservation is removed in place of resized to zero. A reservation of
/// zero bytes is a reservation the frame still carries and still aligns, and
/// this compiler spells an absent thing by its absence.
fn remove_unaddressed_reservations(function: &mut Function) {
    let mut addressed: SecondaryMap<StackSlot, bool> = SecondaryMap::new();
    let mut instructions: Vec<Inst> = Vec::new();
    for block in function.layout.blocks() {
        for inst in function.layout.block_insts(block) {
            instructions.push(inst);
            if let Some(slot) = function.dfg.insts[inst].stack_slot() {
                addressed[slot] = true;
            }
        }
    }
    let mut kept = PrimaryMap::new();
    let mut renamed: SecondaryMap<StackSlot, Option<StackSlot>> = SecondaryMap::new();
    for (slot, data) in function.sized_stack_slots.iter() {
        if !addressed[slot] && data.kind == StackSlotKind::ExplicitSlot {
            continue;
        }
        renamed[slot] = Some(kept.push(data.clone()));
    }
    if kept.len() == function.sized_stack_slots.len() {
        return;
    }
    function.sized_stack_slots = kept;
    for inst in instructions {
        if let InstructionData::StackAddr { stack_slot, .. } = &mut function.dfg.insts[inst] {
            let moved = renamed[*stack_slot].unwrap_or(*stack_slot);
            *stack_slot = moved;
        }
    }
}

/// Answers whether a function branches on a condition outside the interchange backend's selector.
///
/// The selector is total over scalar integers through 64 bits alone, so a
/// vector or an `i128` condition reaches its generated missing-rule assertion
/// in place of `CodegenError::Unsupported`.
fn holds_unselectable_condition(function: &Function) -> bool {
    function.layout.blocks().any(|block| {
        function.layout.block_insts(block).any(|inst| {
            if !matches!(
                function.dfg.insts[inst].opcode(),
                Opcode::Brif | Opcode::Trapz | Opcode::Trapnz | Opcode::Select
            ) {
                return false;
            }
            function.dfg.inst_args(inst).first().is_some_and(|condition| {
                let ty = function.dfg.value_type(*condition);
                !ty.is_int() || ty.is_vector() || ty.lane_bits() > 64
            })
        })
    })
}

fn unselectable_condition(name: String) -> ModuleError {
    ModuleError::Backend(
        std::io::Error::other(format!(
            "PolyASM function `{name}` holds an atomic and a condition the interchange backend cannot select; LANG rule 12 makes an atomic unconditional, so this definition cannot be left to another object"
        ))
        .into(),
    )
}
