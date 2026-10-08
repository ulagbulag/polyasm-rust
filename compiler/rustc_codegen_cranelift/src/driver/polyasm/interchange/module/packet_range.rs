//! Carries a range's unsigned immediate in its instruction relocation.
//!
//! The CLIF declaration has exactly `(i64, i64, i32) -> i8`. Its third
//! argument is an instruction immediate, distinct from a runtime function
//! parameter. Namespace two retains that constant through instruction
//! selection as the constant itself, apart from function definitions and from
//! allocated registers.

use cranelift_codegen::Context;
use cranelift_codegen::ir::{
    ExtFuncData, ExternalName, InstructionData, Opcode, UserExternalName, ValueDef, types,
};
use cranelift_module::{FuncId, ModuleError, ModuleResult};

use super::state::InterchangeModule;

impl InterchangeModule {
    pub(super) fn carry_packet_ranges(&mut self, context: &mut Context) -> ModuleResult<()> {
        let function = &mut context.func;
        let mut sites = Vec::new();
        for block in function.layout.blocks() {
            for instruction in function.layout.block_insts(block) {
                let InstructionData::Call { func_ref, args, .. } = function.dfg.insts[instruction]
                else {
                    continue;
                };
                let external = &function.dfg.ext_funcs[func_ref];
                let ExternalName::User(reference) = external.name else { continue };
                let named = &function.params.user_named_funcs()[reference];
                if named.namespace != 0
                    || self.function_name(FuncId::from_u32(named.index))
                        != "__polyasm_packet_data_range"
                {
                    continue;
                }
                let signature = &function.dfg.signatures[external.signature];
                let arguments = args.as_slice(&function.dfg.value_lists);
                if signature.params.iter().map(|parameter| parameter.value_type).collect::<Vec<_>>()
                    != [types::I64, types::I64, types::I32]
                    || signature
                        .returns
                        .iter()
                        .map(|parameter| parameter.value_type)
                        .collect::<Vec<_>>()
                        != [types::I8]
                    || arguments.len() != 3
                {
                    return Err(invalid());
                }
                let length = function.dfg.resolve_aliases(arguments[2]);
                let ValueDef::Result(definition, _) = function.dfg.value_def(length) else {
                    return Err(invalid());
                };
                let InstructionData::UnaryImm { opcode: Opcode::Iconst, imm } =
                    function.dfg.insts[definition]
                else {
                    return Err(invalid());
                };
                if function.dfg.value_type(length) != types::I32 {
                    return Err(invalid());
                }
                // I32 constants carry a signed CLIF spelling but unsigned bits.
                let length = imm.bits() as u32;
                sites.push((instruction, external.signature, length));
            }
        }
        for (instruction, signature, length) in sites {
            let name = function
                .declare_imported_user_function(UserExternalName { namespace: 2, index: length });
            let callee = function.import_function(ExtFuncData {
                name: ExternalName::user(name),
                signature,
                colocated: true,
                patchable: false,
            });
            let InstructionData::Call { func_ref, .. } = &mut function.dfg.insts[instruction]
            else {
                unreachable!();
            };
            *func_ref = callee;
        }
        Ok(())
    }
}

fn invalid() -> ModuleError {
    ModuleError::Backend(std::io::Error::other(
        "PacketDataRange requires original i64 boundaries, a constant unsigned i32 length and an i8 answer",
    ).into())
}
