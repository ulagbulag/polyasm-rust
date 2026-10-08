//! Module declarations, definitions, and symbolic relocation state.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cranelift_codegen::Context;
use cranelift_codegen::ir::FuncRef;
use cranelift_codegen::isa::TargetIsa;
use cranelift_module::{
    DataId, FuncId, Linkage, Module, ModuleDeclarations, ModuleError, ModuleReloc,
    ModuleRelocTarget, ModuleResult,
};
use polyasm::object::vocabulary::ClosedAtomic;

use super::super::model::{
    CallableKind, PortableData, PortableFunction, PortableImage, Relocation, RelocationTarget,
};
use super::atomic;

pub(in crate::driver::polyasm::interchange) struct InterchangeModule {
    pub(super) isa: Arc<dyn TargetIsa>,
    anonymous_namespace: String,
    pub(super) callable_kinds: BTreeMap<String, CallableKind>,
    compiler_functions: BTreeSet<String>,
    pub(super) declarations: ModuleDeclarations,
    pub(super) functions: Vec<Option<(u32, Vec<u8>, Vec<ModuleReloc>, Vec<(u32, u32)>)>>,
    pub(super) data: Vec<Option<(u32, Vec<u8>, Vec<ModuleReloc>)>>,
    pub(super) host_functions: BTreeSet<String>,
}

impl InterchangeModule {
    pub(in crate::driver::polyasm::interchange) fn new(
        isa: Arc<dyn TargetIsa>,
        anonymous_namespace: String,
        compiler_functions: BTreeSet<String>,
        mut callable_kinds: BTreeMap<String, CallableKind>,
        host_functions: BTreeSet<String>,
    ) -> Self {
        for name in &host_functions {
            let kind = callable_kinds.entry(name.clone()).or_insert(CallableKind::Function);
            if *kind != CallableKind::Function {
                panic!("host function `{name}` has conflicting callable identity");
            }
        }
        Self {
            isa,
            anonymous_namespace,
            callable_kinds,
            compiler_functions,
            declarations: ModuleDeclarations::default(),
            functions: Vec::new(),
            data: Vec::new(),
            host_functions,
        }
    }

    pub(super) fn function_name(&self, id: FuncId) -> String {
        let declaration = self.declarations.get_function_decl(id);
        declaration
            .name
            .clone()
            .unwrap_or_else(|| format!(".L{}.fn{:x}", self.anonymous_namespace, id.as_u32()))
    }

    pub(super) fn data_name(&self, id: DataId) -> String {
        let declaration = self.declarations.get_data_decl(id);
        declaration
            .name
            .clone()
            .unwrap_or_else(|| format!(".L{}.data{:x}", self.anonymous_namespace, id.as_u32()))
    }

    /// Rewrites every Cranelift atomic into a call to its closed helper, and
    /// answers whether this function held one.
    ///
    /// Cranelift's interchange instruction selector represents zero atomics, so
    /// an atomic left in the graph reaches the selector's capability miss and
    /// silently leaves this function undefined. The closed helper carries the
    /// PolyASM atomic instruction outside the selector's spelling. LANG rule 12
    /// makes atomics unconditional, so the answer decides whether this
    /// definition stays open for another object at all.
    pub(super) fn close_atomic_operations(&mut self, context: &mut Context) -> ModuleResult<bool> {
        let mut callees = BTreeMap::<ClosedAtomic, FuncRef>::new();
        let sites = atomic::sites(&context.func).map_err(backend_error)?;
        let held_atomic = !sites.is_empty();
        for (inst, atomic) in sites {
            let callee = match callees.get(&atomic) {
                Some(callee) => *callee,
                None => {
                    let signature = atomic::signature(
                        atomic,
                        self.isa.default_call_conv(),
                        self.isa.pointer_type(),
                    );
                    let id = self.declare_function(&atomic.name(), Linkage::Local, &signature)?;
                    let callee = self.declare_func_in_func(id, &mut context.func);
                    callees.insert(atomic, callee);
                    callee
                }
            };
            atomic::call(&mut context.func, inst, callee);
        }
        Ok(held_atomic)
    }

    /// Defines each closed atomic helper from its encoded instruction bytes.
    ///
    /// A helper body is one PolyASM atomic instruction and a return, so it is
    /// defined from the bytes `polyasm` assembles in place of a Cranelift
    /// graph, and it stands relocation-free at byte alignment.
    pub(in crate::driver::polyasm::interchange) fn define_closed_atomic_helpers(
        &mut self,
        used: &[ClosedAtomic],
    ) -> ModuleResult<()> {
        for &atomic in used {
            let signature =
                atomic::signature(atomic, self.isa.default_call_conv(), self.isa.pointer_type());
            let id = self.declare_function(&atomic.name(), Linkage::Local, &signature)?;
            let body = atomic.body().map_err(backend_error)?;
            self.define_function_bytes(id, 1, &body, &[])?;
        }
        Ok(())
    }

    fn relocation(&self, reloc: ModuleReloc) -> Relocation {
        let target = match reloc.name {
            ModuleRelocTarget::User { namespace: 0, index } => {
                RelocationTarget::Function(self.function_name(FuncId::from_u32(index)))
            }
            ModuleRelocTarget::User { namespace: 1, index } => {
                RelocationTarget::Data(self.data_name(DataId::from_u32(index)))
            }
            ModuleRelocTarget::User { namespace: 2, index: length } => {
                RelocationTarget::PacketDataRange { length }
            }
            ModuleRelocTarget::User { namespace, .. } => {
                panic!("invalid Cranelift module namespace {namespace}")
            }
            ModuleRelocTarget::LibCall(libcall) => RelocationTarget::LibCall(libcall),
            ModuleRelocTarget::KnownSymbol(symbol) => {
                RelocationTarget::KnownSymbol(symbol.to_string())
            }
            ModuleRelocTarget::FunctionOffset(_, offset) => {
                RelocationTarget::SelfFunctionOffset(offset)
            }
        };
        Relocation { offset: reloc.offset, kind: reloc.kind, target, addend: reloc.addend }
    }

    pub(in crate::driver::polyasm::interchange) fn finish(self) -> PortableImage {
        let functions = self
            .declarations
            .get_functions()
            .map(|(id, declaration)| {
                let definition = self.functions.get(id.as_u32() as usize).and_then(Option::as_ref);
                PortableFunction {
                    callable_kind: self
                        .callable_kinds
                        .get(&self.function_name(id))
                        .copied()
                        .unwrap_or(CallableKind::Unknown),
                    name: self.function_name(id),
                    linkage: declaration.linkage,
                    signature: declaration.signature.clone(),
                    alignment: definition.map_or(1, |item| item.0),
                    code: definition.map(|item| item.1.clone()),
                    compiler_instance: declaration
                        .name
                        .as_ref()
                        .is_some_and(|name| self.compiler_functions.contains(name)),
                    relocs: definition.map_or_else(Vec::new, |item| {
                        item.2.iter().cloned().map(|reloc| self.relocation(reloc)).collect()
                    }),
                    source_call_offsets: definition.map_or_else(Vec::new, |item| item.3.clone()),
                }
            })
            .collect();
        let data = self
            .declarations
            .get_data_objects()
            .map(|(id, declaration)| {
                let definition = self.data.get(id.as_u32() as usize).and_then(Option::as_ref);
                PortableData {
                    name: self.data_name(id),
                    linkage: declaration.linkage,
                    writable: declaration.writable,
                    tls: declaration.tls,
                    alignment: definition.map_or(1, |item| item.0),
                    bytes: definition.map(|item| item.1.clone()),
                    relocs: definition.map_or_else(Vec::new, |item| {
                        item.2.iter().cloned().map(|reloc| self.relocation(reloc)).collect()
                    }),
                }
            })
            .collect();
        PortableImage { functions, data }
    }

    pub(super) fn duplicate_function(&self, id: FuncId) -> ModuleError {
        ModuleError::DuplicateDefinition(self.function_name(id))
    }

    pub(super) fn duplicate_data(&self, id: DataId) -> ModuleError {
        ModuleError::DuplicateDefinition(self.data_name(id))
    }
}

/// States one closed atomic stop as the backend error a module reports.
fn backend_error(message: String) -> ModuleError {
    ModuleError::Backend(std::io::Error::other(message).into())
}
