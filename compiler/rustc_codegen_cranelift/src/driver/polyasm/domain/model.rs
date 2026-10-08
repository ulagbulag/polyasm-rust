//! The relocatable semantic-graph model, `polyasm::object::domain`'s.

pub(crate) use polyasm::object::domain::{
    AlwaysScope, AlwaysScopeKind, CallRequest, ClosureEnvironment, DebugLocation, DebugProjection,
    DebugRange, DebugScope, DebugVariable, Fragment, Function, Hint, HintKind, Instruction,
    OFFLOAD_CALL_HINT_TAG, Opcode, Register, Source, SourceLanguage, ValueType,
};
