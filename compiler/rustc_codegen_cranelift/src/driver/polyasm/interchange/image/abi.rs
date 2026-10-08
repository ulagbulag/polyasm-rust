//! Portable function signature queries.

use polyasm_format::executable::AbiParam;

use super::super::model::{FunctionSignature, PortableImage};

impl PortableImage {
    pub(crate) fn data_names(&self) -> impl Iterator<Item = &str> {
        self.fragment.data.iter().map(|data| data.name.as_str())
    }

    pub(crate) fn function_interfaces(&self) -> Vec<FunctionSignature> {
        self.fragment
            .functions
            .iter()
            .map(|function| FunctionSignature {
                callable_kind: function.callable_kind,
                name: function.name.clone(),
                params: function.params.clone(),
                returns: function.returns.clone(),
            })
            .collect()
    }

    pub(crate) fn function_index(&self, name: &str) -> Option<u32> {
        self.fragment
            .functions
            .binary_search_by(|function| function.name.as_str().cmp(name))
            .ok()
            .and_then(|index| u32::try_from(index).ok())
    }

    pub(crate) fn function_abi(&self, name: &str) -> Option<(Vec<AbiParam>, Vec<AbiParam>)> {
        let index = usize::try_from(self.function_index(name)?).ok()?;
        let function = self.fragment.functions.get(index)?;
        Some((function.params.clone(), function.returns.clone()))
    }
}
