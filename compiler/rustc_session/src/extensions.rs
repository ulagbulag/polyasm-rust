use std::any::{Any, TypeId};

use rustc_ast::Attribute;
use rustc_data_structures::{fx::FxHashMap, sync::IntoDynSyncSend};

use crate::{Session, config::Cfg};

pub struct Extensions {
    data: FxHashMap<TypeId, IntoDynSyncSend<Box<dyn Any + Send + Sync>>>,
    pub lang_items: &'static [&'static str],
    pub allow_auto_trait_generics: fn(&Session, &[Attribute]) -> bool,
    pub configure_cfg: fn(&Session, &mut Cfg),
    pub configure_user_cfg: fn(&Session, &Cfg),
}

impl Default for Extensions {
    fn default() -> Self {
        Self {
            data: FxHashMap::default(),
            lang_items: &[],
            allow_auto_trait_generics: |_, _| false,
            configure_cfg: |_, _| {},
            configure_user_cfg: |_, _| {},
        }
    }
}

impl Extensions {
    pub fn insert<T: Any + Send + Sync>(&mut self, value: T) {
        self.data.insert(TypeId::of::<T>(), IntoDynSyncSend(Box::new(value)));
    }

    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.data.get(&TypeId::of::<T>())?.downcast_ref()
    }
}
