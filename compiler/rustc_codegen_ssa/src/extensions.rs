use rustc_middle::ty::TyCtxt;
use rustc_session::config::Lto;

use crate::{ModuleKind, back::write::ModuleConfig};

pub struct Providers {
    pub module_config: for<'tcx> fn(ModuleKind, TyCtxt<'tcx>, &mut ModuleConfig),
    pub lto: for<'tcx> fn(TyCtxt<'tcx>) -> Lto,
}
