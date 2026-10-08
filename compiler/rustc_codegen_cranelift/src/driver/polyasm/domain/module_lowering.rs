//! Crate-wide MIR discovery and relocatable semantic-fragment collection.

use std::collections::BTreeMap;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::mono::MonoItem;
use rustc_middle::ty::{Instance, Ty, TyCtxt};
use rustc_session::config::DebugInfo;
use rustc_span::StableSourceFileId;

use super::super::interchange::PortableImage;
use super::capability::property_capability;
use super::function::{FunctionEmitter, is_compiler_marker, is_core_endian_helper};
use super::model::{Fragment, Source, SourceLanguage};
use super::signature::domain_signature_admits;
use super::source::original_source_bytes;

pub(crate) fn collect<'tcx>(
    tcx: TyCtxt<'tcx>,
    image: &PortableImage,
    negative_witness: &[(String, Ty<'tcx>)],
) -> Fragment {
    ModuleEmitter::new(tcx, image, negative_witness).emit_fragment(image)
}

struct ModuleEmitter<'tcx> {
    capability_exclusions: BTreeMap<String, u32>,
    indices: FxHashMap<String, u32>,
    index_names: Vec<String>,
    instances: Vec<Instance<'tcx>>,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> ModuleEmitter<'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        image: &PortableImage,
        negative_witness: &[(String, Ty<'tcx>)],
    ) -> Self {
        let mut index_names = image
            .function_interfaces()
            .into_iter()
            .map(|function| function.name)
            .collect::<Vec<_>>();
        index_names.extend(image.data_names().map(str::to_owned));
        let indices = index_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index as u32))
            .collect::<FxHashMap<_, _>>();
        let mut capability_exclusions = BTreeMap::<String, u32>::new();
        for (name, property) in negative_witness {
            if let Some(capability) = property_capability(tcx, *property) {
                *capability_exclusions.entry(name.clone()).or_default() |= capability;
            }
        }
        let mut instances = Vec::new();
        for cgu in tcx.collect_and_partition_mono_items(()).codegen_units {
            for (item, _) in cgu.items_in_deterministic_order(tcx) {
                match item {
                    MonoItem::Fn(instance) => {
                        if is_core_endian_helper(tcx, instance)
                            || is_compiler_marker(tcx, instance.def_id())
                        {
                            continue;
                        }
                        let name = tcx.symbol_name(instance).name.to_string();
                        if indices.contains_key(&name)
                            && !instances.iter().any(|other| {
                                tcx.symbol_name(*other).name == tcx.symbol_name(instance).name
                            })
                        {
                            instances.push(instance);
                        }
                    }
                    MonoItem::Static(_) => {}
                    MonoItem::GlobalAsm(item_id) => {
                        tcx.dcx().span_err(
                            tcx.def_span(item_id.owner_id.to_def_id()),
                            "PolyASM does not support global assembly",
                        );
                    }
                }
            }
        }
        Self { capability_exclusions, indices, index_names, instances, tcx }
    }

    fn emit_fragment(mut self, image: &PortableImage) -> Fragment {
        let (sources, source_indices) = self.sources();
        let interfaces = image.function_interfaces();
        let mut emitted = Vec::with_capacity(interfaces.len());
        let mut call_requests = Vec::new();
        emitted.resize_with(interfaces.len(), || None);
        for &instance in &self.instances {
            let name = self.tcx.symbol_name(instance).name.to_string();
            let index = self.indices[&name] as usize;
            // An upstream generic gets instantiated while compiling this
            // crate, but its definition still belongs to the upstream crate.
            // Attaching this crate's source record would make identical
            // cross-crate semantic definitions differ at final linkage.
            let (function, requests) = FunctionEmitter::new(
                self.tcx,
                instance,
                &self.indices,
                image,
                &mut self.capability_exclusions,
                &source_indices,
                &interfaces[index],
            )
            .emit();
            call_requests.extend(requests);
            let function = function.filter(|function| {
                domain_signature_admits(&interfaces[index], &function.params, function.result)
            });
            emitted[index] = function;
        }
        for function in emitted.iter_mut().flatten() {
            if let Some(exclusions) = self.capability_exclusions.get(&function.name)
                && let Some(root) = function.always_scopes.first_mut()
            {
                root.capabilities &= !exclusions;
            }
        }
        Fragment {
            call_requests,
            functions: emitted.into_iter().flatten().collect(),
            index_names: self.index_names,
            sources,
        }
    }

    fn sources(&self) -> (Vec<Source>, BTreeMap<StableSourceFileId, u32>) {
        if self.tcx.sess.opts.debuginfo == DebugInfo::None {
            return (Vec::new(), BTreeMap::new());
        }
        let mut indices = BTreeMap::new();
        let mut sources = Vec::new();
        for file in
            self.tcx.sess.source_map().files().iter().filter(|file| file.cnum == LOCAL_CRATE)
        {
            let Some(source) = file.src.as_ref() else {
                continue;
            };
            let index = u32::try_from(sources.len())
                .unwrap_or_else(|_| self.tcx.dcx().fatal("PolyASM source index exceeds u32"));
            if indices.insert(file.stable_id, index).is_some() {
                self.tcx.dcx().fatal("PolyASM source map contains a duplicate stable file ID");
            }
            sources.push(Source {
                contents: original_source_bytes(self.tcx, file, source),
                language: SourceLanguage::Rust,
                path: file.name.prefer_remapped_unconditionally().to_string(),
            });
        }
        (sources, indices)
    }
}
