//! Compile-time construction of numeric work-lifecycle authority.
//!
//! Export names are compiler input only. Every arbitrary
//! `wasm-direct::guest::<namespace>::<name>` group is checked for a complete
//! lifecycle, converted to a stable numeric identity and carried by symbol in
//! the object's work fragment. The final link closes every fragment into
//! `polyasm_format::work`'s table through
//! `polytime_frontend_cranelift::CraneliftFrontend`.

use std::collections::BTreeMap;

use polyasm::object::work::{Fragment, WorkRecord};
use polyasm_format::executable::AbiParam;
use polyasm_format::work::{
    self, AbiPlace, GUEST_SYMBOL_PREFIX, HEAP_ALLOC_SYMBOL, HEAP_DEALLOC_SYMBOL,
    MANIFEST_LENGTH_KIND, MANIFEST_POINTER_KIND, Role, SYMBOL_SEPARATOR, Symbol, Work,
};
use rustc_middle::mir::interpret::{GlobalAlloc, Scalar};
use rustc_middle::mir::{ConstValue, Operand, Rvalue, StatementKind, TerminatorKind};
use rustc_middle::ty::{Instance, TyCtxt, TypingEnv};
use rustc_span::DUMMY_SP;

use super::debug;
use super::interchange::PortableImage;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ExportRole {
    Alloc,
    Dealloc,
    Drop,
    ManifestLen,
    ManifestPtr,
    Poll,
    Start,
}

#[derive(Default)]
struct WorkExports {
    names: BTreeMap<ExportRole, String>,
}

pub(crate) fn collect(tcx: TyCtxt<'_>, image: &PortableImage) -> Option<Fragment> {
    let functions = debug::mono_functions(tcx);
    let mut groups = BTreeMap::<(String, String), WorkExports>::new();
    for name in functions.keys() {
        let Some((namespace, work_name, role)) = parse_work_export(name) else {
            continue;
        };
        let group = groups.entry((namespace.to_owned(), work_name.to_owned())).or_default();
        if let Some(previous) = group.names.insert(role, name.clone()) {
            tcx.dcx().fatal(format!(
                "duplicate work lifecycle role `{role:?}`: `{previous}` and `{name}`"
            ));
        }
    }
    if groups.is_empty() {
        return None;
    }

    let mut ids = BTreeMap::<u32, (String, String)>::new();
    let mut works = BTreeMap::<u32, WorkRecord>::new();
    let width = super::pointer_width(tcx.sess);
    for ((namespace, name), exports) in groups {
        let label = format!("{namespace}::{name}");
        let alloc_name = required_role(tcx, &exports, ExportRole::Alloc, &label);
        let start_name = required_role(tcx, &exports, ExportRole::Start, &label);
        let poll_name = required_role(tcx, &exports, ExportRole::Poll, &label);
        let drop_name = required_role(tcx, &exports, ExportRole::Drop, &label);
        let dealloc_name = required_role(tcx, &exports, ExportRole::Dealloc, &label);

        check_abi(tcx, image, alloc_name, AbiPlace::Alloc.signature_at(width));
        for function in [start_name, poll_name] {
            check_abi(tcx, image, function, AbiPlace::Advance.signature_at(width));
        }
        for function in [drop_name, dealloc_name] {
            check_abi(tcx, image, function, AbiPlace::Release.signature_at(width));
        }

        let manifest = match (
            exports.names.get(&ExportRole::ManifestPtr),
            exports.names.get(&ExportRole::ManifestLen),
        ) {
            (None, None) => format!("{namespace}::{name}").into_bytes(),
            (Some(pointer), Some(length)) => evaluate_manifest(
                tcx,
                required_instance(tcx, &functions, pointer, &label),
                required_instance(tcx, &functions, length, &label),
                &label,
            ),
            _ => tcx.dcx().fatal(format!(
                "incomplete work lifecycle `{label}`: `{MANIFEST_POINTER_KIND}` and `{MANIFEST_LENGTH_KIND}` are exported together or both absent"
            )),
        };

        let id = work::compiler_id(Symbol { namespace: &namespace, name: &name });
        if let Some((other_namespace, other_name)) =
            ids.insert(id, (namespace.clone(), name.clone()))
            && (other_namespace != namespace || other_name != name)
        {
            tcx.dcx().fatal(format!(
                "work identity collision {id:#010x}: `{other_namespace}::{other_name}` and `{namespace}::{name}`"
            ));
        }
        let work = WorkRecord {
            alloc: alloc_name.to_owned(),
            dealloc: dealloc_name.to_owned(),
            drop: drop_name.to_owned(),
            id,
            manifest,
            poll: poll_name.to_owned(),
            start: start_name.to_owned(),
        };
        for function in [alloc_name, start_name, poll_name, drop_name, dealloc_name] {
            required_index(tcx, image, function, &label);
        }
        if works.insert(id, work).is_some() {
            tcx.dcx().fatal(format!("duplicate work identity {id:#010x} for `{label}`"));
        }
    }

    Some(Fragment {
        heap_alloc: HEAP_ALLOC_SYMBOL.to_owned(),
        heap_dealloc: HEAP_DEALLOC_SYMBOL.to_owned(),
        works: works.into_values().collect(),
    })
}

fn parse_work_export(name: &str) -> Option<(&str, &str, ExportRole)> {
    let mut components = name.strip_prefix(GUEST_SYMBOL_PREFIX)?.split(SYMBOL_SEPARATOR);
    let namespace = components.next()?;
    let work = components.next()?;
    let role = match components.next()? {
        MANIFEST_LENGTH_KIND => ExportRole::ManifestLen,
        MANIFEST_POINTER_KIND => ExportRole::ManifestPtr,
        kind => match Work::roles().into_iter().find(|role| Work::kind(*role) == kind)? {
            Role::Alloc => ExportRole::Alloc,
            Role::Dealloc => ExportRole::Dealloc,
            Role::Drop => ExportRole::Drop,
            Role::Poll => ExportRole::Poll,
            Role::Start => ExportRole::Start,
        },
    };
    if namespace.is_empty() || work.is_empty() || components.next().is_some() {
        return None;
    }
    Some((namespace, work, role))
}

fn required_role<'a>(
    tcx: TyCtxt<'_>,
    exports: &'a WorkExports,
    role: ExportRole,
    label: &str,
) -> &'a str {
    exports.names.get(&role).map(String::as_str).unwrap_or_else(|| {
        tcx.dcx().fatal(format!(
            "incomplete work lifecycle `{label}`: required `{role:?}` export is absent"
        ))
    })
}

fn required_index(tcx: TyCtxt<'_>, image: &PortableImage, name: &str, label: &str) -> u32 {
    image.function_index(name).unwrap_or_else(|| {
        tcx.dcx().fatal(format!(
            "incomplete work lifecycle `{label}`: required export `{name}` is absent from the interchange image"
        ))
    })
}

fn required_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &BTreeMap<String, Instance<'tcx>>,
    name: &str,
    label: &str,
) -> Instance<'tcx> {
    functions.get(name).copied().unwrap_or_else(|| {
        tcx.dcx().fatal(format!(
            "incomplete work lifecycle `{label}`: required constant export `{name}` is absent"
        ))
    })
}

fn check_abi(
    tcx: TyCtxt<'_>,
    image: &PortableImage,
    name: &str,
    (expected_params, expected_returns): (Vec<AbiParam>, Vec<AbiParam>),
) {
    let Some((actual_params, actual_returns)) = image.function_abi(name) else {
        tcx.dcx().fatal(format!("required lifecycle function `{name}` is absent"));
    };
    if actual_params != expected_params || actual_returns != expected_returns {
        tcx.dcx().fatal(format!(
            "lifecycle function `{name}` has ABI {actual_params:?} -> {actual_returns:?}, expected {expected_params:?} -> {expected_returns:?}"
        ));
    }
}

/// Reads the one constant an exported accessor returns.
///
/// A generated lifecycle export is an ordinary `extern "C"` function, which the
/// const evaluator stops at, and evaluating it anyway would abort the compiler.
/// Its body hands back one compile-time constant, which the built MIR still
/// carries, so this reads that constant directly.
fn returned_constant<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Option<ConstValue> {
    let def_id = instance.def_id();
    if tcx.is_const_fn(def_id)
        && let Ok(value) =
            tcx.const_eval_instance(TypingEnv::fully_monomorphized(), instance, DUMMY_SP)
    {
        return Some(value);
    }
    if !tcx.is_mir_available(def_id) {
        return None;
    }
    let body = tcx.optimized_mir(def_id);
    let assigned =
        body.basic_blocks.iter().flat_map(|data| data.statements.iter()).filter_map(|statement| {
            match &statement.kind {
                StatementKind::Assign(assignment) => match &(**assignment).1 {
                    Rvalue::Use(Operand::Constant(constant), _)
                    | Rvalue::Cast(_, Operand::Constant(constant), _) => Some(&**constant),
                    _ => None,
                },
                _ => None,
            }
        });
    // The accessor usually hands its constant straight to a `core` helper, so
    // the value lives in a call argument rather than in an assignment.
    let passed = body.basic_blocks.iter().flat_map(|data| match &data.terminator().kind {
        TerminatorKind::Call { args, .. } | TerminatorKind::TailCall { args, .. } => args
            .iter()
            .filter_map(|argument| match &argument.node {
                Operand::Constant(constant) => Some(&**constant),
                _ => None,
            })
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    });
    assigned
        .chain(passed)
        .filter_map(|constant| {
            constant.const_.eval(tcx, TypingEnv::fully_monomorphized(), DUMMY_SP).ok()
        })
        .find(|value| !matches!(value, ConstValue::ZeroSized))
}

fn evaluate_manifest<'tcx>(
    tcx: TyCtxt<'tcx>,
    pointer: Instance<'tcx>,
    length: Instance<'tcx>,
    label: &str,
) -> Vec<u8> {
    // A manifest constant arrives either as one wide slice value or as a
    // pointer whose length its sibling export reports.
    if let Some(ConstValue::Slice { alloc_id, meta }) = returned_constant(tcx, pointer) {
        let end = usize::try_from(meta).unwrap_or_else(|_| {
            tcx.dcx().fatal(format!("work lifecycle `{label}` manifest length exceeds host usize"))
        });
        let GlobalAlloc::Memory(allocation) = tcx.global_alloc(alloc_id) else {
            tcx.dcx().fatal(format!(
                "work lifecycle `{label}` manifest pointer is not backed by immutable memory"
            ));
        };
        return normalize_manifest(
            tcx,
            allocation.inner().inspect_with_uninit_and_ptr_outside_interpreter(0..end).to_vec(),
            label,
        );
    }
    let length = returned_constant(tcx, length)
        .and_then(|value| value.try_to_target_usize(tcx))
        .unwrap_or_else(|| {
            tcx.dcx()
                .fatal(format!("work lifecycle `{label}` manifest length is not const-evaluable"))
        });
    let pointer = returned_constant(tcx, pointer).unwrap_or_else(|| {
        tcx.dcx().fatal(format!("work lifecycle `{label}` manifest pointer is not const-evaluable"))
    });
    let ConstValue::Scalar(Scalar::Ptr(pointer, _)) = pointer else {
        tcx.dcx().fatal(format!(
            "work lifecycle `{label}` manifest pointer did not evaluate to a pointer"
        ));
    };
    let (provenance, offset) = pointer.prov_and_relative_offset();
    let start = usize::try_from(offset.bytes()).unwrap_or_else(|_| {
        tcx.dcx().fatal(format!("work lifecycle `{label}` manifest offset exceeds host usize"))
    });
    let length = usize::try_from(length).unwrap_or_else(|_| {
        tcx.dcx().fatal(format!("work lifecycle `{label}` manifest length exceeds host usize"))
    });
    let end = start.checked_add(length).unwrap_or_else(|| {
        tcx.dcx().fatal(format!("work lifecycle `{label}` manifest range overflow"))
    });
    let bytes = match tcx.global_alloc(provenance.alloc_id()) {
        GlobalAlloc::Memory(allocation) => {
            allocation.inner().inspect_with_uninit_and_ptr_outside_interpreter(start..end)
        }
        GlobalAlloc::Static(def_id) => tcx
            .eval_static_initializer(def_id)
            .unwrap_or_else(|_| {
                tcx.dcx()
                    .fatal(format!("work lifecycle `{label}` manifest static cannot be evaluated"))
            })
            .inner()
            .inspect_with_uninit_and_ptr_outside_interpreter(start..end),
        _ => tcx.dcx().fatal(format!(
            "work lifecycle `{label}` manifest pointer is not backed by immutable memory"
        )),
    };
    normalize_manifest(tcx, bytes.to_vec(), label)
}

/// Stores one manifest NUL-free at its end and stops at an interior NUL.
///
/// A guest publishes its manifest either as a NUL-terminated C string or as
/// a plain byte string; both describe the same bytes.
fn normalize_manifest(tcx: TyCtxt<'_>, mut manifest: Vec<u8>, label: &str) -> Vec<u8> {
    if manifest.last().copied() == Some(0) {
        manifest.pop();
    }
    if manifest.contains(&0) {
        tcx.dcx().fatal(format!("work lifecycle `{label}` manifest contains an interior NUL byte"));
    }
    manifest
}
