//! Exact source-byte reconstruction and source-position recovery.

use std::collections::BTreeMap;

use rustc_middle::ty::TyCtxt;
use rustc_span::{Pos, SourceFileAndLine, Span, StableSourceFileId, hygiene};

pub(super) fn original_source_bytes(
    tcx: TyCtxt<'_>,
    file: &rustc_span::SourceFile,
    source: &str,
) -> Vec<u8> {
    let mut contents = source.as_bytes().to_vec();
    for (index, normalized) in file.normalized_pos.iter().enumerate().rev() {
        let previous = index.checked_sub(1).map_or(0, |index| file.normalized_pos[index].diff);
        let Some(removed) = normalized.diff.checked_sub(previous) else {
            tcx.dcx().fatal("PolyASM source normalization offsets are not monotonic");
        };
        let position = normalized.pos.to_usize();
        match (position, removed) {
            (0, 3) if index == 0 => {
                contents.insert(0, 0xbf);
                contents.insert(0, 0xbb);
                contents.insert(0, 0xef);
            }
            (position, 1)
                if position.checked_sub(1).and_then(|position| source.as_bytes().get(position))
                    == Some(&b'\n') =>
            {
                contents.insert(position - 1, b'\r');
            }
            _ => tcx.dcx().fatal("PolyASM cannot reverse an unknown rustc source normalization"),
        }
    }
    let Ok(original) = std::str::from_utf8(&contents) else {
        tcx.dcx().fatal("PolyASM reconstructed invalid UTF-8 debug source");
    };
    if contents.len() != file.unnormalized_source_len as usize || !file.src_hash.matches(original) {
        tcx.dcx().fatal("PolyASM could not reconstruct the exact original debug source");
    }
    contents
}

pub(super) fn debug_position(
    tcx: TyCtxt<'_>,
    source_indices: &BTreeMap<StableSourceFileId, u32>,
    function_span: Span,
    span: Span,
) -> Option<(u32, u32, u32)> {
    let span = hygiene::walk_chain_collapsed(span, function_span);
    if span.is_dummy() {
        return None;
    }
    let Ok(SourceFileAndLine { sf, line }) = tcx.sess.source_map().lookup_line(span.lo()) else {
        return None;
    };
    let source = *source_indices.get(&sf.stable_id)?;
    let line_position = sf.lines()[line];
    let column = (sf.relative_position(span.lo()) - line_position).to_u32().checked_add(1)?;
    let line = u32::try_from(line).ok()?.checked_add(1)?;
    Some((source, line, column))
}
