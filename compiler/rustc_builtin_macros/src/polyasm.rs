use rustc_ast::{self as ast, GeneratedAttr};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::Span;

/// Marks an `extern "polyasm" fn` named `__<name>` as the extension function
/// `name` of the extension instruction set `namespace`.
pub(crate) fn expand(
    ecx: &mut ExtCtxt<'_>,
    attr_sp: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    let list = meta_item.meta_item_list().unwrap_or_default();
    let words: Option<Vec<_>> = list
        .iter()
        .map(|entry| entry.meta_item().filter(|meta_item| meta_item.is_word())?.ident())
        .collect();
    let Some(&[namespace, name]) = words.as_deref() else {
        ecx.dcx().span_err(meta_item.span, "expected the namespace and the name of the extension");
        return vec![item];
    };

    let function = match &mut item {
        Annotatable::Item(p) => match &mut **p {
            ast::Item { attrs, kind: ast::ItemKind::Fn(f), .. } => Some((attrs, &**f)),
            _ => None,
        },
        Annotatable::AssocItem(p, _) => match &mut **p {
            ast::Item { attrs, kind: ast::AssocItemKind::Fn(f), .. } => Some((attrs, &**f)),
            _ => None,
        },
        _ => None,
    };
    let Some((attrs, function)) = function else {
        ecx.dcx().span_err(meta_item.span, "an extension marks an `extern \"polyasm\" fn`");
        return vec![item];
    };

    if !matches!(
        function.sig.header.ext,
        ast::Extern::Explicit(ref abi, _) if abi.symbol_unescaped.as_str() == "polyasm"
    ) {
        ecx.dcx().span_err(function.sig.span, "an extension function takes `extern \"polyasm\"`");
    }
    if function.ident.name.as_str() != format!("__{name}") {
        let message = format!("the extension function `{name}` takes the name `__{name}`");
        ecx.dcx().span_err(function.ident.span, message);
    }
    attrs.push(
        ecx.generated_attr(attr_sp, GeneratedAttr::RustcPolyasmExtension(namespace.name, name.name)),
    );

    vec![item]
}
