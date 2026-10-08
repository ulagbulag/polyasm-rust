use rustc_abi::{HasDataLayout, TyAbiInterface};

use crate::callconv::{ArgAbi, FnAbi};

/// Classifies one PolyASM return value.
///
/// PolyASM answers a 128-bit value in a register pair, which its own wide
/// arithmetic already names, so the indirect return WebAssembly's C ABI gives
/// `__int128_t` would disagree with the signature the compiler's own 128-bit
/// helpers are declared with. Everything narrower is classified as
/// WebAssembly's ABI classifies it, which is what the container encoding and
/// the aggregate rules are shared with.
fn classify_ret<'a, Ty, C>(ret: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    ret.extend_integer_width_to(32);
    if ret.layout.is_aggregate() {
        ret.make_indirect();
    }
}

fn classify_arg<'a, Ty, C>(cx: &C, arg: &mut ArgAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !arg.layout.is_sized() {
        return;
    }
    if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
        arg.make_indirect();
        return;
    }
    arg.extend_integer_width_to(32);
    if arg.layout.is_aggregate() {
        arg.make_indirect();
    }
}

pub(crate) fn compute_abi_info<'a, Ty, C>(cx: &C, fn_abi: &mut FnAbi<'a, Ty>)
where
    Ty: TyAbiInterface<'a, C> + Copy,
    C: HasDataLayout,
{
    if !fn_abi.ret.is_ignore() {
        classify_ret::<Ty, C>(&mut fn_abi.ret);
    }

    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(cx, arg);
    }
}
