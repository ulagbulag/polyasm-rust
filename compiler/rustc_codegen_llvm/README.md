The `codegen` crate contains the code to convert from MIR into LLVM IR,
and then from LLVM IR into machine code. In general it contains code
that runs towards the end of the compilation process.

For more information about how codegen works, see the [rustc dev guide].

[rustc dev guide]: https://rustc-dev-guide.rust-lang.org/backend/codegen.html

PolyASM packet intrinsics each retain their own `__polyasm_packet_*` LLVM
declaration. The PolyASM frontend consumes those declarations as native
opcodes. Packet loads and boundary access have unrestricted effects and leave
the entire invocation on failure.
`packet.rs` follows monomorphized calls, drops, and recursive components so
their callers and call sites promise LLVM `nounwind` only where it holds.
Opaque calls remain conservative; closed graphs without packet access retain
ordinary Rust optimization attributes.

These graphs require the PolyASM default, `-Cpanic=abort`. Rust cleanup edges
would otherwise hide stores preceding a packet exit, which skips those
cleanups. An explicit unwind strategy is rejected for packet-capable graphs.

`PacketDataRange` continues with a boolean on either outcome. Its declaration
is `i1 @__polyasm_packet_data_range(ptr start, ptr end, i32 length)`, with
unrestricted effects so LLVM preserves the original range-carrying call.
Lowering projects the named `window.start` and `window.end` fields through
their evaluated Rust layouts. Lengths above `u32::MAX` are rejected before
constructing the unsigned instruction immediate.
