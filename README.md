## fastalloc: a sample implementation of SSRA

In the `RegallocOptions`, setting `use_fastalloc` will run a sample SSRA
(https://www.mattkeeter.com/blog/2022-10-04-ssra/) implementation.

It only supports registers of class int and it can handle multiple basic
blocks.

To test it out on a toy language: https://github.com/d-sonuga/reverse-linear-scan-regalloc-concept-2.

## regalloc2: another register allocator

This is a register allocator that started life as, and is about 50%
still, a port of IonMonkey's backtracking register allocator to
Rust. In many regards, it has been generalized, optimized, and
improved since the initial port.

In addition, it contains substantial amounts of testing infrastructure
(fuzzing harnesses and checkers) that does not exist in the original
IonMonkey allocator.

See the [design overview](doc/DESIGN.md) for (much!) more detail on
how the allocator works.

## License

This crate is licensed under the Apache 2.0 License with LLVM
Exception. This license text can be found in the file `LICENSE`.

Parts of the code are derived from regalloc.rs: in particular,
`src/checker.rs` and `src/domtree.rs`. This crate has the same license
as regalloc.rs, so the license on these files does not differ.
