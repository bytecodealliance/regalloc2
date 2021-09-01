## regalloc2: another register allocator

This is a register allocator that started life as, and is about 50%
still, a port of IonMonkey's backtracking register allocator to
Rust. In many regards, it has been generalized, optimized, and
improved since the initial port, and now supports both SSA and non-SSA
use-cases.

In addition, it contains substantial amounts of testing infrastructure
(fuzzing harnesses and checkers) that does not exist in the original
IonMonkey allocator.

See the [design overview](doc/DESIGN.md) for (much!) more detail on
how the allocator works.

## License

Unless otherwise specified, code in this crate is licensed under the Apache 2.0
License with LLVM Exception. This license text can be found in the file
`LICENSE`.

Files in the `src/ion/` directory are directly ported from original C++ code in
IonMonkey, a part of the Firefox codebase. Parts of `src/lib.rs` are also
definitions that are directly translated from this original code. As a result,
these files are derivative works and are covered by the Mozilla Public License
(MPL) 2.0, as described in license headers in those files. Please see the
notices in relevant files for links to the original IonMonkey source files from
which they have been translated/derived. The MPL text can be found in
`src/ion/LICENSE`.

Parts of the code are derived from regalloc.rs: in particular,
`src/checker.rs` and `src/domtree.rs`. This crate has the same license
as regalloc.rs, so the license on these files does not differ.
