## regalloc2: another register allocator

This is a register allocator that started life as, and is about 75%
still, a port of IonMonkey's backtracking register allocator to
Rust. The data structures and invariants have been simplified a little
bit, and the interfaces made a little more generic and reusable. In
addition, it contains substantial amounts of testing infrastructure
(fuzzing harnesses and checkers) that does not exist in the original
IonMonkey allocator.

### Design Overview

TODO

- SSA with blockparams

- Operands with constraints, and clobbers, and reused regs; contrast
  with regalloc.rs approach of vregs and pregs and many moves that get
  coalesced/elided

### Differences from IonMonkey Backtracking Allocator

There are a number of differences between the [IonMonkey
allocator](https://searchfox.org/mozilla-central/source/js/src/jit/BacktrackingAllocator.cpp)
and this one:

* Most significantly, there are [fuzz/fuzz_targets/](many different
  fuzz targets) that exercise the allocator, including a full symbolic
  checker (`ion_checker` target) based on the [symbolic checker in
  regalloc.rs](https://cfallin.org/blog/2021/03/15/cranelift-isel-3/)
  and, e.g., a targetted fuzzer for the parallel move-resolution
  algorithm (`moves`) and the SSA generator used for generating cases
  for the other fuzz targets (`ssagen`).

* The data-structure invariants are simplified. While the IonMonkey
  allocator allowed for LiveRanges and Bundles to overlap in certain
  cases, this allocator sticks to a strict invariant: ranges do not
  overlap in bundles, and bundles do not overlap. There are other
  examples too: e.g., the definition of minimal bundles is very simple
  and does not depend on scanning the code at all. In general, we
  should be able to state simple invariants and see by inspection (as
  well as fuzzing -- see above) that they hold.
  
* Many of the algorithms in the IonMonkey allocator are built with
  helper functions that do linear scans. These "small quadratic" loops
  are likely not a huge issue in practice, but nevertheless have the
  potential to be in corner cases. As much as possible, all work in
  this allocator is done in linear scans. For example, bundle
  splitting is done in a single compound scan over a bundle, ranges in
  the bundle, and a sorted list of split-points.
  
* There are novel schemes for solving certain interesting design
  challenges. One example: in IonMonkey, liveranges are connected
  across blocks by, when reaching one end of a control-flow edge in a
  scan, doing a lookup of the allocation at the other end. This is in
  principle a linear lookup (so quadratic overall). We instead
  generate a list of "half-moves", keyed on the edge and from/to
  vregs, with each holding one of the allocations. By sorting and then
  scanning this list, we can generate all edge moves in one linear
  scan. There are a number of other examples of simplifications: for
  example, we handle multiple conflicting
  physical-register-constrained uses of a vreg in a single instruction
  by recording a copy to do in a side-table, then removing constraints
  for the core regalloc. Ion instead has to tweak its definition of
  minimal bundles and create two liveranges that overlap (!) to
  represent the two uses.
  
* Using block parameters rather than phi-nodes significantly
  simplifies handling of inter-block data movement. IonMonkey had to
  special-case phis in many ways because they are actually quite
  weird: their uses happen semantically in other blocks, and their
  defs happen in parallel at the top of the block. Block parameters
  naturally and explicitly reprsent these semantics in a direct way.

* The allocator supports irreducible control flow and arbitrary block
  ordering (its only CFG requirement is that critical edges are
  split). It handles loops during live-range computation in a way that
  is similar in spirit to IonMonkey's allocator -- in a single pass,
  when we discover a loop, we just mark the whole loop as a liverange
  for values live at the top of the loop -- but we find the loop body
  without the fixpoint workqueue loop that IonMonkey uses, instead
  doing a single linear scan for backedges and finding the minimal
  extent that covers all intermingled loops. In order to support
  arbitrary block order and irreducible control flow, we relax the
  invariant that the first liverange for a vreg always starts at its
  def; instead, the def can happen anywhere, and a liverange may
  overapproximate. It turns out this is not too hard to handle and is
  a more robust invariant. (It also means that non-SSA code *may* not
  be too hard to adapt to, though I haven't seriously thought about
  this.)

### Rough Performance Comparison with Regalloc.rs

The allocator has not yet been wired up to a suitable compiler backend
(such as Cranelift) to perform a true apples-to-apples compile-time
and runtime comparison. However, we can get some idea of compile speed
by running suitable test cases through the allocator and measuring
*throughput*: that is, instructions per second for which registers are
allocated.

To do so, I measured the `qsort2` benchmark in
[regalloc.rs](https://github.com/bytecodealliance/regalloc.rs),
register-allocated with default options in that crate's backtracking
allocator, using the Criterion benchmark framework to measure ~620K
instructions per second:


```plain
benches/0               time:   [365.68 us 367.36 us 369.04 us]
                        thrpt:  [617.82 Kelem/s 620.65 Kelem/s 623.49 Kelem/s]
```

I then measured three different fuzztest-SSA-generator test cases in
this allocator, `regalloc2`, measuring between 1.05M and 2.3M
instructions per second (closer to the former for larger functions):

```plain
==== 459 instructions
benches/0               time:   [424.46 us 425.65 us 426.59 us]
                        thrpt:  [1.0760 Melem/s 1.0784 Melem/s 1.0814 Melem/s]

==== 225 instructions
benches/1               time:   [213.05 us 213.28 us 213.54 us]
                        thrpt:  [1.0537 Melem/s 1.0549 Melem/s 1.0561 Melem/s]

Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
==== 21 instructions
benches/2               time:   [9.0495 us 9.0571 us 9.0641 us]
                        thrpt:  [2.3168 Melem/s 2.3186 Melem/s 2.3206 Melem/s]

Found 4 outliers among 100 measurements (4.00%)
  2 (2.00%) high mild
  2 (2.00%) high severe
```

Though not apples-to-apples (SSA vs. non-SSA, completely different
code only with similar length), this is at least some evidence that
`regalloc2` is likely to lead to at least a compile-time improvement
when used in e.g. Cranelift.

### License

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
