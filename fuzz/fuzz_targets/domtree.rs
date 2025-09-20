/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

#![no_main]
use libfuzzer_sys::fuzz_target;
use regalloc2::fuzzing::domtree;

fuzz_target!(|test_case: domtree::TestCase| {
    let _ = env_logger::try_init();
    domtree::check(test_case);
});
