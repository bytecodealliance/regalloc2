/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

#![no_main]
use libfuzzer_sys::fuzz_target;
use regalloc2::fuzzing::fastalloc;

fuzz_target!(|test_case: fastalloc::TestCase| {
    let _ = env_logger::try_init();
    fastalloc::check(test_case);
});
