/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

#![no_main]
use regalloc2::fuzzing::func::Func;
use regalloc2::fuzzing::fuzz_target;

fuzz_target!(|func: Func| {
    let _ = env_logger::try_init();
    log::trace!("func:\n{:?}", func);
    let env = regalloc2::fuzzing::func::machine_env();
    let _out = regalloc2::fuzzing::ion::run(&func, &env, false).expect("regalloc did not succeed");
});
