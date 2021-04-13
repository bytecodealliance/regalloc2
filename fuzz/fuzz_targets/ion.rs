#![no_main]
use libfuzzer_sys::fuzz_target;

use regalloc2::fuzzing::func::Func;

fuzz_target!(|func: Func| {
    let _ = env_logger::try_init();
    log::debug!("func:\n{:?}", func);
    let env = regalloc2::fuzzing::func::machine_env();
    let _out = regalloc2::ion::run(&func, &env).expect("regalloc did not succeed");
});
