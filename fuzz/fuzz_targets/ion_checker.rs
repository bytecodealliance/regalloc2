#![no_main]
use libfuzzer_sys::fuzz_target;
use libfuzzer_sys::arbitrary::{Arbitrary, Unstructured, Result};

use regalloc2::fuzzing::func::{Func, Options};
use regalloc2::checker::Checker;

#[derive(Clone, Debug)]
struct TestCase {
    func: Func,
}

impl Arbitrary for TestCase {
    fn arbitrary(u: &mut Unstructured) -> Result<TestCase> {
        Ok(TestCase {
            func: Func::arbitrary_with_options(u, &Options {
                reused_inputs: true,
                fixed_regs: true,
                clobbers: true,
                control_flow: true,
                reducible: false,
                block_params: true,
                always_local_uses: false,
            })?,
        })
    }
}

fuzz_target!(|testcase: TestCase| {
    let func = testcase.func;
    let _ = env_logger::try_init();
    log::debug!("func:\n{:?}", func);
    let env = regalloc2::fuzzing::func::machine_env();
    let out = regalloc2::ion::run(&func, &env).expect("regalloc did not succeed");

    let mut checker = Checker::new(&func);
    checker.prepare(&out);
    checker.run().expect("checker failed");
});
