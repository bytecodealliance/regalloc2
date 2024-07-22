/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

#![no_main]
use regalloc2::fuzzing::arbitrary::{Arbitrary, Result, Unstructured};
use regalloc2::fuzzing::checker::Checker;
use regalloc2::fuzzing::func::{Func, Options};
use regalloc2::fuzzing::fuzz_target;

#[derive(Clone, Debug)]
struct TestCase {
    func: Func,
}

impl Arbitrary<'_> for TestCase {
    fn arbitrary(u: &mut Unstructured) -> Result<TestCase> {
        Ok(TestCase {
            func: Func::arbitrary_with_options(
                u,
                &Options {
                    reused_inputs: true,
                    fixed_regs: true,
                    fixed_nonallocatable: true,
                    clobbers: false,
                    reftypes: false,
                },
            )?,
        })
    }
}

fuzz_target!(|testcase: TestCase| {
    let func = testcase.func;
    let _ = env_logger::try_init();
    log::trace!("func:\n{:?}", func);
    let env = regalloc2::fuzzing::func::machine_env();
    let out =
        regalloc2::fuzzing::fastalloc::run(&func, &env, true, false).expect("regalloc did not succeed");

    let mut checker = Checker::new(&func, &env);
    checker.prepare(&out);
    checker.run().expect("checker failed");
});
