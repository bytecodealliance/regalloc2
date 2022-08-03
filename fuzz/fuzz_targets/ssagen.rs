/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

#![no_main]
use regalloc2::fuzzing::arbitrary::{Arbitrary, Result, Unstructured};
use regalloc2::fuzzing::cfg::CFGInfo;
use regalloc2::fuzzing::func::{Func, Options};
use regalloc2::fuzzing::fuzz_target;
use regalloc2::fuzzing::ssa::validate_ssa;

#[derive(Debug)]
struct TestCase {
    f: Func,
}

impl Arbitrary<'_> for TestCase {
    fn arbitrary(u: &mut Unstructured) -> Result<Self> {
        Ok(TestCase {
            f: Func::arbitrary_with_options(
                u,
                &Options {
                    reused_inputs: true,
                    fixed_regs: true,
                    clobbers: true,
                    control_flow: true,
                    reducible: false,
                    always_local_uses: false,
                    block_params: true,
                    reftypes: true,
                },
            )?,
        })
    }
}

fuzz_target!(|t: TestCase| {
    let cfginfo = CFGInfo::new(&t.f).expect("could not create CFG info");
    validate_ssa(&t.f, &cfginfo).expect("invalid SSA");
});
