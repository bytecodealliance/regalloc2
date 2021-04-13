#![no_main]
use libfuzzer_sys::arbitrary::{Arbitrary, Result, Unstructured};
use libfuzzer_sys::fuzz_target;

use regalloc2::cfg::CFGInfo;
use regalloc2::fuzzing::func::{Func, Options};
use regalloc2::ssa::validate_ssa;

#[derive(Debug)]
struct TestCase {
    f: Func,
}

impl Arbitrary for TestCase {
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
                },
            )?,
        })
    }
}

fuzz_target!(|t: TestCase| {
    let cfginfo = CFGInfo::new(&t.f);
    validate_ssa(&t.f, &cfginfo).expect("invalid SSA");
});
