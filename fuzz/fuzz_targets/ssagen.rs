/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

#![no_main]
use regalloc2::fuzzing::arbitrary::{Arbitrary, Result, Unstructured};
use regalloc2::fuzzing::cfg::CFGInfo;
use regalloc2::fuzzing::func::{Func, Options};
use regalloc2::fuzzing::fuzz_target;
use regalloc2::ssa::validate_ssa;

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
                    fixed_nonallocatable: true,
                    clobbers: true,
                    reftypes: true,
                },
            )?,
        })
    }
}

fuzz_target!(|t: TestCase| {
    thread_local! {
        // we thets that ctx is cleared properly between runs
        static CFG_INFO: std::cell::RefCell<CFGInfo> = std::cell::RefCell::default();
    }

    CFG_INFO.with(|cfginfo| {
        cfginfo
            .borrow_mut()
            .init(&t.f)
            .expect("could not create CFG info");
        validate_ssa(&t.f, &cfginfo.borrow()).expect("invalid SSA");
    });
});
