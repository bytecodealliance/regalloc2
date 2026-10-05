//! Fuzz the `ion` register allocator.

use crate::{checker, fuzzing::func, ion};
use arbitrary::{Arbitrary, Result, Unstructured};
use core::cell::RefCell;
use std::thread_local;

/// `ion`-specific options for generating functions.
const OPTIONS: func::Options = func::Options {
    reused_inputs: true,
    fixed_regs: true,
    fixed_nonallocatable: true,
    clobbers: true,
    fixed_def_clobbers: true,
    reftypes: true,
    callsite_ish_constraints: true,
    ..func::Options::DEFAULT
};

/// A convenience wrapper to generate a [`func::Func`] with `ion`-specific
/// options enabled.
#[derive(Clone, Debug)]
pub struct TestCase {
    func: func::Func,
    annotate: bool,
    check_ssa: bool,
}

impl Arbitrary<'_> for TestCase {
    fn arbitrary(u: &mut Unstructured) -> Result<TestCase> {
        let func = func::Func::arbitrary_with_options(u, &OPTIONS)?;
        let annotate = bool::arbitrary(u)?;
        let check_ssa = bool::arbitrary(u)?;
        Ok(TestCase {
            func,
            annotate,
            check_ssa,
        })
    }
}

/// Test a single function with the `ion` allocator.
///
/// This also:
/// - optionally creates annotations
/// - optionally verifies the incoming SSA
/// - runs the [`checker`].
pub fn check(t: TestCase) {
    let TestCase {
        func,
        annotate,
        check_ssa,
    } = &t;
    log::trace!("func:\n{func:?}");

    let env = func::machine_env();
    thread_local! {
        // We test that ctx is cleared properly between runs.
        static CTX: RefCell<ion::Ctx> = RefCell::default();
    }

    CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        let result = ion::run(func, &env, &mut ctx, *annotate, *check_ssa);
        if let Some(expected) = func.expected_fail() {
            // The oracle currently returns only the payload-free TooManyLiveRegs variant.
            assert!(
                matches!(&result, Err(actual) if core::mem::discriminant(actual) == core::mem::discriminant(&expected)),
                "expected {:?}, got {:?}",
                expected,
                result
            );
        } else {
            result.expect("regalloc did not succeed");
            let mut checker = checker::Checker::new(func, &env);
            checker.prepare(&ctx.output);
            checker.run().expect("checker failed");
        }
    });
}

#[test]
fn smoke() {
    arbtest::arbtest(|u| {
        let test_case = TestCase::arbitrary(u)?;
        check(test_case);
        Ok(())
    })
    .budget_ms(1_000);
}
