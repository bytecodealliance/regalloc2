#![no_main]
use libfuzzer_sys::arbitrary::{Arbitrary, Result, Unstructured};
use libfuzzer_sys::fuzz_target;

use regalloc2::moves::ParallelMoves;
use regalloc2::{Allocation, PReg, RegClass};
use std::collections::HashSet;

#[derive(Clone, Debug)]
struct TestCase {
    moves: Vec<(Allocation, Allocation)>,
}

impl Arbitrary for TestCase {
    fn arbitrary(u: &mut Unstructured) -> Result<Self> {
        let mut ret = TestCase { moves: vec![] };
        let mut written = HashSet::new();
        while bool::arbitrary(u)? {
            let reg1 = u.int_in_range(0..=30)?;
            let reg2 = u.int_in_range(0..=30)?;
            if written.contains(&reg2) {
                break;
            }
            written.insert(reg2);
            ret.moves.push((
                Allocation::reg(PReg::new(reg1, RegClass::Int)),
                Allocation::reg(PReg::new(reg2, RegClass::Int)),
            ));
        }
        Ok(ret)
    }
}

fuzz_target!(|testcase: TestCase| {
    let _ = env_logger::try_init();
    let scratch = Allocation::reg(PReg::new(31, RegClass::Int));
    let mut par = ParallelMoves::new(scratch);
    for &(src, dst) in &testcase.moves {
        par.add(src, dst);
    }
    let moves = par.resolve();

    // Compute the final source reg for each dest reg in the original
    // parallel-move set.
    let mut final_src_per_dest: Vec<Option<usize>> = vec![None; 32];
    for &(src, dst) in &testcase.moves {
        if let (Some(preg_src), Some(preg_dst)) = (src.as_reg(), dst.as_reg()) {
            final_src_per_dest[preg_dst.hw_enc()] = Some(preg_src.hw_enc());
        }
    }

    // Simulate the sequence of moves.
    let mut regfile: Vec<Option<usize>> = vec![None; 32];
    for i in 0..32 {
        regfile[i] = Some(i);
    }
    for (src, dst) in moves {
        if let (Some(preg_src), Some(preg_dst)) = (src.as_reg(), dst.as_reg()) {
            let data = regfile[preg_src.hw_enc()];
            regfile[preg_dst.hw_enc()] = data;
        } else {
            panic!("Bad allocation in move list");
        }
    }

    // Assert that the expected register-moves occurred.
    // N.B.: range up to 31 (not 32) to skip scratch register.
    for i in 0..31 {
        if let Some(orig_src) = final_src_per_dest[i] {
            assert_eq!(regfile[i], Some(orig_src));
        } else {
            // Should be untouched.
            assert_eq!(regfile[i], Some(i));
        }
    }
});
