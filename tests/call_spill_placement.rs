//! Values live across a call after a loop should spill at the call, not on
//! every iteration. Branch arguments are implicit uses that trimming must
//! account for when deciding where to end a register allocation.
use regalloc2::*;

struct LoopThenCall {
    operands: Vec<Vec<Operand>>,
    ranges: Vec<InstRange>,
    preds: Vec<Vec<Block>>,
    succs: Vec<Vec<Block>>,
    params: Vec<Vec<VReg>>,
    args: Vec<Vec<Vec<VReg>>>,
    call: usize,
}

impl Function for LoopThenCall {
    fn num_insts(&self) -> usize {
        self.operands.len()
    }
    fn num_blocks(&self) -> usize {
        self.ranges.len()
    }
    fn entry_block(&self) -> Block {
        Block::new(0)
    }
    fn block_insns(&self, b: Block) -> InstRange {
        self.ranges[b.index()]
    }
    fn block_preds(&self, b: Block) -> &[Block] {
        &self.preds[b.index()]
    }
    fn block_succs(&self, b: Block) -> &[Block] {
        &self.succs[b.index()]
    }
    fn block_params(&self, b: Block) -> &[VReg] {
        &self.params[b.index()]
    }
    fn branch_blockparams(&self, b: Block, _: Inst, s: usize) -> &[VReg] {
        &self.args[b.index()][s]
    }
    fn is_ret(&self, i: Inst) -> bool {
        i.index() == self.num_insts() - 1
    }
    fn is_branch(&self, i: Inst) -> bool {
        !self.is_ret(i) && self.ranges.iter().any(|r| r.last() == i)
    }
    fn inst_operands(&self, i: Inst) -> &[Operand] {
        &self.operands[i.index()]
    }
    fn inst_clobbers(&self, i: Inst) -> PRegSet {
        if i.index() == self.call {
            [PReg::new(0, RegClass::Float), PReg::new(1, RegClass::Float)]
                .iter()
                .copied()
                .collect()
        } else {
            PRegSet::empty()
        }
    }
    fn num_vregs(&self) -> usize {
        3
    }
    fn spillslot_size(&self, _: RegClass) -> usize {
        1
    }
}

fn check_loop(threshold: bool) {
    let v = |i| VReg::new(i, RegClass::Float);
    let b = Block::new;
    let r = |a, z| InstRange::new(Inst::new(a), Inst::new(z));
    let def = |i| Operand::reg_def(v(i));
    let use_ = |i| Operand::reg_use(v(i));
    let ret = Operand::reg_fixed_use(
        v(if threshold { 2 } else { 1 }),
        PReg::new(0, RegClass::Float),
    );
    // Both layouts have entry -> header, a backedge block, and an exit
    // containing a call and return. The threshold form updates in the header;
    // the counted form updates only on the backedge.
    let func = if threshold {
        LoopThenCall {
            operands: vec![
                vec![def(0)],
                vec![],
                vec![use_(1), def(2)],
                vec![use_(2)],
                vec![],
                vec![],
                vec![],
                vec![ret],
            ],
            ranges: vec![r(0, 2), r(2, 5), r(5, 6), r(6, 8)],
            preds: vec![vec![], vec![b(0), b(2)], vec![b(1)], vec![b(1)]],
            succs: vec![vec![b(1)], vec![b(2), b(3)], vec![b(1)], vec![]],
            params: vec![vec![], vec![v(1)], vec![], vec![]],
            args: vec![
                vec![vec![v(0)]],
                vec![vec![], vec![]],
                vec![vec![v(2)]],
                vec![],
            ],
            call: 6,
        }
    } else {
        LoopThenCall {
            operands: vec![
                vec![def(0)],
                vec![],
                vec![],
                vec![use_(1), def(2)],
                vec![],
                vec![],
                vec![ret],
            ],
            ranges: vec![r(0, 2), r(2, 3), r(3, 5), r(5, 7)],
            preds: vec![vec![], vec![b(0), b(2)], vec![b(1)], vec![b(1)]],
            succs: vec![vec![b(1)], vec![b(2), b(3)], vec![b(1)], vec![]],
            params: vec![vec![], vec![v(1)], vec![], vec![]],
            args: vec![
                vec![vec![v(0)]],
                vec![vec![], vec![]],
                vec![vec![v(2)]],
                vec![],
            ],
            call: 5,
        }
    };
    let env = MachineEnv {
        preferred_regs_by_class: [
            PRegSet::empty(),
            [PReg::new(0, RegClass::Float), PReg::new(1, RegClass::Float)]
                .iter()
                .copied()
                .collect(),
            PRegSet::empty(),
        ],
        non_preferred_regs_by_class: [PRegSet::empty(); 3],
        scratch_by_class: [None; 3],
        fixed_stack_slots: vec![],
    };
    let output = run(
        &func,
        &env,
        &RegallocOptions {
            validate_ssa: true,
            ..RegallocOptions::default()
        },
    )
    .unwrap();
    let mut checker = checker::Checker::new(&func, &env);
    checker.prepare(&output);
    checker.run().unwrap();
    let mut stores = 0;
    let mut loads = 0;
    for (point, edit) in &output.edits {
        let &Edit::Move { from, to } = edit;
        if from.is_stack() || to.is_stack() {
            assert!(
                point.inst().index() >= func.call,
                "stack traffic in loop: {:?}: {:?}",
                point,
                edit
            );
            stores += usize::from(to.is_stack());
            loads += usize::from(from.is_stack());
        }
    }
    assert_eq!((stores, loads), (1, 1));
}

#[test]
fn counted_loop_then_call() {
    check_loop(false);
}

#[test]
fn threshold_loop_then_call() {
    check_loop(true);
}
