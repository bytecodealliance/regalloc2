//! Lightweight CFG analyses.

use crate::{domtree, postorder, Block, Function, Inst, OperandKind, ProgPoint};

#[derive(Clone, Debug)]
pub struct CFGInfo {
    /// Postorder traversal of blocks.
    pub postorder: Vec<Block>,
    /// Domtree parents, indexed by block.
    pub domtree: Vec<Block>,
    /// For each instruction, the block it belongs to.
    pub insn_block: Vec<Block>,
    /// For each vreg, the instruction that defines it, if any.
    pub vreg_def_inst: Vec<Inst>,
    /// For each vreg, the block that defines it as a blockparam, if
    /// any. (Every vreg must have a valid entry in either
    /// `vreg_def_inst` or `vreg_def_blockparam`.)
    pub vreg_def_blockparam: Vec<(Block, u32)>,
    /// For each block, the first instruction.
    pub block_entry: Vec<ProgPoint>,
    /// For each block, the last instruction.
    pub block_exit: Vec<ProgPoint>,
    /// For each block, what is its position in its successor's preds,
    /// if it has a single successor?
    ///
    /// (Because we require split critical edges, we always either have a single
    /// successor (which itself may have multiple preds), or we have multiple
    /// successors but each successor itself has only one pred; so we can store
    /// just one value per block and always know any block's position in its
    /// successors' preds lists.)
    pub pred_pos: Vec<usize>,
}

impl CFGInfo {
    pub fn new<F: Function>(f: &F) -> CFGInfo {
        let postorder =
            postorder::calculate(f.blocks(), f.entry_block(), |block| f.block_succs(block));
        let domtree = domtree::calculate(
            f.blocks(),
            |block| f.block_preds(block),
            &postorder[..],
            f.entry_block(),
        );
        let mut insn_block = vec![Block::invalid(); f.insts()];
        let mut vreg_def_inst = vec![Inst::invalid(); f.num_vregs()];
        let mut vreg_def_blockparam = vec![(Block::invalid(), 0); f.num_vregs()];
        let mut block_entry = vec![ProgPoint::before(Inst::invalid()); f.blocks()];
        let mut block_exit = vec![ProgPoint::before(Inst::invalid()); f.blocks()];
        let mut pred_pos = vec![0; f.blocks()];

        for block in 0..f.blocks() {
            let block = Block::new(block);
            for (i, param) in f.block_params(block).iter().enumerate() {
                vreg_def_blockparam[param.vreg()] = (block, i as u32);
            }
            for inst in f.block_insns(block).iter() {
                insn_block[inst.index()] = block;
                for operand in f.inst_operands(inst) {
                    match operand.kind() {
                        OperandKind::Def => {
                            vreg_def_inst[operand.vreg().vreg()] = inst;
                        }
                        _ => {}
                    }
                }
            }
            block_entry[block.index()] = ProgPoint::before(f.block_insns(block).first());
            block_exit[block.index()] = ProgPoint::after(f.block_insns(block).last());

            if f.block_preds(block).len() > 1 {
                for (i, &pred) in f.block_preds(block).iter().enumerate() {
                    // Assert critical edge condition.
                    assert_eq!(
                        f.block_succs(pred).len(),
                        1,
                        "Edge {} -> {} is critical",
                        pred.index(),
                        block.index(),
                    );
                    pred_pos[pred.index()] = i;
                }
            }
        }

        CFGInfo {
            postorder,
            domtree,
            insn_block,
            vreg_def_inst,
            vreg_def_blockparam,
            block_entry,
            block_exit,
            pred_pos,
        }
    }

    pub fn dominates(&self, a: Block, b: Block) -> bool {
        domtree::dominates(&self.domtree[..], a, b)
    }

    /// Return the position of this block in its successor's predecessor list.
    ///
    /// Because the CFG must have split critical edges, we actually do not need
    /// to know *which* successor: if there is more than one, then each
    /// successor has only one predecessor (that's this block), so the answer is
    /// `0` no matter which successor we are considering.
    pub fn pred_position(&self, block: Block) -> usize {
        self.pred_pos[block.index()]
    }
}
