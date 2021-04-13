//! SSA-related utilities.

use crate::cfg::CFGInfo;

use crate::{Block, Function, Inst, OperandKind, RegAllocError};

pub fn validate_ssa<F: Function>(f: &F, cfginfo: &CFGInfo) -> Result<(), RegAllocError> {
    // Walk the blocks in arbitrary order. Check, for every use, that
    // the def is either in the same block in an earlier inst, or is
    // defined (by inst or blockparam) in some other block that
    // dominates this one. Also check that for every block param and
    // inst def, that this is the only def.
    let mut defined = vec![false; f.num_vregs()];
    for block in 0..f.blocks() {
        let block = Block::new(block);
        for blockparam in f.block_params(block) {
            if defined[blockparam.vreg()] {
                return Err(RegAllocError::SSA(*blockparam, Inst::invalid()));
            }
            defined[blockparam.vreg()] = true;
        }
        for iix in f.block_insns(block).iter() {
            let operands = f.inst_operands(iix);
            for operand in operands {
                match operand.kind() {
                    OperandKind::Use => {
                        let def_block = if cfginfo.vreg_def_inst[operand.vreg().vreg()].is_valid() {
                            cfginfo.insn_block[cfginfo.vreg_def_inst[operand.vreg().vreg()].index()]
                        } else {
                            cfginfo.vreg_def_blockparam[operand.vreg().vreg()].0
                        };
                        if def_block.is_invalid() {
                            return Err(RegAllocError::SSA(operand.vreg(), iix));
                        }
                        if !cfginfo.dominates(def_block, block) {
                            return Err(RegAllocError::SSA(operand.vreg(), iix));
                        }
                    }
                    OperandKind::Def => {
                        if defined[operand.vreg().vreg()] {
                            return Err(RegAllocError::SSA(operand.vreg(), iix));
                        }
                        defined[operand.vreg().vreg()] = true;
                    }
                }
            }
        }
    }

    // Check that the length of branch args matches the sum of the
    // number of blockparams in their succs, and that the end of every
    // block ends in this branch or in a ret, and that there are no
    // other branches or rets in the middle of the block.
    for block in 0..f.blocks() {
        let block = Block::new(block);
        let insns = f.block_insns(block);
        for insn in insns.iter() {
            if insn == insns.last() {
                if !(f.is_branch(insn) || f.is_ret(insn)) {
                    return Err(RegAllocError::BB(block));
                }
                if f.is_branch(insn) {
                    let expected = f
                        .block_succs(block)
                        .iter()
                        .map(|&succ| f.block_params(succ).len())
                        .sum();
                    if f.inst_operands(insn).len() != expected {
                        return Err(RegAllocError::Branch(insn));
                    }
                }
            } else {
                if f.is_branch(insn) || f.is_ret(insn) {
                    return Err(RegAllocError::BB(block));
                }
            }
        }
    }

    // Check that the entry block has no block args: otherwise it is
    // undefined what their value would be.
    if f.block_params(f.entry_block()).len() > 0 {
        return Err(RegAllocError::BB(f.entry_block()));
    }

    Ok(())
}
