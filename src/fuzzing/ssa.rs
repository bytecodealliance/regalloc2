/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

//! SSA-related utilities.

use crate::cfg::CFGInfo;

use crate::{Block, Function, Inst, OperandKind, RegAllocError};

pub fn validate_ssa<F: Function>(f: &F, cfginfo: &CFGInfo) -> Result<(), RegAllocError> {
    // For each vreg, the instruction that defines it, if any.
    let mut vreg_def_inst = vec![Inst::invalid(); f.num_vregs()];
    // For each vreg, the block that defines it as a blockparam, if
    // any. (Every vreg must have a valid entry in either
    // `vreg_def_inst` or `vreg_def_blockparam`.)
    let mut vreg_def_blockparam = vec![(Block::invalid(), 0); f.num_vregs()];

    for block in 0..f.num_blocks() {
        let block = Block::new(block);
        for (i, param) in f.block_params(block).iter().enumerate() {
            vreg_def_blockparam[param.vreg()] = (block, i as u32);
        }
        for inst in f.block_insns(block).iter() {
            for operand in f.inst_operands(inst) {
                match operand.kind() {
                    OperandKind::Def => {
                        vreg_def_inst[operand.vreg().vreg()] = inst;
                    }
                    _ => {}
                }
            }
        }
    }

    // Walk the blocks in arbitrary order. Check, for every use, that
    // the def is either in the same block in an earlier inst, or is
    // defined (by inst or blockparam) in some other block that
    // dominates this one. Also check that for every block param and
    // inst def, that this is the only def.
    let mut defined = vec![false; f.num_vregs()];
    for block in 0..f.num_blocks() {
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
                        let def_block = if vreg_def_inst[operand.vreg().vreg()].is_valid() {
                            cfginfo.insn_block[vreg_def_inst[operand.vreg().vreg()].index()]
                        } else {
                            vreg_def_blockparam[operand.vreg().vreg()].0
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
                    OperandKind::Mod => {
                        // Mod (modify) operands are not used in SSA,
                        // but can be used by non-SSA code (e.g. with
                        // the regalloc.rs compatibility shim).
                        return Err(RegAllocError::SSA(operand.vreg(), iix));
                    }
                }
            }
        }
    }

    // Check that the length of branch args matches the sum of the
    // number of blockparams in their succs, and that the end of every
    // block ends in this branch or in a ret, and that there are no
    // other branches or rets in the middle of the block.
    for block in 0..f.num_blocks() {
        let block = Block::new(block);
        let insns = f.block_insns(block);
        for insn in insns.iter() {
            if insn == insns.last() {
                if !(f.is_branch(insn) || f.is_ret(insn)) {
                    return Err(RegAllocError::BB(block));
                }
                if f.is_branch(insn) {
                    for (i, &succ) in f.block_succs(block).iter().enumerate() {
                        let blockparams_in = f.block_params(succ);
                        let blockparams_out = f.branch_blockparams(block, insn, i);
                        if blockparams_in.len() != blockparams_out.len() {
                            return Err(RegAllocError::Branch(insn));
                        }
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
