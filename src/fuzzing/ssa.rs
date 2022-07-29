/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

//! SSA-related utilities.

use crate::cfg::CFGInfo;
use crate::{Block, Function, Inst, OperandKind, RegAllocError};
use std::collections::HashMap;

pub fn validate_ssa<F: Function>(f: &F, cfginfo: &CFGInfo) -> Result<(), RegAllocError> {
    // Breadth-first traversal of the dominance tree, without recursion (to avoid stack overflow in
    // case of a deep dominance tree) and in O(num_blocks) time. Note that if any blocks are
    // unreachable from the entry block, they will not be validated.
    let blocks = {
        let mut blocks = Vec::with_capacity(f.num_blocks());
        // First invert the dominance tree so we know, for each block, all the blocks that it
        // immediately dominates.
        let mut domtree = HashMap::new();
        for (child, &parent) in cfginfo.domtree.iter().enumerate() {
            if !parent.is_invalid() {
                let child = Block::new(child);
                domtree.entry(parent).or_insert_with(Vec::new).push(child);
            }
        }

        // Then concatenate those lists of immediately-dominated blocks in order.
        blocks.push(f.entry_block());
        for i in 0.. {
            if let Some(parent) = blocks.get(i) {
                if let Some(mut children) = domtree.remove(parent) {
                    blocks.append(&mut children);
                }
            } else {
                break;
            }
        }
        blocks
    };

    // Visit each block once, only after its dominators have been visited. Check, for every use,
    // that the def is either in the same block in an earlier inst, or is defined (by inst or
    // blockparam) in some other block that dominates this one. Also check that for every block
    // param and inst def, that this is the only def.
    let mut defined_in = vec![Block::invalid(); f.num_vregs()];
    for block in blocks {
        for blockparam in f.block_params(block) {
            let def_block = &mut defined_in[blockparam.vreg()];
            if !def_block.is_invalid() {
                return Err(RegAllocError::SSA(*blockparam, Inst::invalid()));
            }
            *def_block = block;
        }
        for iix in f.block_insns(block).iter() {
            let operands = f.inst_operands(iix);
            for operand in operands {
                let def_block = &mut defined_in[operand.vreg().vreg()];
                match operand.kind() {
                    OperandKind::Use => {
                        if def_block.is_invalid() {
                            return Err(RegAllocError::SSA(operand.vreg(), iix));
                        }
                        if !cfginfo.dominates(*def_block, block) {
                            return Err(RegAllocError::SSA(operand.vreg(), iix));
                        }
                    }
                    OperandKind::Def => {
                        if !def_block.is_invalid() {
                            return Err(RegAllocError::SSA(operand.vreg(), iix));
                        }
                        *def_block = block;
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
