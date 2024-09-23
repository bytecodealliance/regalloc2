/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

//! Fast postorder computation.

use crate::{Block, VecExt};
use alloc::vec::Vec;
use smallvec::{smallvec, SmallVec};

pub fn calculate<'a, SuccFn: Fn(Block) -> &'a [Block]>(
    num_blocks: usize,
    entry: Block,
    visited_scratch: &mut Vec<bool>,
    out: &mut Vec<Block>,
    succ_blocks: SuccFn,
) {
    let out = out.repopuate(num_blocks, Block::invalid());
    // State: visited-block map, and explicit DFS stack.
    let visited = visited_scratch.repopuate(num_blocks, false);
    calculate_soa(entry, visited, out, succ_blocks);
}

pub fn calculate_soa<'a, SuccFn: Fn(Block) -> &'a [Block]>(
    entry: Block,
    visited: &mut [bool],
    out: &mut [Block],
    succ_blocks: SuccFn,
) {
    struct State<'a> {
        block: Block,
        succs: core::slice::Iter<'a, Block>,
    }

    let mut stack: SmallVec<[State; 64]> = smallvec![];
    let mut out_iter = out.iter_mut();

    visited[entry.index()] = true;
    stack.push(State {
        block: entry,
        succs: succ_blocks(entry).iter(),
    });

    while let Some(ref mut state) = stack.last_mut() {
        // Perform one action: push to new succ, skip an already-visited succ, or pop.
        if let Some(&succ) = state.succs.next() {
            if !visited[succ.index()] {
                visited[succ.index()] = true;
                stack.push(State {
                    block: succ,
                    succs: succ_blocks(succ).iter(),
                });
            }
        } else {
            *out_iter.next().unwrap() = state.block;
            stack.pop();
        }
    }
}
