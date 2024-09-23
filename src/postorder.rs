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
    // State: visited-block map, and explicit DFS stack.
    struct State<'a> {
        block: Block,
        succs: core::slice::Iter<'a, Block>,
    }

    let visited = visited_scratch.repopuate(num_blocks, false);
    let mut stack: SmallVec<[State; 64]> = smallvec![];
    out.clear();

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
            out.push(state.block);
            stack.pop();
        }
    }
}
