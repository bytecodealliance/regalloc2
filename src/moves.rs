use crate::Allocation;
use smallvec::{smallvec, SmallVec};

pub type MoveVec = SmallVec<[(Allocation, Allocation); 16]>;

/// A `ParallelMoves` represents a list of alloc-to-alloc moves that
/// must happen in parallel -- i.e., all reads of sources semantically
/// happen before all writes of destinations, and destinations are
/// allowed to overwrite sources. It can compute a list of sequential
/// moves that will produce the equivalent data movement, possibly
/// using a scratch register if one is necessary.
pub struct ParallelMoves {
    parallel_moves: MoveVec,
    scratch: Allocation,
}

impl ParallelMoves {
    pub fn new(scratch: Allocation) -> Self {
        Self {
            parallel_moves: smallvec![],
            scratch,
        }
    }

    pub fn add(&mut self, from: Allocation, to: Allocation) {
        self.parallel_moves.push((from, to));
    }

    fn sources_overlap_dests(&self) -> bool {
        // Assumes `parallel_moves` has already been sorted in `resolve()` below.
        for &(_, dst) in &self.parallel_moves {
            if self
                .parallel_moves
                .binary_search_by_key(&dst, |&(src, _)| src)
                .is_ok()
            {
                return true;
            }
        }
        false
    }

    pub fn resolve(mut self) -> MoveVec {
        // Easy case: zero or one move. Just return our vec.
        if self.parallel_moves.len() <= 1 {
            return self.parallel_moves;
        }

        // Sort moves by source so that we can efficiently test for
        // presence.
        self.parallel_moves.sort();

        // Do any dests overlap sources? If not, we can also just
        // return the list.
        if !self.sources_overlap_dests() {
            return self.parallel_moves;
        }

        // General case: some moves overwrite dests that other moves
        // read as sources. We'll use a general algorithm.
        //
        // *Important property*: because we expect that each register
        // has only one writer (otherwise the effect of the parallel
        // move is undefined), each move can only block one other move
        // (with its one source corresponding to the one writer of
        // that source). Thus, we *can only have simple cycles*: there
        // are no SCCs that are more complex than that. We leverage
        // this fact below to avoid having to do a full Tarjan SCC DFS
        // (with lowest-index computation, etc.): instead, as soon as
        // we find a cycle, we know we have the full cycle and we can
        // do a cyclic move sequence and continue.

        // Sort moves by destination and check that each destination
        // has only one writer.
        self.parallel_moves.sort_by_key(|&(_, dst)| dst);
        if cfg!(debug) {
            let mut last_dst = None;
            for &(_, dst) in &self.parallel_moves {
                if last_dst.is_some() {
                    assert!(last_dst.unwrap() != dst);
                }
                last_dst = Some(dst);
            }
        }

        // Construct a mapping from move indices to moves they must
        // come before. Any given move must come before a move that
        // overwrites its destination; we have moves sorted by dest
        // above so we can efficiently find such a move, if any.
        let mut must_come_before: SmallVec<[Option<usize>; 16]> =
            smallvec![None; self.parallel_moves.len()];
        for (i, &(src, _)) in self.parallel_moves.iter().enumerate() {
            if let Ok(move_to_dst_idx) = self
                .parallel_moves
                .binary_search_by_key(&src, |&(_, dst)| dst)
            {
                must_come_before[i] = Some(move_to_dst_idx);
            }
        }

        // Do a simple stack-based DFS and emit moves in postorder,
        // then reverse at the end for RPO. Unlike Tarjan's SCC
        // algorithm, we can emit a cycle as soon as we find one, as
        // noted above.
        let mut ret: MoveVec = smallvec![];
        let mut stack: SmallVec<[usize; 16]> = smallvec![];
        let mut visited: SmallVec<[bool; 16]> = smallvec![false; self.parallel_moves.len()];
        let mut onstack: SmallVec<[bool; 16]> = smallvec![false; self.parallel_moves.len()];

        stack.push(0);
        onstack[0] = true;
        loop {
            if stack.is_empty() {
                if let Some(next) = visited.iter().position(|&flag| !flag) {
                    stack.push(next);
                    onstack[next] = true;
                } else {
                    break;
                }
            }

            let top = *stack.last().unwrap();
            visited[top] = true;
            match must_come_before[top] {
                None => {
                    ret.push(self.parallel_moves[top]);
                    onstack[top] = false;
                    stack.pop();
                    while let Some(top) = stack.pop() {
                        ret.push(self.parallel_moves[top]);
                        onstack[top] = false;
                    }
                }
                Some(next) if visited[next] && !onstack[next] => {
                    ret.push(self.parallel_moves[top]);
                    onstack[top] = false;
                    stack.pop();
                    while let Some(top) = stack.pop() {
                        ret.push(self.parallel_moves[top]);
                        onstack[top] = false;
                    }
                }
                Some(next) if !visited[next] && !onstack[next] => {
                    stack.push(next);
                    onstack[next] = true;
                    continue;
                }
                Some(next) => {
                    // Found a cycle -- emit a cyclic-move sequence
                    // for the cycle on the top of stack, then normal
                    // moves below it. Recall that these moves will be
                    // reversed in sequence, so from the original
                    // parallel move set
                    //
                    //     { B := A, C := B, A := B }
                    //
                    // we will generate something like:
                    //
                    //     A := scratch
                    //     B := A
                    //     C := B
                    //     scratch := C
                    //
                    // which will become:
                    //
                    //     scratch := C
                    //     C := B
                    //     B := A
                    //     A := scratch
                    let mut last_dst = None;
                    let mut scratch_src = None;
                    while let Some(move_idx) = stack.pop() {
                        onstack[move_idx] = false;
                        let (mut src, dst) = self.parallel_moves[move_idx];
                        if last_dst.is_none() {
                            scratch_src = Some(src);
                            src = self.scratch;
                        } else {
                            assert_eq!(last_dst.unwrap(), src);
                        }
                        ret.push((src, dst));

                        last_dst = Some(dst);

                        if move_idx == next {
                            break;
                        }
                    }
                    if let Some(src) = scratch_src {
                        ret.push((src, self.scratch));
                    }
                }
            }
        }

        ret.reverse();
        ret
    }
}
