/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

//! Lightweight CFG analyses.

use crate::alloc::vec::Vec;
use alloc::Layout;

use crate::{domtree, postorder, Block, Function, Inst, ProgPoint, RegAllocError, VecExt};
use allocator_api2::alloc;
use smallvec::{smallvec, SmallVec};

macro_rules! decl_soa {
    (
        #[derive(Soa $($derived:tt)*)]
        #[soa(slice = $slice_name:ident)]
        #[soa(slice_mut = $slice_mut_name:ident)]
        $vis:vis struct $name:ident {
            $(
                #[soa(default = $fill:expr)]
                $(#[$fmeta:meta])*
                $field:ident: $fty:ty,
            )*
        }
    ) => {
        #[derive($($derived)*)]
        $vis struct $name {
            $($(#[$fmeta])* $field: *mut $fty,)*
            cap: usize,
            len: usize,
        }

        $vis struct $slice_mut_name<'a> {
            $($(#[$fmeta])* $vis $field: &'a mut [$fty],)*
        }

        $vis struct $slice_name<'a> {
            $($(#[$fmeta])* $vis $field: &'a [$fty],)*
        }

        impl $name {
            fn resize(&mut self, len: usize) {
                let layout = Layout::new::<()>();

                $(let (layout, $field) = layout.extend(Layout::array::<Block>(len).unwrap()).unwrap();)*

                let bytes = if self.cap < len {
                    unsafe { crate::alloc::alloc::realloc(self.base_ptr(), Self::layout(self.cap), layout.size()) }
                } else {
                    self.base_ptr()
                };

                self.cap = len.max(self.cap);
                self.len = len;

                unsafe {
                    $(self.$field = bytes.add($field) as _;)*
                }

                unsafe {
                    $(core::slice::from_raw_parts_mut(self.$field, self.len).fill($fill);)*
                }
            }

            $vis fn slice(&self) -> $slice_name {
                unsafe {
                    $slice_name {
                        $($field: core::slice::from_raw_parts(self.$field, self.len),)*
                    }
                }
            }

            $vis fn slice_mut(&mut self) -> $slice_mut_name {
                unsafe {
                    $slice_mut_name {
                        $($field: core::slice::from_raw_parts_mut(self.$field, self.len),)*
                    }
                }
            }

            fn base_ptr(&mut self) -> *mut u8 {
                decl_soa!(@first_field self $($field)*) as _
            }

            fn layout(cap: usize) -> Layout {
                Layout::new::<()>()
                    $(.extend(Layout::array::<$fty>(cap).unwrap()).unwrap().0)*
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    $($field: core::ptr::null_mut(),)*
                    cap: 0,
                    len: 0,
                }
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                unsafe {
                    $(core::ptr::slice_from_raw_parts_mut(self.$field, self.len).drop_in_place();)*
                    crate::alloc::alloc::dealloc(self.base_ptr(), Self::layout(self.cap));
                }
            }
        }
    };

    (@first_field $self:ident $first:ident $($rest:ident)*) => { $self.$first };
}

decl_soa! {
    #[derive(Soa)]
    #[soa(slice = CfgInfoSlice)]
    #[soa(slice_mut = CfgInfoSliceMut)]
    pub struct CFGInfoSoa {
        #[soa(default = Block::invalid())]
        /// Postorder traversal of blocks.
        postorder: Block,
        #[soa(default = Block::invalid())]
        /// Domtree parents, indexed by block.
        domtree: Block,
        #[soa(default = ProgPoint::before(Inst::invalid()))]
        /// For each block, the first instruction.
        block_entry: ProgPoint,
        #[soa(default = ProgPoint::before(Inst::invalid()))]
        /// For each block, the last instruction.
        block_exit: ProgPoint,
        #[soa(default = 0)]
        /// For each block, what is the approximate loop depth?
        ///
        /// This measure is fully precise iff the input CFG is reducible
        /// and blocks are in RPO, so that loop backedges are precisely
        /// those whose block target indices are less than their source
        /// indices. Otherwise, it will be approximate, but should still
        /// be usable for heuristic purposes.
        approx_loop_depth: u32,

        #[soa(default = 0)]
        backedge_in: u32,
        #[soa(default = 0)]
        backedge_out: u32,
        #[soa(default = u32::MAX)]
        block_to_rpo_scratch: u32,
        #[soa(default = false)]
        visited_scratch: bool,
    }
}

#[derive(Default)]
pub struct CompactCFGInfo {
    soa: CFGInfoSoa,
    /// For each instruction, the block it belongs to.
    pub insn_block: Vec<Block>,
}

impl CompactCFGInfo {
    pub fn init<F: Function>(&mut self, f: &F) -> Result<(), RegAllocError> {
        self.soa.resize(f.num_blocks());
        let CfgInfoSliceMut {
            postorder,
            domtree,
            block_entry,
            block_exit,
            approx_loop_depth,
            backedge_in,
            backedge_out,
            block_to_rpo_scratch,
            visited_scratch,
        } = self.soa.slice_mut();

        postorder::calculate_soa(f.entry_block(), visited_scratch, postorder, |block| {
            f.block_succs(block)
        });

        domtree::calculate_soa(
            |block| f.block_preds(block),
            postorder,
            block_to_rpo_scratch,
            domtree,
            f.entry_block(),
        );

        let insn_block = self.insn_block.repopuate(f.num_insts(), Block::invalid());

        for block in 0..f.num_blocks() {
            let block = Block::new(block);
            for inst in f.block_insns(block).iter() {
                insn_block[inst.index()] = block;
            }
            block_entry[block.index()] = ProgPoint::before(f.block_insns(block).first());
            block_exit[block.index()] = ProgPoint::after(f.block_insns(block).last());

            // Check critical edge condition: if there is more than
            // one predecessor, each must have only one successor
            // (this block).
            let preds = f.block_preds(block).len() + if block == f.entry_block() { 1 } else { 0 };
            if preds > 1 {
                for &pred in f.block_preds(block) {
                    let succs = f.block_succs(pred).len();
                    if succs > 1 {
                        return Err(RegAllocError::CritEdge(pred, block));
                    }
                }
            }

            // Check branch-arg condition: if any successors have more
            // than one predecessor (given above, there will only be
            // one such successor), then the last instruction of this
            // block (the branch) cannot have any args other than the
            // blockparams.
            let mut require_no_branch_args = false;
            for &succ in f.block_succs(block) {
                let preds = f.block_preds(succ).len() + if succ == f.entry_block() { 1 } else { 0 };
                if preds > 1 {
                    require_no_branch_args = true;
                    break;
                }
            }
            if require_no_branch_args {
                let last = f.block_insns(block).last();
                if !f.inst_operands(last).is_empty() {
                    return Err(RegAllocError::DisallowedBranchArg(last));
                }
            }

            for &succ in f.block_succs(block) {
                if succ.index() <= block.index() {
                    backedge_in[succ.index()] += 1;
                    backedge_out[block.index()] += 1;
                }
            }
        }

        let mut approx_loop_depth = approx_loop_depth.iter_mut();
        let mut backedge_stack: SmallVec<[u32; 4]> = smallvec![];
        let mut cur_depth = 0;
        for block in 0..f.num_blocks() {
            if backedge_in[block] > 0 {
                cur_depth += 1;
                backedge_stack.push(backedge_in[block]);
            }

            *approx_loop_depth.next().unwrap() = cur_depth;

            while backedge_stack.len() > 0 && backedge_out[block] > 0 {
                backedge_out[block] -= 1;
                *backedge_stack.last_mut().unwrap() -= 1;
                if *backedge_stack.last().unwrap() == 0 {
                    cur_depth -= 1;
                    backedge_stack.pop();
                }
            }
        }

        Ok(())
    }

    pub fn slice(&self) -> CfgInfoSlice {
        self.soa.slice()
    }

    pub fn dominates(&self, a: Block, b: Block) -> bool {
        domtree::dominates(&self.slice().domtree[..], a, b)
    }
}

#[derive(Debug, Default)]
pub struct CFGInfo {
    /// Postorder traversal of blocks.
    pub postorder: Vec<Block>,
    /// Domtree parents, indexed by block.
    pub domtree: Vec<Block>,
    /// For each instruction, the block it belongs to.
    pub insn_block: Vec<Block>,
    /// For each block, the first instruction.
    pub block_entry: Vec<ProgPoint>,
    /// For each block, the last instruction.
    pub block_exit: Vec<ProgPoint>,
    /// For each block, what is the approximate loop depth?
    ///
    /// This measure is fully precise iff the input CFG is reducible
    /// and blocks are in RPO, so that loop backedges are precisely
    /// those whose block target indices are less than their source
    /// indices. Otherwise, it will be approximate, but should still
    /// be usable for heuristic purposes.
    pub approx_loop_depth: Vec<u32>,

    visited_scratch: Vec<bool>,
    block_to_rpo_scratch: Vec<u32>,
    backedge_scratch: Vec<u32>,
}

impl CFGInfo {
    pub fn init<F: Function>(&mut self, f: &F) -> Result<(), RegAllocError> {
        let nb = f.num_blocks();

        postorder::calculate(
            nb,
            f.entry_block(),
            &mut self.visited_scratch,
            &mut self.postorder,
            |block| f.block_succs(block),
        );

        domtree::calculate(
            nb,
            |block| f.block_preds(block),
            &self.postorder,
            &mut self.block_to_rpo_scratch,
            &mut self.domtree,
            f.entry_block(),
        );

        let insn_block = self.insn_block.repopuate(f.num_insts(), Block::invalid());
        let block_entry = self
            .block_entry
            .repopuate(nb, ProgPoint::before(Inst::invalid()));
        let block_exit = self
            .block_exit
            .repopuate(nb, ProgPoint::before(Inst::invalid()));
        let (backedge_in, backedge_out) =
            self.backedge_scratch.repopuate(nb * 2, 0).split_at_mut(nb);

        for block in 0..f.num_blocks() {
            let block = Block::new(block);
            for inst in f.block_insns(block).iter() {
                insn_block[inst.index()] = block;
            }
            block_entry[block.index()] = ProgPoint::before(f.block_insns(block).first());
            block_exit[block.index()] = ProgPoint::after(f.block_insns(block).last());

            // Check critical edge condition: if there is more than
            // one predecessor, each must have only one successor
            // (this block).
            let preds = f.block_preds(block).len() + if block == f.entry_block() { 1 } else { 0 };
            if preds > 1 {
                for &pred in f.block_preds(block) {
                    let succs = f.block_succs(pred).len();
                    if succs > 1 {
                        return Err(RegAllocError::CritEdge(pred, block));
                    }
                }
            }

            // Check branch-arg condition: if any successors have more
            // than one predecessor (given above, there will only be
            // one such successor), then the last instruction of this
            // block (the branch) cannot have any args other than the
            // blockparams.
            let mut require_no_branch_args = false;
            for &succ in f.block_succs(block) {
                let preds = f.block_preds(succ).len() + if succ == f.entry_block() { 1 } else { 0 };
                if preds > 1 {
                    require_no_branch_args = true;
                    break;
                }
            }
            if require_no_branch_args {
                let last = f.block_insns(block).last();
                if !f.inst_operands(last).is_empty() {
                    return Err(RegAllocError::DisallowedBranchArg(last));
                }
            }

            for &succ in f.block_succs(block) {
                if succ.index() <= block.index() {
                    backedge_in[succ.index()] += 1;
                    backedge_out[block.index()] += 1;
                }
            }
        }

        let approx_loop_depth = self.approx_loop_depth.cleared();
        let mut backedge_stack: SmallVec<[u32; 4]> = smallvec![];
        let mut cur_depth = 0;
        for block in 0..nb {
            if backedge_in[block] > 0 {
                cur_depth += 1;
                backedge_stack.push(backedge_in[block]);
            }

            approx_loop_depth.push(cur_depth);

            while backedge_stack.len() > 0 && backedge_out[block] > 0 {
                backedge_out[block] -= 1;
                *backedge_stack.last_mut().unwrap() -= 1;
                if *backedge_stack.last().unwrap() == 0 {
                    cur_depth -= 1;
                    backedge_stack.pop();
                }
            }
        }

        Ok(())
    }

    pub fn dominates(&self, a: Block, b: Block) -> bool {
        domtree::dominates(&self.domtree[..], a, b)
    }
}
