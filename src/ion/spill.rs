/*
 * This file was initially derived from the files
 * `js/src/jit/BacktrackingAllocator.h` and
 * `js/src/jit/BacktrackingAllocator.cpp` in Mozilla Firefox, and was
 * originally licensed under the Mozilla Public License 2.0. We
 * subsequently relicensed it to Apache-2.0 WITH LLVM-exception (see
 * https://github.com/bytecodealliance/regalloc2/issues/7).
 *
 * Since the initial port, the design has been substantially evolved
 * and optimized.
 */

//! Spillslot allocation.

use super::{
    AllocRegResult, CodeRange, Env, LiveBundleIndex, LiveBundleVec, LiveRangeFlag, LiveRangeIndex,
    LiveRangeKey, LiveRangeList, LiveRangeListEntry, PRegIndex, RegTraversalIter, SpillSetIndex,
    SpillSlotData, SpillSlotIndex, UseList, VRegIndex,
};
use crate::{Allocation, Block, Function, PRegSet, ProgPoint, RegClass, SpillSlot};
use smallvec::SmallVec;

/// Which of the per-function data used by `try_allocating_regs_in_loops`
/// has been computed.
#[derive(Default)]
struct LoopInfoComputed {
    loop_runs: bool,
    clobber_points: [bool; 3],
}

impl<'a, F: Function> Env<'a, F> {
    pub fn try_allocating_regs_for_spilled_bundles(&mut self) {
        trace!("allocating regs for spilled bundles");
        let mut scratch = core::mem::take(&mut self.ctx.scratch_conflicts);
        let mut failed = core::mem::take(&mut self.ctx.scratch_bundle);
        failed.clear();
        self.ctx.any_in_spill_slot = false;
        for i in 0..self.ctx.spilled_bundles.len() {
            let bundle = self.ctx.spilled_bundles[i]; // don't borrow self

            if self.ctx.bundles[bundle].ranges.is_empty() {
                continue;
            }

            let class = self.ctx.spillsets[self.ctx.bundles[bundle].spillset].class;
            let hint = self.ctx.spillsets[self.ctx.bundles[bundle].spillset]
                .hint
                .as_valid();

            // This may be an empty-range bundle whose ranges are not
            // sorted; sort all range-lists again here.
            self.ctx.bundles[bundle]
                .ranges
                .sort_unstable_by_key(|entry| entry.range.from);

            let mut success = false;
            self.ctx.output.stats.spill_bundle_reg_probes += 1;
            let limit = self.bundles[bundle].limit.map(|l| l as usize);
            for preg in RegTraversalIter::new(self.env, class, None, hint, bundle.index(), limit) {
                trace!("trying bundle {:?} to preg {:?}", bundle, preg);
                let preg_idx = PRegIndex::new(preg.index());
                if let AllocRegResult::Allocated(_) =
                    self.try_to_allocate_bundle_to_reg(bundle, preg_idx, None, &mut scratch)
                {
                    self.ctx.output.stats.spill_bundle_reg_success += 1;
                    success = true;
                    break;
                }
            }

            if !success {
                trace!(
                    "spilling bundle {:?}: marking spillset {:?} as required",
                    bundle,
                    self.ctx.bundles[bundle].spillset
                );
                self.ctx.spillsets[self.ctx.bundles[bundle].spillset].required = true;
                failed.push(bundle);
            }
        }

        // Only once every spilled bundle has had its chance at a register
        // as a whole do we carve loops out of the ones that failed, so that
        // a loop region cannot take the register that a later bundle would
        // have fit in entirely.
        if !failed.is_empty() && self.ctx.cfginfo.approx_loop_depth.iter().any(|&d| d > 0) {
            let mut computed = LoopInfoComputed::default();
            for &bundle in &failed {
                self.try_allocating_regs_in_loops(bundle, &mut computed, &mut scratch);
            }
        }
        self.ctx.scratch_bundle = failed;
        self.ctx.scratch_conflicts = scratch;
    }

    /// Second chance for a spilled bundle that did not fit in a register
    /// as a whole, usually because it is live across a call that clobbers
    /// every register of its class: give a register to its pieces inside
    /// loops that contain uses, so that values used in a loop are not
    /// loaded from and stored to the stack on every iteration.
    ///
    /// Loops are tried outermost first, and a loop's pieces are only taken
    /// out of the spill bundle if they all fit in one register, so the
    /// moves this adds are at loop entries and exits. Pieces that are live
    /// across a point clobbering every register of the class, or whose vreg is
    /// neither used nor defined in the loop, are left on the stack.
    fn try_allocating_regs_in_loops(
        &mut self,
        bundle: LiveBundleIndex,
        computed: &mut LoopInfoComputed,
        conflicts: &mut LiveBundleVec,
    ) {
        let spillset = self.ctx.bundles[bundle].spillset;
        let class = self.ctx.spillsets[spillset].class;
        let hint = self.ctx.spillsets[spillset].hint.as_valid();
        let limit = self.ctx.bundles[bundle].limit.map(|l| l as usize);

        // Only bundles live across a call are of interest here; the others
        // failed for lack of registers, and their loops seldom have any free.
        if !computed.clobber_points[class as usize] {
            self.compute_clobber_points(class);
            computed.clobber_points[class as usize] = true;
        }
        let call_points = &self.ctx.scratch_call_points[class as usize];
        if !self.ctx.bundles[bundle]
            .ranges
            .iter()
            .any(|entry| any_point_in(call_points, entry.range))
        {
            return;
        }

        // Gather every use of the bundle's vregs once, sorted, so that the
        // searches below stay logarithmic however many ranges they have.
        let mut uses = core::mem::take(&mut self.ctx.scratch_loop_uses);
        uses.clear();
        let mut vregs: SmallVec<[VRegIndex; 4]> = self.ctx.bundles[bundle]
            .ranges
            .iter()
            .map(|entry| self.ctx.ranges[entry.index].vreg)
            .collect();
        vregs.sort_unstable();
        vregs.dedup();
        for &vreg in &vregs {
            for vreg_entry in &self.ctx.vregs[vreg].ranges {
                for u in &self.ctx.ranges[vreg_entry.index].uses {
                    uses.push((vreg, u.pos));
                }
            }
        }
        uses.sort_unstable();

        // Find the blocks in loops where the bundle's vregs are used, within
        // the bundle's ranges. Consecutive uses are mostly in one block.
        let mut use_blocks: SmallVec<[u32; 16]> = SmallVec::new();
        let ranges = &self.ctx.bundles[bundle].ranges;
        for &(vreg, pos) in &uses {
            let i = ranges.partition_point(|entry| entry.range.to <= pos);
            if i == ranges.len()
                || !ranges[i].range.contains_point(pos)
                || self.ctx.ranges[ranges[i].index].vreg != vreg
            {
                continue;
            }
            let block = self.ctx.cfginfo.insn_block[pos.inst().index()].raw_u32();
            if self.ctx.cfginfo.approx_loop_depth[block as usize] > 0
                && use_blocks.last() != Some(&block)
            {
                use_blocks.push(block);
            }
        }
        if use_blocks.is_empty() {
            self.ctx.scratch_loop_uses = uses;
            return;
        }
        if !computed.loop_runs {
            self.compute_loop_runs();
            computed.loop_runs = true;
        }

        // Collect the loops, at every depth, around each of those blocks.
        let mut regions = core::mem::take(&mut self.ctx.scratch_loop_regions);
        regions.clear();
        let depth = &self.ctx.cfginfo.approx_loop_depth;
        let runs = &mut self.ctx.scratch_loop_runs;
        let mut last_innermost = None;
        for &block in &use_blocks {
            let (mut start, mut end) = loop_run(depth, runs, block);
            if last_innermost == Some((start, end)) {
                continue;
            }
            last_innermost = Some((start, end));
            let mut d = depth[block as usize];
            loop {
                regions.push((d, start, end));
                let left = if start > 0 {
                    depth[start as usize - 1]
                } else {
                    0
                };
                let right = depth.get(end as usize + 1).copied().unwrap_or(0);
                let parent = core::cmp::max(left, right);
                if parent == 0 {
                    break;
                }
                let outer = if left >= right { start - 1 } else { end + 1 };
                (start, end) = loop_run(depth, runs, outer);
                d = parent;
            }
        }
        regions.sort_unstable();
        regions.dedup();

        let mut probe = LiveBundleIndex::invalid();
        let mut loop_pieces = SmallVec::new();
        let mut pieces: SmallVec<[(LiveRangeIndex, CodeRange); 4]> = SmallVec::new();
        for &(_, start, end) in &regions {
            let region = CodeRange {
                from: self.ctx.cfginfo.block_entry[start as usize],
                to: self.ctx.cfginfo.block_exit[end as usize].next(),
            };

            pieces.clear();
            let ranges = &self.ctx.bundles[bundle].ranges;
            let first = ranges.partition_point(|entry| entry.range.to <= region.from);
            let mut whole = first == 0;
            let mut past = first;
            for entry in &ranges[first..] {
                if entry.range.from >= region.to {
                    break;
                }
                past += 1;
                let piece = CodeRange {
                    from: core::cmp::max(entry.range.from, region.from),
                    to: core::cmp::min(entry.range.to, region.to),
                };
                if piece != entry.range {
                    whole = false;
                }
                // A value defined in the loop has no use in its piece, which
                // starts after the def, so look for uses in the whole loop.
                if !any_use_in(&uses, self.ctx.ranges[entry.index].vreg, region)
                    || any_point_in(&self.ctx.scratch_full_clobbers[class as usize], piece)
                {
                    whole = false;
                    continue;
                }
                pieces.push((entry.index, piece));
            }
            // Nothing to gain, or the whole bundle, which already failed.
            if pieces.is_empty() || (whole && past == ranges.len()) {
                continue;
            }

            if !probe.is_valid() {
                probe = self.ctx.bundles.add(self.ctx.scratch_bump.clone());
                self.ctx.bundles[probe].spillset = spillset;
            }
            self.ctx.bundles[probe].ranges.clear();
            for &(_, piece) in &pieces {
                // The real ranges are created only if the probe succeeds;
                // their indices are patched into the preg's map then.
                self.ctx.bundles[probe].ranges.push(LiveRangeListEntry {
                    range: piece,
                    index: LiveRangeIndex::invalid(),
                });
            }

            self.ctx.output.stats.spill_bundle_loop_region_probes += 1;
            // Every preferred register is clobbered at a call, so a region
            // with one can only go in a non-preferred register.
            let call_points = &self.ctx.scratch_call_points[class as usize];
            let skip = if pieces
                .iter()
                .any(|&(_, piece)| any_point_in(call_points, piece))
            {
                self.env.preferred_regs_by_class[class as usize]
            } else {
                PRegSet::empty()
            };
            let mut allocated = None;
            for preg in RegTraversalIter::new(self.env, class, None, hint, bundle.index(), limit) {
                if skip.contains(preg) {
                    continue;
                }
                let preg_idx = PRegIndex::new(preg.index());
                if let AllocRegResult::Allocated(_) =
                    self.try_to_allocate_bundle_to_reg(probe, preg_idx, Some(0), conflicts)
                {
                    allocated = Some(preg_idx);
                    break;
                }
            }
            let Some(preg_idx) = allocated else {
                continue;
            };
            // A value not already in the spill slot on the way into the loop,
            // as one defined in it, must be stored at each exit where it is
            // live instead of once at its def. Keep it on the stack if that is
            // more than one exit, so as not to add stores. This is judged only
            // for the few regions that fit, and what remains still fits.
            let mut verdicts: SmallVec<[(VRegIndex, bool); 4]> = SmallVec::new();
            for &(lr, _) in &pieces {
                let vreg = self.ctx.ranges[lr].vreg;
                if !verdicts.iter().any(|v| v.0 == vreg) {
                    verdicts.push((vreg, true));
                }
            }
            for i in 0..verdicts.len() {
                let vreg = verdicts[i].0;
                let live_in = pieces.iter().any(|&(lr, piece)| {
                    piece.from == region.from && self.ctx.ranges[lr].vreg == vreg
                });
                verdicts[i].1 = (live_in
                    && self.in_spill_bundle_on_entry(bundle, vreg, start, end))
                    || self.loop_exits_holding(bundle, vreg, start, end, &verdicts) <= 1;
            }
            if verdicts.iter().any(|v| !v.1) {
                let preg_allocs = &mut self.ctx.pregs[preg_idx.index()].allocations.btree;
                let ranges = &self.ctx.ranges;
                pieces.retain(|&mut (lr, piece)| {
                    let vreg = ranges[lr].vreg;
                    let keep = verdicts.iter().find(|v| v.0 == vreg).unwrap().1;
                    if !keep {
                        preg_allocs.remove(&LiveRangeKey::from_range(&piece));
                    }
                    keep
                });
                self.ctx.bundles[probe].ranges.clear();
                if pieces.is_empty() {
                    self.ctx.bundles[probe].allocation = Allocation::none();
                    continue;
                }
                for &(_, piece) in &pieces {
                    self.ctx.bundles[probe].ranges.push(LiveRangeListEntry {
                        range: piece,
                        index: LiveRangeIndex::invalid(),
                    });
                }
            }
            trace!(
                "loop region {:?} of spilled bundle {:?} allocated to {:?} as bundle {:?}",
                region,
                bundle,
                preg_idx,
                probe
            );
            self.ctx.output.stats.spill_bundle_loop_region_success += 1;
            self.split_out_loop_pieces(
                bundle,
                probe,
                preg_idx,
                &pieces,
                (start, end),
                &mut loop_pieces,
            );
            probe = LiveBundleIndex::invalid();
        }

        // A value live into a loop and in the spill slot on every way into
        // it stays there, as the slot is reserved for the whole spillset: none
        // of its pieces in the loop needs a store at the exits. Judge this once
        // all the loops have been split out, as a later one may take the value
        // out of the slot on a way into an earlier one.
        let mut in_slot: SmallVec<[(VRegIndex, u32, u32); 4]> = SmallVec::new();
        for &(lr, start, end, live_in) in &loop_pieces {
            let vreg = self.ctx.ranges[lr].vreg;
            if live_in && self.in_spill_bundle_on_entry(bundle, vreg, start, end) {
                in_slot.push((vreg, start, end));
            }
        }
        if !in_slot.is_empty() {
            for &(lr, start, end, _) in &loop_pieces {
                if in_slot.contains(&(self.ctx.ranges[lr].vreg, start, end)) {
                    self.ctx.ranges[lr].set_flag(LiveRangeFlag::InSpillSlot);
                    self.ctx.any_in_spill_slot = true;
                }
            }
        }

        self.ctx.scratch_loop_regions = regions;
        self.ctx.scratch_loop_uses = uses;
    }

    /// Moves `pieces` out of the ranges of spilled `bundle` into `region_bundle`,
    /// which has just been allocated to `preg`: each range covering a piece is cut
    /// into the part before it, which keeps its index, the piece, and the part
    /// after it, both new ranges.
    fn split_out_loop_pieces(
        &mut self,
        bundle: LiveBundleIndex,
        region_bundle: LiveBundleIndex,
        preg: PRegIndex,
        pieces: &[(LiveRangeIndex, CodeRange)],
        (start, end): (u32, u32),
        loop_pieces: &mut SmallVec<[(LiveRangeIndex, u32, u32, bool); 4]>,
    ) {
        let bump = self.ctx.bump();
        let old_ranges = core::mem::replace(
            &mut self.ctx.bundles[bundle].ranges,
            LiveRangeList::new_in(bump.clone()),
        );
        let mut next_piece = 0;
        for entry in old_ranges {
            let lr = entry.index;
            if next_piece == pieces.len() || pieces[next_piece].0 != lr {
                self.ctx.bundles[bundle].ranges.push(entry);
                continue;
            }
            let piece = pieces[next_piece].1;
            let vreg = self.ctx.ranges[lr].vreg;

            let piece_lr = self.ctx.ranges.add(piece, bump.clone());
            self.ctx.ranges[piece_lr].vreg = vreg;
            self.ctx.ranges[piece_lr].bundle = region_bundle;
            // Whether a value live into the loop is in the spill slot there is
            // judged later, from the loop's predecessors.
            let live_in = piece.from == self.ctx.cfginfo.block_entry[start as usize];
            loop_pieces.push((piece_lr, start, end, live_in));
            self.ctx.bundles[region_bundle].ranges[next_piece].index = piece_lr;
            *self.ctx.pregs[preg.index()]
                .allocations
                .btree
                .get_mut(&LiveRangeKey::from_range(&piece))
                .unwrap() = piece_lr;
            self.ctx.vregs[vreg].ranges.push(LiveRangeListEntry {
                range: piece,
                index: piece_lr,
            });
            next_piece += 1;

            let tail_lr = if piece.to < entry.range.to {
                let tail = CodeRange {
                    from: piece.to,
                    to: entry.range.to,
                };
                let tail_lr = self.ctx.ranges.add(tail, bump.clone());
                self.ctx.ranges[tail_lr].vreg = vreg;
                self.ctx.ranges[tail_lr].bundle = bundle;
                self.ctx.vregs[vreg].ranges.push(LiveRangeListEntry {
                    range: tail,
                    index: tail_lr,
                });
                Some(tail_lr)
            } else {
                None
            };

            let uses =
                core::mem::replace(&mut self.ctx.ranges[lr].uses, UseList::new_in(bump.clone()));
            for u in uses {
                let to = if u.pos < piece.from {
                    lr
                } else if u.pos < piece.to {
                    piece_lr
                } else {
                    tail_lr.unwrap()
                };
                self.ctx.ranges[to].uses.push(u);
            }

            if piece.from > entry.range.from {
                self.ctx.ranges[lr].range.to = piece.from;
                self.ctx.bundles[bundle].ranges.push(LiveRangeListEntry {
                    range: self.ctx.ranges[lr].range,
                    index: lr,
                });
            } else {
                // The piece starts the range, so it takes over its flags and
                // the old range goes away.
                let flags = self.ctx.ranges[lr].flag_word();
                self.ctx.ranges[piece_lr].merge_flags(flags);
                self.ctx.vregs[vreg].ranges.retain(|e| e.index != lr);
            }
            if let Some(tail_lr) = tail_lr {
                self.ctx.bundles[bundle].ranges.push(LiveRangeListEntry {
                    range: self.ctx.ranges[tail_lr].range,
                    index: tail_lr,
                });
            }
        }
        debug_assert_eq!(next_piece, pieces.len());
    }

    /// Whether `vreg` is in a range of spilled `bundle` at the end of each
    /// predecessor, outside the run of blocks `start..=end`, of a block in it,
    /// the first block having at least one such predecessor.
    fn in_spill_bundle_on_entry(
        &self,
        bundle: LiveBundleIndex,
        vreg: VRegIndex,
        start: u32,
        end: u32,
    ) -> bool {
        let ranges = &self.ctx.bundles[bundle].ranges;
        // The loop's first block must have a way in, or the value may be
        // defined there, as at the entry block, and not yet be stored.
        let first = Block::new(start as usize);
        if self
            .func
            .block_preds(first)
            .iter()
            .all(|pred| pred.raw_u32() >= start && pred.raw_u32() <= end)
        {
            return false;
        }
        (start..=end).all(|block| {
            self.func
                .block_preds(Block::new(block as usize))
                .iter()
                .filter(|pred| pred.raw_u32() < start || pred.raw_u32() > end)
                .all(|pred| {
                    let exit = self.ctx.cfginfo.block_exit[pred.index()];
                    let i = ranges.partition_point(|entry| entry.range.to <= exit);
                    i < ranges.len()
                        && ranges[i].range.contains_point(exit)
                        && self.ctx.ranges[ranges[i].index].vreg == vreg
                })
        })
    }

    /// The number, up to 2, of edges on which `vreg`, in a range of `bundle`
    /// at the end of a block in the run `start..=end`, is moved out of the
    /// loop's registers: those leaving the run, and those in it passing the
    /// value to a block parameter none of whose pieces, `in_loop`, is in them.
    fn loop_exits_holding(
        &self,
        bundle: LiveBundleIndex,
        vreg: VRegIndex,
        start: u32,
        end: u32,
        in_loop: &[(VRegIndex, bool)],
    ) -> usize {
        let ranges = &self.ctx.bundles[bundle].ranges;
        let mut exits = 0;
        for block in start..=end {
            let exit = self.ctx.cfginfo.block_exit[block as usize];
            let i = ranges.partition_point(|entry| entry.range.to <= exit);
            if i == ranges.len()
                || !ranges[i].range.contains_point(exit)
                || self.ctx.ranges[ranges[i].index].vreg != vreg
            {
                continue;
            }
            for succ in self.func.block_succs(Block::new(block as usize)) {
                if succ.raw_u32() < start || succ.raw_u32() > end {
                    exits += 1;
                    if exits > 1 {
                        return exits;
                    }
                }
            }
        }
        let outs = &self.ctx.blockparam_outs;
        let first = outs.partition_point(|out| out.from_vreg < vreg);
        for out in &outs[first..] {
            if out.from_vreg != vreg {
                break;
            }
            let (from, to) = (out.from_block.raw_u32(), out.to_block.raw_u32());
            if from < start
                || from > end
                || to < start
                || to > end
                || in_loop.iter().any(|v| v.0 == out.to_vreg)
            {
                continue;
            }
            exits += 1;
            if exits > 1 {
                break;
            }
        }
        exits
    }

    /// Prepares the cache of `loop_run`.
    fn compute_loop_runs(&mut self) {
        let num_blocks = self.ctx.cfginfo.approx_loop_depth.len();
        let runs = &mut self.ctx.scratch_loop_runs;
        runs.clear();
        runs.resize(num_blocks, (u32::MAX, u32::MAX));
    }

    /// Collects the points at which every allocatable register of `class` is
    /// clobbered, and those at which every preferred register is. Either is a
    /// fixed reservation on the first preferred register, so it is enough to
    /// look at the fixed reservations of that one.
    fn compute_clobber_points(&mut self, class: RegClass) {
        let class = class as usize;
        let mut preferred = self.env.preferred_regs_by_class[class];
        let allocatable = preferred | self.env.non_preferred_regs_by_class[class];
        if preferred == PRegSet::empty() {
            preferred = allocatable;
        }
        let full = &mut self.ctx.scratch_full_clobbers[class];
        let calls = &mut self.ctx.scratch_call_points[class];
        full.clear();
        calls.clear();
        let Some(reg) = preferred.into_iter().chain(allocatable).next() else {
            return;
        };
        for (key, lr) in &self.ctx.pregs[reg.index()].allocations.btree {
            if lr.is_valid() {
                continue;
            }
            let point = ProgPoint::after(ProgPoint::from_index(key.from).inst());
            if calls.last() == Some(&point) || full.last() == Some(&point) {
                continue;
            }
            let clobbers = self.func.inst_clobbers(point.inst());
            if clobbers & preferred == preferred {
                calls.push(point);
            }
            if clobbers & allocatable == allocatable {
                full.push(point);
            }
        }
    }

    pub fn spillslot_can_fit_spillset(
        &mut self,
        spillslot: SpillSlotIndex,
        spillset: SpillSetIndex,
    ) -> bool {
        !self.ctx.spillslots[spillslot.index()]
            .ranges
            .btree
            .contains_key(&LiveRangeKey::from_range(
                &self.ctx.spillsets[spillset].range,
            ))
    }

    pub fn allocate_spillset_to_spillslot(
        &mut self,
        spillset: SpillSetIndex,
        spillslot: SpillSlotIndex,
    ) {
        self.ctx.spillsets[spillset].slot = spillslot;

        let res = self.ctx.spillslots[spillslot.index()].ranges.btree.insert(
            LiveRangeKey::from_range(&self.ctx.spillsets[spillset].range),
            spillset,
        );

        debug_assert!(res.is_none());
    }

    pub fn allocate_spillslots(&mut self) {
        const MAX_ATTEMPTS: usize = 10;

        for spillset in 0..self.ctx.spillsets.len() {
            trace!("allocate spillslot: {}", spillset);
            let spillset = SpillSetIndex::new(spillset);
            if !self.ctx.spillsets[spillset].required {
                continue;
            }
            let class = self.ctx.spillsets[spillset].class as usize;
            // Try a few existing spillslots.
            let mut i = self.ctx.slots_by_class[class].probe_start;
            let mut success = false;
            // Never probe the same element more than once: limit the
            // attempt count to the number of slots in existence.
            for _attempt in
                0..core::cmp::min(self.ctx.slots_by_class[class].slots.len(), MAX_ATTEMPTS)
            {
                // Note: this indexing of `slots` is always valid
                // because either the `slots` list is empty and the
                // iteration limit above consequently means we don't
                // run this loop at all, or else `probe_start` is
                // in-bounds (because it is made so below when we add
                // a slot, and it always takes on the last index `i`
                // after this loop).
                let spillslot = self.ctx.slots_by_class[class].slots[i];

                if self.spillslot_can_fit_spillset(spillslot, spillset) {
                    self.allocate_spillset_to_spillslot(spillset, spillslot);
                    success = true;
                    self.ctx.slots_by_class[class].probe_start = i;
                    break;
                }

                i = self.ctx.slots_by_class[class].next_index(i);
            }

            if !success {
                // Allocate a new spillslot.
                let spillslot = SpillSlotIndex::new(self.ctx.spillslots.len());
                self.ctx.spillslots.push(SpillSlotData {
                    ranges: self.ctx.scratch_spillset_pool.pop().unwrap_or_default(),
                    alloc: Allocation::none(),
                    slots: self.func.spillslot_size(self.ctx.spillsets[spillset].class) as u32,
                });
                self.ctx.slots_by_class[class].slots.push(spillslot);
                self.ctx.slots_by_class[class].probe_start =
                    self.ctx.slots_by_class[class].slots.len() - 1;

                self.allocate_spillset_to_spillslot(spillset, spillslot);
            }
        }

        // Assign actual slot indices to spillslots.
        for i in 0..self.ctx.spillslots.len() {
            self.ctx.spillslots[i].alloc = self.allocate_spillslot(self.ctx.spillslots[i].slots);
        }

        trace!("spillslot allocator done");
    }

    pub fn allocate_spillslot(&mut self, size: u32) -> Allocation {
        let mut offset = self.ctx.output.num_spillslots as u32;
        // Align up to `size`.
        debug_assert!(size.is_power_of_two());
        offset = (offset + size - 1) & !(size - 1);
        let slot = if self.func.multi_spillslot_named_by_last_slot() {
            offset + size - 1
        } else {
            offset
        };
        offset += size;
        self.ctx.output.num_spillslots = offset as _;
        Allocation::stack(SpillSlot::new(slot as usize))
    }
}

/// Whether one of the sorted `points` lies in `range`.
fn any_point_in(points: &[ProgPoint], range: CodeRange) -> bool {
    let i = points.partition_point(|&p| p < range.from);
    i < points.len() && points[i] < range.to
}

/// Whether `vreg` has a use in `range`, given all uses sorted by vreg and
/// position.
fn any_use_in(uses: &[(VRegIndex, ProgPoint)], vreg: VRegIndex, range: CodeRange) -> bool {
    let i = uses.partition_point(|&u| u < (vreg, range.from));
    i < uses.len() && uses[i].0 == vreg && uses[i].1 < range.to
}

/// The first and last block of the innermost loop containing `block`, that
/// is the maximal run of consecutive blocks around it at least as deep.
/// Cached in `runs` for every block of that depth in the run, which is
/// their innermost loop too.
fn loop_run(depth: &[u32], runs: &mut [(u32, u32)], block: u32) -> (u32, u32) {
    let b = block as usize;
    if runs[b].0 != u32::MAX {
        return runs[b];
    }
    let d = depth[b];
    let start = depth[..b].iter().rposition(|&x| x < d).map_or(0, |i| i + 1);
    let end = depth[b..]
        .iter()
        .position(|&x| x < d)
        .map_or(depth.len(), |i| b + i)
        - 1;
    let run = (start as u32, end as u32);
    for i in start..=end {
        if depth[i] == d {
            runs[i] = run;
        }
    }
    run
}
