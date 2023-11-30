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
    data_structures::VRegIndex, AllocRegResult, Env, LiveRangeKey, PReg, PRegIndex,
    RegTraversalIter, SpillSlotData, SpillSlotIndex,
};
use crate::{ion::data_structures::SpillSetRanges, Allocation, Function, SpillSlot};

impl<'a, F: Function> Env<'a, F> {
    pub fn try_allocating_regs_for_spilled_bundles(&mut self) {
        trace!("allocating regs for spilled bundles");

        let mut bundles = core::mem::take(&mut self.spilled_bundles);

        bundles.retain(|&bundle| {
            let spillset = self.bundles[bundle].spillset;
            if self.spillsets[spillset].required {
                trace!("spill required, not attempting to find a preg");
                return true;
            }

            if self.bundles[bundle].ranges.is_empty() {
                return false;
            }

            let class = self.spillsets[self.bundles[bundle].spillset].class;
            let hint = self.spillsets[self.bundles[bundle].spillset].reg_hint;

            // This may be an empty-range bundle whose ranges are not
            // sorted; sort all range-lists again here.
            self.bundles[bundle]
                .ranges
                .sort_unstable_by_key(|entry| entry.range.from);

            let mut success = false;
            self.stats.spill_bundle_reg_probes += 1;
            for preg in
                RegTraversalIter::new(self.env, class, hint, PReg::invalid(), bundle.index(), None)
            {
                trace!("trying bundle {:?} to preg {:?}", bundle, preg);
                let preg_idx = PRegIndex::new(preg.index());
                if let AllocRegResult::Allocated(_) =
                    self.try_to_allocate_bundle_to_reg(bundle, preg_idx, None)
                {
                    self.stats.spill_bundle_reg_success += 1;
                    success = true;
                    break;
                }
            }
            if !success {
                trace!("spilling bundle {bundle:?}: marking spillset {spillset:?} as required");
                self.spillsets[spillset].required = true;
            }

            !success
        });

        self.spilled_bundles = bundles;
    }

    pub fn spillslot_can_fit_vreg(&mut self, spillslot: SpillSlotIndex, vreg: VRegIndex) -> bool {
        let range = self.vregs[vreg]
            .range
            .as_ref()
            .expect("spilling a vreg with no uses");
        !self.spillslots[spillslot.index()]
            .ranges
            .btree
            .contains_key(&LiveRangeKey::from_range(range))
    }

    pub fn allocate_vreg_to_spillslot(&mut self, vreg: VRegIndex, spillslot: SpillSlotIndex) {
        self.vregs[vreg].slot = spillslot;

        let range = self.vregs[vreg]
            .range
            .as_ref()
            .expect("spilling a vreg with no uses");

        let res = self.spillslots[spillslot.index()]
            .ranges
            .btree
            .insert(LiveRangeKey::from_range(range), vreg);

        debug_assert!(res.is_none());
    }

    pub fn allocate_spillslots(&mut self) {
        const MAX_ATTEMPTS: usize = 10;

        for bundle in core::mem::take(&mut self.spilled_bundles) {
            trace!("spill bundle: {bundle:?}");

            let spillset = self.bundles[bundle].spillset;
            if !self.spillsets[spillset].required {
                continue;
            }

            let spillset = self.bundles[bundle].spillset;
            let class = self.spillsets[spillset].class as usize;

            let mut i = self.slots_by_class[class].probe_start;

            for range in 0..self.bundles[bundle].ranges.len() {
                let range = self.bundles[bundle].ranges[range];
                let vreg = self.ranges[range.index].vreg;

                assert!(vreg.is_valid(), "valid vreg");

                trace!(" -> spilling vreg {vreg:?}");

                // If this vreg has spilled already, skip it.
                if self.vregs[vreg].slot.is_valid() {
                    trace!("  -> already spilled; skipping");
                    // TODO: perhaps we could reverse the allocation to the slots_by_class index,
                    // and set `i` to that? This would heuristically push us towards spill bundles
                    // using the same spillslot when vregs are spilled from multiple places.
                    continue;
                }

                // Try a few existing spillslots.
                let mut success = false;

                // Never probe the same element more than once: limit the
                // attempt count to the number of slots in existence.
                for _attempt in
                    0..core::cmp::min(self.slots_by_class[class].slots.len(), MAX_ATTEMPTS)
                {
                    // Note: this indexing of `slots` is always valid
                    // because either the `slots` list is empty and the
                    // iteration limit above consequently means we don't
                    // run this loop at all, or else `probe_start` is
                    // in-bounds (because it is made so below when we add
                    // a slot, and it always takes on the last index `i`
                    // after this loop).
                    let spillslot = self.slots_by_class[class].slots[i];

                    if self.spillslot_can_fit_vreg(spillslot, vreg) {
                        self.allocate_vreg_to_spillslot(vreg, spillslot);
                        success = true;
                        break;
                    }

                    i = self.slots_by_class[class].next_index(i);
                }

                if !success {
                    // Allocate a new spillslot.
                    let spillslot = SpillSlotIndex::new(self.spillslots.len());
                    self.spillslots.push(SpillSlotData {
                        ranges: SpillSetRanges::new(),
                        alloc: Allocation::none(),
                        slots: self.func.spillslot_size(self.spillsets[spillset].class) as u32,
                    });
                    i = self.slots_by_class[class].slots.len();
                    self.slots_by_class[class].slots.push(spillslot);
                    self.slots_by_class[class].probe_start = i;

                    self.allocate_vreg_to_spillslot(vreg, spillslot);
                }
            }

            self.slots_by_class[class].probe_start = i;
        }

        // Assign actual slot indices to spillslots.
        for i in 0..self.spillslots.len() {
            self.spillslots[i].alloc = self.allocate_spillslot(self.spillslots[i].slots);
        }

        trace!("spillslot allocator done");
    }

    pub fn allocate_spillslot(&mut self, size: u32) -> Allocation {
        let mut offset = self.num_spillslots;
        // Align up to `size`.
        debug_assert!(size.is_power_of_two());
        offset = (offset + size - 1) & !(size - 1);
        let slot = if self.func.multi_spillslot_named_by_last_slot() {
            offset + size - 1
        } else {
            offset
        };
        offset += size;
        self.num_spillslots = offset;
        Allocation::stack(SpillSlot::new(slot as usize))
    }
}
