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
    AllocRegResult, Env, LiveRangeKey, LiveRangeSet, PReg, PRegIndex, RegClass, RegTraversalIter,
    SpillSetIndex, SpillSlotData, SpillSlotIndex, SpillSlotList,
};
use crate::{Allocation, Function, SpillSlot};

impl<'a, F: Function> Env<'a, F> {
    pub fn try_allocating_regs_for_spilled_bundles(&mut self) {
        trace!("allocating regs for spilled bundles");
        for i in 0..self.spilled_bundles.len() {
            let bundle = self.spilled_bundles[i]; // don't borrow self

            let class = self.spillsets[self.bundles[bundle.index()].spillset.index()].class;
            let hint = self.spillsets[self.bundles[bundle.index()].spillset.index()].reg_hint;

            trace!("bundle {:?} with class {:?} hint {:?}", bundle, class, hint);

            // This may be an empty-range bundle whose ranges are not
            // sorted; sort all range-lists again here.
            self.bundles[bundle.index()]
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
                trace!(
                    "spilling bundle {:?}: marking spillset {:?} as required",
                    bundle,
                    self.bundles[bundle.index()].spillset
                );
                self.spillsets[self.bundles[bundle.index()].spillset.index()].required = true;
            }
        }
    }

    pub fn spillslot_can_fit_spillset(
        &mut self,
        spillslot: SpillSlotIndex,
        spillset: SpillSetIndex,
    ) -> bool {
        for &vreg in &self.spillsets[spillset.index()].vregs {
            for entry in &self.vregs[vreg.index()].ranges {
                if self.spillslots[spillslot.index()]
                    .ranges
                    .btree
                    .contains_key(&LiveRangeKey::from_range(&entry.range))
                {
                    return false;
                }
            }
        }
        true
    }

    pub fn allocate_spillset_to_spillslot(
        &mut self,
        spillset: SpillSetIndex,
        spillslot: SpillSlotIndex,
    ) {
        self.spillsets[spillset.index()].slot = spillslot;
        for i in 0..self.spillsets[spillset.index()].vregs.len() {
            // don't borrow self
            let vreg = self.spillsets[spillset.index()].vregs[i];
            trace!(
                "spillslot {:?} alloc'ed to spillset {:?}: vreg {:?}",
                spillslot,
                spillset,
                vreg,
            );
            for entry in &self.vregs[vreg.index()].ranges {
                trace!(
                    "spillslot {:?} getting range {:?} from LR {:?} from vreg {:?}",
                    spillslot,
                    entry.range,
                    entry.index,
                    vreg,
                );
                self.spillslots[spillslot.index()]
                    .ranges
                    .btree
                    .insert(LiveRangeKey::from_range(&entry.range), entry.index);
            }
        }
    }

    pub fn allocate_spillslots(&mut self) {
        for spillset in 0..self.spillsets.len() {
            trace!("allocate spillslot: {}", spillset);
            let spillset = SpillSetIndex::new(spillset);
            if !self.spillsets[spillset.index()].required {
                continue;
            }
            // Get or create the spillslot list for this size.
            let size = self.spillsets[spillset.index()].size as usize;
            if size >= self.slots_by_size.len() {
                self.slots_by_size.resize(
                    size + 1,
                    SpillSlotList {
                        first_spillslot: SpillSlotIndex::invalid(),
                        last_spillslot: SpillSlotIndex::invalid(),
                    },
                );
            }
            // Try a few existing spillslots.
            let mut spillslot_iter = self.slots_by_size[size].first_spillslot;
            let mut first_slot = SpillSlotIndex::invalid();
            let mut prev = SpillSlotIndex::invalid();
            let mut success = false;
            for _attempt in 0..10 {
                if spillslot_iter.is_invalid() {
                    break;
                }
                if spillslot_iter == first_slot {
                    // We've started looking at slots we placed at the end; end search.
                    break;
                }
                if first_slot.is_invalid() {
                    first_slot = spillslot_iter;
                }

                if self.spillslot_can_fit_spillset(spillslot_iter, spillset) {
                    self.allocate_spillset_to_spillslot(spillset, spillslot_iter);
                    success = true;
                    break;
                }
                // Remove the slot and place it at the end of the respective list.
                let next = self.spillslots[spillslot_iter.index()].next_spillslot;
                if prev.is_valid() {
                    self.spillslots[prev.index()].next_spillslot = next;
                } else {
                    self.slots_by_size[size].first_spillslot = next;
                }
                if !next.is_valid() {
                    self.slots_by_size[size].last_spillslot = prev;
                }

                let last = self.slots_by_size[size].last_spillslot;
                if last.is_valid() {
                    self.spillslots[last.index()].next_spillslot = spillslot_iter;
                } else {
                    self.slots_by_size[size].first_spillslot = spillslot_iter;
                }
                self.slots_by_size[size].last_spillslot = spillslot_iter;

                prev = spillslot_iter;
                spillslot_iter = next;
            }

            if !success {
                // Allocate a new spillslot.
                let spillslot = SpillSlotIndex::new(self.spillslots.len());
                let next = self.slots_by_size[size].first_spillslot;
                self.spillslots.push(SpillSlotData {
                    ranges: LiveRangeSet::new(),
                    next_spillslot: next,
                    alloc: Allocation::none(),
                    class: self.spillsets[spillset.index()].class,
                });
                self.slots_by_size[size].first_spillslot = spillslot;
                if !next.is_valid() {
                    self.slots_by_size[size].last_spillslot = spillslot;
                }

                self.allocate_spillset_to_spillslot(spillset, spillslot);
            }
        }

        // Assign actual slot indices to spillslots.
        for i in 0..self.spillslots.len() {
            self.spillslots[i].alloc = self.allocate_spillslot(self.spillslots[i].class);
        }

        trace!("spillslot allocator done");
    }

    pub fn allocate_spillslot(&mut self, class: RegClass) -> Allocation {
        let size = self.func.spillslot_size(class) as u32;
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
        Allocation::stack(SpillSlot::new(slot as usize, class))
    }
}
