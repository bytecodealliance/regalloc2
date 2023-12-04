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

//! Stackmap computation.

use super::{Env, ProgPoint, VRegIndex};
use crate::{ion::data_structures::u64_key, Function};

impl<'a, F: Function> Env<'a, F> {
    pub fn compute_stackmaps(&mut self) {
        // For each ref-typed vreg, iterate through ranges and find
        // safepoints in-range. Add the SpillSlot to the stackmap.

        if self.func.reftype_vregs().is_empty() {
            return;
        }

        // Given `safepoints_per_vreg` from the liveness computation,
        // all we have to do is, for each vreg in this map, step
        // through the LiveRanges along with a sorted list of
        // safepoints; and for each safepoint in the current range,
        // emit the allocation into the `safepoint_slots` list.

        trace!("safepoints_per_vreg = {:?}", self.safepoints_per_vreg);

        for vreg in self.func.reftype_vregs() {
            trace!("generating safepoint info for vreg {}", vreg);
            let vreg = VRegIndex::new(vreg.vreg());
            let spill_alloc = self.get_spill_alloc_for(vreg);

            // If this vreg didn't spill, it'll never live on the stack.
            if spill_alloc.is_none() {
                continue;
            }

            let range = self.vregs[vreg].range.expect("vreg missing a lifetime");
            trace!(" -> range {:?}: alloc {}", range, spill_alloc);

            for &inst in self.safepoints_per_vreg.get(&vreg.index()).unwrap() {
                let safepoint = ProgPoint::before(inst);
                if range.contains_point(safepoint) {
                    trace!("  -> covers safepoint {:?}", safepoint);
                    self.safepoint_slots.push((safepoint, spill_alloc));
                }
            }
        }

        self.safepoint_slots
            .sort_unstable_by_key(|(progpoint, slot)| u64_key(progpoint.to_index(), slot.bits()));
        trace!("final safepoint slots info: {:?}", self.safepoint_slots);
    }
}
