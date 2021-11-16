/*
 * The following license applies to this file, which was initially
 * derived from the files `js/src/jit/BacktrackingAllocator.h` and
 * `js/src/jit/BacktrackingAllocator.cpp` in Mozilla Firefox:
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Since the initial port, the design has been substantially evolved
 * and optimized.
 */

//! Stackmap computation.

use super::{Env, ProgPoint, VRegIndex};
use crate::Function;

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

        log::trace!("safepoints_per_vreg = {:?}", self.safepoints_per_vreg);

        for vreg in self.func.reftype_vregs() {
            log::trace!("generating safepoint info for vreg {}", vreg);
            let vreg = VRegIndex::new(vreg.vreg());
            let mut safepoints: Vec<ProgPoint> = self
                .safepoints_per_vreg
                .get(&vreg.index())
                .unwrap()
                .iter()
                .map(|&inst| ProgPoint::before(inst))
                .collect();
            safepoints.sort_unstable();
            log::trace!(" -> live over safepoints: {:?}", safepoints);

            let mut safepoint_idx = 0;
            for entry in &self.vregs[vreg.index()].ranges {
                let range = entry.range;
                let alloc = self.get_alloc_for_range(entry.index);
                log::trace!(" -> range {:?}: alloc {}", range, alloc);
                while safepoint_idx < safepoints.len() && safepoints[safepoint_idx] < range.to {
                    if safepoints[safepoint_idx] < range.from {
                        safepoint_idx += 1;
                        continue;
                    }
                    log::trace!("    -> covers safepoint {:?}", safepoints[safepoint_idx]);

                    self.safepoint_slots
                        .push((safepoints[safepoint_idx], alloc));
                    safepoint_idx += 1;
                }
            }
        }

        self.safepoint_slots.sort_unstable();
        log::trace!("final safepoint slots info: {:?}", self.safepoint_slots);
    }
}
