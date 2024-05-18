use crate::{find_nth, MachineEnv, PReg, RegClass};
/// This iterator represents a traversal through all allocatable
/// registers of a given class, in a certain order designed to
/// minimize allocation contention.
///
/// The order in which we try registers is somewhat complex:
/// - First, if there is a hint, we try that.
/// - Then, we try registers in a traversal order that is based on an
///   "offset" (usually the bundle index) spreading pressure evenly
///   among registers to reduce commitment-map contention.
/// - Within that scan, we try registers in two groups: first,
///   prferred registers; then, non-preferred registers. (In normal
///   usage, these consist of caller-save and callee-save registers
///   respectively, to minimize clobber-saves; but they need not.)

pub struct RegTraversalIter {
    pref_regs_first: u64,
    pref_regs_second: u64,
    non_pref_regs_first: u64,
    non_pref_regs_second: u64,
    hint_regs: u64,
    is_fixed: bool,
    fixed: Option<PReg>,
    class_mask: u8,
}

impl RegTraversalIter {
    pub fn new(
        env: &MachineEnv,
        class: RegClass,
        hint_reg: PReg,
        hint2_reg: PReg,
        offset: usize,
        fixed: Option<PReg>,
    ) -> Self {
        // get a mask for the hint registers
        let mut hint_mask = 0u64;

        if hint_reg != PReg::invalid() {
            let mask = 1u64 << (hint_reg.bits & 0b0011_1111);
            hint_mask |= mask;
        }

        if hint2_reg != PReg::invalid() {
            let mask = 1u64 << (hint2_reg.bits & 0b0011_1111);
            hint_mask |= mask;
        }

        let class = class as u8 as usize;

        let pref_regs_by_class = env.preferred_regs_by_class.bits[class];
        let non_pref_regs_by_class = env.non_preferred_regs_by_class.bits[class];

        let n_pref_regs = pref_regs_by_class.count_ones() as usize;
        let n_non_pref_regs = non_pref_regs_by_class.count_ones() as usize;

        let offset_pref = if n_pref_regs > 0 {
            offset % n_pref_regs
        } else {
            0
        };
        let offset_non_pref = if n_non_pref_regs > 0 {
            offset % n_non_pref_regs
        } else {
            0
        };

        // we want to split the pref registers bit vectors into two sets
        // with the offset lowest bits in one and the rest in the other
        let split_num = (n_pref_regs - offset_pref) as u64;
        let split_pos = find_nth(pref_regs_by_class, split_num);
        let mask = (1 << split_pos) - 1;
        let pref_regs_first = pref_regs_by_class & !mask;
        let pref_regs_second = pref_regs_by_class & mask;

        let split_num = (n_non_pref_regs - offset_non_pref) as u64;
        let split_pos = find_nth(non_pref_regs_by_class, split_num);
        let mask = (1 << split_pos) - 1;
        let non_pref_regs_first = non_pref_regs_by_class & !mask;
        let non_pref_regs_second = non_pref_regs_by_class & mask;

        // remove the hint registers from the bit vectors
        let pref_regs_first = pref_regs_first & !hint_mask;
        let pref_regs_second = pref_regs_second & !hint_mask;
        let non_pref_regs_first = non_pref_regs_first & !hint_mask;
        let non_pref_regs_second = non_pref_regs_second & !hint_mask;

        let class_mask = (class as u8) << 6;

        Self {
            pref_regs_first,
            pref_regs_second,
            non_pref_regs_first,
            non_pref_regs_second,
            hint_regs: hint_mask,
            is_fixed: fixed.is_some(),
            fixed,
            class_mask,
        }
    }
}

impl core::iter::Iterator for RegTraversalIter {
    type Item = PReg;

    fn next(&mut self) -> Option<PReg> {
        // only take the fixed register if it exists
        if self.is_fixed {
            let ret = self.fixed;
            self.fixed = None;
            return ret;
        }

        // if there are hints, return them first
        if self.hint_regs != 0 {
            let index = self.hint_regs.trailing_zeros() as u8;
            self.hint_regs &= !(1u64 << index);
            let reg_index = index as u8 | self.class_mask;
            return Some(PReg::from(reg_index));
        }

        // iterate over the preferred register rotated by offset
        // iterate over first half
        if self.pref_regs_first != 0 {
            let index = self.pref_regs_first.trailing_zeros() as u8;
            self.pref_regs_first &= !(1u64 << index);
            let reg_index = index as u8 | self.class_mask;
            return Some(PReg::from(reg_index));
        }
        // iterate over second half
        if self.pref_regs_second != 0 {
            let index = self.pref_regs_second.trailing_zeros() as u8;
            self.pref_regs_second &= !(1u64 << index);
            let reg_index = index as u8 | self.class_mask;
            return Some(PReg::from(reg_index));
        }

        // iterate over the nonpreferred register rotated by offset
        // iterate over first half
        if self.non_pref_regs_first != 0 {
            let index = self.non_pref_regs_first.trailing_zeros() as u8;
            self.non_pref_regs_first &= !(1u64 << index);
            let reg_index = index as u8 | self.class_mask;
            return Some(PReg::from(reg_index));
        }
        // iterate over second half
        if self.non_pref_regs_second != 0 {
            let index = self.non_pref_regs_second.trailing_zeros() as u8;
            self.non_pref_regs_second &= !(1u64 << index);
            let reg_index = index as u8 | self.class_mask;
            return Some(PReg::from(reg_index));
        }
        None
    }
}

/// Wrapping function to wrap around the index for an iterator
fn wrap(idx: usize, limit: usize) -> usize {
    if idx >= limit {
        idx - limit
    } else {
        idx
    }
}
