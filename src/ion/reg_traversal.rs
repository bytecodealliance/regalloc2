use crate::{MachineEnv, PReg, PRegClass, RegClass};
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
    pref_regs_by_class: PRegClass,
    non_pref_regs_by_class: PRegClass,
    hints: [Option<PReg>; 2],
    hint_idx: usize,
    pref_idx: usize,
    non_pref_idx: usize,
    offset_pref: usize,
    offset_non_pref: usize,
    is_fixed: bool,
    fixed: Option<PReg>,
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
        let mut hint_reg = if hint_reg != PReg::invalid() {
            Some(hint_reg)
        } else {
            None
        };
        let mut hint2_reg = if hint2_reg != PReg::invalid() {
            Some(hint2_reg)
        } else {
            None
        };

        if hint_reg.is_none() {
            hint_reg = hint2_reg;
            hint2_reg = None;
        }
        let hints = [hint_reg, hint2_reg];
        let class = class as u8 as usize;

        let pref_regs_by_class = env.preferred_regs_by_class.to_preg_class(class);
        let non_pref_regs_by_class = env.non_preferred_regs_by_class.to_preg_class(class);

        let offset_pref = if pref_regs_by_class.len() > 0 {
            offset % pref_regs_by_class.len()
        } else {
            0
        };
        let offset_non_pref = if non_pref_regs_by_class.len() > 0 {
            offset % non_pref_regs_by_class.len()
        } else {
            0
        };
        Self {
            pref_regs_by_class,
            non_pref_regs_by_class,
            hints,
            hint_idx: 0,
            pref_idx: 0,
            non_pref_idx: 0,
            offset_pref,
            offset_non_pref,
            is_fixed: fixed.is_some(),
            fixed,
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
        if self.hint_idx < 2 && self.hints[self.hint_idx].is_some() {
            let h = self.hints[self.hint_idx];
            self.hint_idx += 1;
            return h;
        }

        // iterate over the preferred register rotated by offset
        // ignoring hint register
        let n_pref_regs = self.pref_regs_by_class.len();
        while self.pref_idx < n_pref_regs {
            let mut arr = self.pref_regs_by_class.into_iter();
            let r = arr.nth(wrap(self.pref_idx + self.offset_pref, n_pref_regs));
            self.pref_idx += 1;
            if r == self.hints[0] || r == self.hints[1] {
                continue;
            }
            return r;
        }

        // iterate over the nonpreferred register rotated by offset
        // ignoring hint register
        let n_non_pref_regs = self.non_pref_regs_by_class.len();
        while self.non_pref_idx < n_non_pref_regs {
            let mut arr = self.non_pref_regs_by_class.into_iter();
            let r = arr.nth(wrap(
                self.non_pref_idx + self.offset_non_pref,
                n_non_pref_regs,
            ));
            self.non_pref_idx += 1;
            if r == self.hints[0] || r == self.hints[1] {
                continue;
            }
            return r;
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
