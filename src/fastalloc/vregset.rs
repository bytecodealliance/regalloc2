use core::fmt;

use alloc::vec;
use alloc::vec::Vec;
use crate::{RegClass, VReg};

#[derive(Clone)]
struct VRegNode {
    next: u32,
    prev: u32,
    class: RegClass,
}

// Using a non-circular doubly linked list here for fast insertion,
// removal and iteration.
pub struct VRegSet {
    items: Vec<VRegNode>,
    head: u32,
}

impl VRegSet {
    pub fn with_capacity(num_vregs: usize) -> Self {
        Self {
            items: vec![VRegNode { prev: u32::MAX, next: u32::MAX, class: RegClass::Int }; num_vregs],
            head: u32::MAX,
        }
    }

    pub fn insert(&mut self, vreg: VReg) {
        // Intentionally assuming that the set doesn't already
        // contain `vreg`.
        if self.head == u32::MAX {
            self.items[vreg.vreg()] = VRegNode {
                next: u32::MAX,
                prev: u32::MAX,
                class: vreg.class(),
            };
            self.head = vreg.vreg() as u32;
        } else {
            let old_head_next = self.items[self.head as usize].next;
            if old_head_next != u32::MAX {
                self.items[old_head_next as usize].prev = vreg.vreg() as u32;
            }
            self.items[self.head as usize].next = vreg.vreg() as u32;
            self.items[vreg.vreg()] = VRegNode {
                next: old_head_next,
                prev: self.head,
                class: vreg.class(),
            };
        }
    }

    pub fn remove(&mut self, vreg_num: usize) {
        let prev = self.items[vreg_num].prev;
        let next = self.items[vreg_num].next;
        if prev != u32::MAX {
            self.items[prev as usize].next = next;
        }
        if next != u32::MAX {
            self.items[next as usize].prev = prev;
        }
        if vreg_num as u32 == self.head {
            self.head = next;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.head == u32::MAX
    }

    pub fn iter(&self) -> VRegSetIter {
        VRegSetIter {
            curr_item: self.head,
            head: self.head,
            items: &self.items,
        }
    }
}

pub struct VRegSetIter<'a> {
    curr_item: u32,
    head: u32,
    items: &'a [VRegNode],
}

impl<'a> Iterator for VRegSetIter<'a> {
    type Item = VReg;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_item != u32::MAX {
            let item = self.items[self.curr_item as usize].clone();
            let vreg = VReg::new(self.curr_item as usize, item.class);
            self.curr_item = item.next;
            Some(vreg)
        } else {
            None
        }
    }
}

impl fmt::Debug for VRegSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ ")?;
        for vreg in self.iter() {
            write!(f, "{vreg} ")?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use RegClass::*;
    const VREG: fn(usize, RegClass) -> VReg = VReg::new;

    #[test]
    fn operations() {
        let mut set = VRegSet::with_capacity(3090);
        set.insert(VREG(10, Int));
        set.insert(VREG(2000, Int));
        set.insert(VREG(11, Vector));
        set.insert(VREG(199, Float));
        set.insert(VREG(23, Int));
        let els = [
            VREG(10, Int),
            VREG(23, Int),
            VREG(199, Float),
            VREG(11, Vector),
            VREG(2000, Int),
        ];
        for (actual_el, expected_el) in set.iter().zip(els.iter()) {
            assert_eq!(actual_el, *expected_el);
        }
        set.remove(23);
        set.insert(VREG(73, Vector));
        let els = [
            VREG(10, Int),
            VREG(73, Vector),
            VREG(199, Float),
            VREG(11, Vector),
            VREG(2000, Int),
        ];
        for (actual_el, expected_el) in set.iter().zip(els.iter()) {
            assert_eq!(actual_el, *expected_el);
        }
    }

    #[test]
    fn empty() {
        let mut set = VRegSet::with_capacity(2000);
        assert!(set.is_empty());
        set.insert(VREG(100, Int));
        assert!(!set.is_empty());
        set.remove(100);
        assert!(set.is_empty());
    }
}
