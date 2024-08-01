use alloc::vec::Vec;
use alloc::vec;
use core::ops::IndexMut;
use std::ops::Index;
use crate::{RegClass, PReg};

/// A least-recently-used cache organized as a linked list based on a vector.
#[derive(Debug)]
pub struct Lru {
    /// The list of node information.
    ///
    /// Each node corresponds to a physical register.
    /// The index of a node is the `address` from the perspective of the linked list.
    pub data: Vec<LruNode>,
    /// Index of the most recently used register.
    pub head: usize,
    /// Class of registers in the cache.
    pub regclass: RegClass,
}

#[derive(Clone, Copy, Debug)]
pub struct LruNode {
    /// The previous physical register in the list.
    pub prev: usize,
    /// The next physical register in the list.
    pub next: usize,
}

impl Lru {
    pub fn new(regclass: RegClass, regs: &[PReg]) -> Self {
        let mut data = vec![LruNode { prev: 0, next: 0 }; PReg::MAX + 1];
        let no_of_regs = regs.len();
        for i in 0..no_of_regs {
            let (reg, prev_reg, next_reg) = (
                regs[i],
                regs[i.checked_sub(1).unwrap_or(no_of_regs - 1)],
                regs[if i >= no_of_regs - 1 { 0 } else { i + 1 }]
            );
            data[reg.hw_enc()].prev = prev_reg.hw_enc();
            data[reg.hw_enc()].next = next_reg.hw_enc();
        }
        Self {
            head: if regs.is_empty() { usize::MAX } else { regs[0].hw_enc() },
            data,
            regclass,
        }
    }

    /// Marks the physical register `i` as the most recently used
    /// and sets `vreg` as the virtual register it contains.
    pub fn poke(&mut self, preg: PReg) {
        let prev_newest = self.head;
        let i = preg.hw_enc();
        if i == prev_newest {
            return;
        }
        if self.data[prev_newest].prev != i {
            self.remove(i);
            self.insert_before(i, self.head);
        }
        self.head = i;
    }

    /// Gets the least recently used physical register.
    pub fn pop(&mut self) -> PReg {
        let oldest = self.data[self.head].prev;
        PReg::new(oldest, self.regclass)
    }

    /// Splices out a node from the list.
    pub fn remove(&mut self, i: usize) {
        let (iprev, inext) = (self.data[i].prev, self.data[i].next);
        self.data[iprev].next = self.data[i].next;
        self.data[inext].prev = self.data[i].prev;
    }

    /// Sets the node `i` to the last in the list.
    pub fn append(&mut self, i: usize) {
        let last_node = self.data[self.head].prev;
        self.data[last_node].next = i;
        self.data[self.head].prev = i;
        self.data[i].prev = last_node;
        self.data[i].next = self.head;
    }

    /// Insert node `i` before node `j` in the list.
    pub fn insert_before(&mut self, i: usize, j: usize) {
        let prev = self.data[j].prev;
        self.data[prev].next = i;
        self.data[j].prev = i;
        self.data[i] = LruNode {
            next: j,
            prev,
        };
    }
}

#[derive(Debug)]
pub struct PartedByRegClass<T: std::fmt::Debug> {
    pub items: [T; 3],
}

impl<T: std::fmt::Debug> Index<RegClass> for PartedByRegClass<T> {
    type Output = T;

    fn index(&self, index: RegClass) -> &Self::Output {
        &self.items[index as usize]
    }
}

impl<T: std::fmt::Debug> IndexMut<RegClass> for PartedByRegClass<T> {
    fn index_mut(&mut self, index: RegClass) -> &mut Self::Output {
        &mut self.items[index as usize]
    }
}

/// Least-recently-used caches for register classes Int, Float, and Vector, respectively.
pub type Lrus = PartedByRegClass<Lru>;

impl Lrus {
    pub fn new(int_regs: &[PReg], float_regs: &[PReg], vec_regs: &[PReg]) -> Self {
        Self {
            items: [
                Lru::new(RegClass::Int, int_regs),
                Lru::new(RegClass::Float, float_regs),
                Lru::new(RegClass::Vector, vec_regs),
            ]
        }
    }
}
