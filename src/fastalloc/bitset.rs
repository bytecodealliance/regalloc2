use alloc::vec;
use alloc::vec::Vec;

type Frame = u64;
const BITS_PER_FRAME: usize = core::mem::size_of::<Frame>() * 8;

pub struct BitSet {
    bits: Vec<Frame>,
}

impl BitSet {
    pub fn with_capacity(n: usize) -> Self {
        let quot = n / BITS_PER_FRAME;
        let no_of_frames = quot + 1;
        Self {
            bits: vec![0; no_of_frames],
        }
    }

    pub fn compute_index(&self, el: usize) -> (usize, usize) {
        (el / BITS_PER_FRAME, el % BITS_PER_FRAME)
    }

    pub fn insert(&mut self, el: usize) {
        let (frame_no, idx) = self.compute_index(el);
        self.bits[frame_no] |= 1 << idx;
    }

    pub fn remove(&mut self, el: usize) {
        let (frame_no, idx) = self.compute_index(el);
        self.bits[frame_no] &= !(1 << idx);
    }

    pub fn contains(&self, el: usize) -> bool {
        let (frame_no, idx) = self.compute_index(el);
        self.bits[frame_no] & (1 << idx) != 0
    }

    pub fn clear(&mut self) {
        for frame in self.bits.iter_mut() {
            *frame = 0;
        }
    }

    pub fn is_empty(&mut self) -> bool {
        self.bits.iter().all(|frame| *frame == 0)
    }

    pub fn iter(&self) -> BitSetIter {
        BitSetIter {
            next_frame_idx: 0,
            curr_frame: 0,
            bits: &self.bits,
        }
    }
}

pub struct BitSetIter<'a> {
    next_frame_idx: usize,
    curr_frame: Frame,
    bits: &'a [Frame],
}

impl<'a> Iterator for BitSetIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            while self.curr_frame == 0 {
                if self.next_frame_idx >= self.bits.len() {
                    return None;
                }
                self.curr_frame = self.bits[self.next_frame_idx];
                self.next_frame_idx += 1;
            }
            let skip = self.curr_frame.trailing_zeros();
            self.curr_frame &= !(1 << skip);
            return Some((self.next_frame_idx - 1) * BITS_PER_FRAME + skip as usize);
        }
    }
}

use core::fmt;

impl fmt::Debug for BitSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ ")?;
        for el in self.iter() {
            write!(f, "{el} ")?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operations() {
        let mut set = BitSet::with_capacity(200);
        set.insert(10);
        set.insert(11);
        set.insert(199);
        set.insert(23);
        set.insert(45);
        let els = [10, 11, 23, 45, 199];
        for (actual_el, expected_el) in set.iter().zip(els.iter()) {
            assert_eq!(actual_el, *expected_el as usize);
        }
        assert!(set.contains(10));
        assert!(!set.contains(12));
        assert!(!set.contains(197));
        assert!(set.contains(45));
        assert!(set.contains(23));
        assert!(set.contains(11));
        set.remove(23);
        assert!(!set.contains(23));
        set.insert(73);
        let els = [10, 11, 45, 73, 199];
        for (actual_el, expected_el) in set.iter().zip(els.iter()) {
            assert_eq!(actual_el, *expected_el as usize);
        }
    }

    #[test]
    fn empty() {
        let mut set = BitSet::with_capacity(2000);
        assert!(set.is_empty());
        set.insert(100);
        assert!(!set.is_empty());
        set.remove(100);
        assert!(set.is_empty());
    }
}
