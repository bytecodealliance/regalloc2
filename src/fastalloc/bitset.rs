use alloc::vec::Vec;
use alloc::vec;

type Frame = u64;
const BITS_PER_FRAME: usize = core::mem::size_of::<Frame>() * 8;

pub struct BitSet {
    bits: Vec<Frame>
}

impl BitSet {

    pub fn with_capacity(n: usize) -> Self {
        let quot = n / BITS_PER_FRAME;
        // The number of frames needed cannot be > the quotient;
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

    pub fn is_empty(&mut self) {
        
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
        set.insert(23);
        set.insert(45);
        assert!(set.contains(10));
        assert!(!set.contains(12));
        assert!(!set.contains(2000));
        assert!(set.contains(45));
        assert!(set.contains(23));
        assert!(set.contains(11));
        set.remove(10);
        assert!(!set.contains(10));
    }
}
