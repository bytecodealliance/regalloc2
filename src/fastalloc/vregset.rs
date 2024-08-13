use alloc::vec::Vec;
use alloc::vec;
use core::fmt;
use core::convert::{TryFrom, TryInto};
use crate::{RegClass, VReg};

struct RegClassNum;

impl RegClassNum {
    const INVALID: u8 = 0b00;
    const MAX: u8 = 0b11;
    // 0b11
    const INT: u8 = Self::MAX - RegClass::Int as u8;
    // 0b10
    const FLOAT: u8 = Self::MAX - RegClass::Float as u8;
    // 0b01
    const VECTOR: u8 = Self::MAX - RegClass::Vector as u8;
}

impl TryFrom<u64> for RegClass {
    type Error = ();
    fn try_from(value: u64) -> Result<Self, ()> {
        if value == RegClassNum::INT as u64 {
            Ok(RegClass::Int)
        } else if value == RegClassNum::FLOAT as u64 {
            Ok(RegClass::Float)
        } else if value == RegClassNum::VECTOR as u64 {
            Ok(RegClass::Vector)
        } else if value == RegClassNum::INVALID as u64 {
            Err(())
        } else {
            unreachable!()
        }
    }
}

impl From<RegClass> for Frame {
    fn from(value: RegClass) -> Self {
        (match value {
            RegClass::Int => RegClassNum::INT,
            RegClass::Float => RegClassNum::FLOAT,
            RegClass::Vector => RegClassNum::VECTOR
        }) as Frame
    }
}

type Frame = u64;
const BITS_PER_FRAME: usize = core::mem::size_of::<Frame>() * 8;
const VREGS_PER_FRAME: usize = BITS_PER_FRAME / 2;
const EMPTY_FRAME: Frame = RegClassNum::INVALID as Frame;

pub struct VRegSet {
    bits: Vec<Frame>
}

impl VRegSet {

    pub fn with_capacity(n: usize) -> Self {
        let no_of_bits_needed = 2 * n;
        let quot = no_of_bits_needed / BITS_PER_FRAME;
        let no_of_frames = quot + 1;
        Self {
            bits: vec![RegClassNum::INVALID as Frame; no_of_frames],
        }
    }

    fn compute_index(&self, el: usize) -> (usize, usize) {
        (el / BITS_PER_FRAME, el % BITS_PER_FRAME)
    }

    pub fn insert(&mut self, vreg: VReg) {
        let (frame_no, idx) = self.compute_index(vreg.vreg() * 2);
        let reg_class_num: Frame = vreg.class().into();
        self.bits[frame_no] |= reg_class_num << idx;
    }

    pub fn remove(&mut self, vreg_num: usize) {
        let (frame_no, idx) = self.compute_index(vreg_num * 2);
        self.bits[frame_no] &= !(0b11 << idx);
    }

    pub fn contains(&self, vreg_num: usize) -> bool {
        let (frame_no, idx) = self.compute_index(vreg_num * 2);
        self.bits[frame_no] & (0b11 << idx) != RegClassNum::INVALID as Frame
    }

    pub fn clear(&mut self) {
        for frame in self.bits.iter_mut() {
            *frame = RegClassNum::INVALID as Frame;
        }
    }

    pub fn is_empty(&mut self) -> bool {
        self.bits.iter()
            .all(|frame| *frame == EMPTY_FRAME)
    }

    pub fn iter(&self) -> BitSetIter {
        BitSetIter {
            next_frame_idx: 0,
            curr_frame: EMPTY_FRAME,
            bits: &self.bits
        }
    }
}

pub struct BitSetIter<'a> {
    next_frame_idx: usize,
    curr_frame: Frame,
    bits: &'a [Frame]
}

impl<'a> Iterator for BitSetIter<'a> {
    type Item = VReg;

    fn next(&mut self) -> Option<VReg> {
        loop {
            while self.curr_frame == EMPTY_FRAME {
                if self.next_frame_idx >= self.bits.len() {
                    return None;
                }
                self.curr_frame = self.bits[self.next_frame_idx];
                self.next_frame_idx += 1;
            }
            let mut skip = self.curr_frame.trailing_zeros();
            if skip % 2 != 0 {
                skip -= 1;
            }
            let vreg_num = (self.next_frame_idx - 1) * VREGS_PER_FRAME + (skip / 2) as usize;
            let class = (self.curr_frame >> skip) & 0b11;
            self.curr_frame &= !(0b11 << skip);
            return Some(VReg::new(vreg_num, class.try_into().unwrap()));
        }
    }
}


impl fmt::Debug for VRegSet {
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
            VREG(11, Vector), 
            VREG(23, Int), 
            VREG(199, Float), 
            VREG(2000, Int)
        ];
        for (actual_el, expected_el) in set.iter().zip(els.iter()) {
            assert_eq!(actual_el, *expected_el);
        }
        assert!(set.contains(10));
        assert!(!set.contains(12));
        assert!(!set.contains(197));
        assert!(set.contains(23));
        assert!(set.contains(11));
        set.remove(23);
        assert!(!set.contains(23));
        set.insert(VREG(73, Vector));
        let els = [
            VREG(10, Int),
            VREG(11, Vector),
            VREG(73, Vector),
            VREG(199, Float),
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
