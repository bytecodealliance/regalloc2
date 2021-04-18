/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

//! Bit vectors.

use smallvec::{smallvec, SmallVec};

/// A conceptually infinite-length bitvector that allows bitwise operations and
/// iteration over set bits efficiently.
#[derive(Clone, Debug)]
pub struct BitVec {
    bits: SmallVec<[u64; 2]>,
}

const BITS_PER_WORD: usize = 64;

impl BitVec {
    pub fn new() -> Self {
        Self { bits: smallvec![] }
    }

    pub fn with_capacity(len: usize) -> Self {
        let words = (len + BITS_PER_WORD - 1) / BITS_PER_WORD;
        Self {
            bits: SmallVec::with_capacity(words),
        }
    }

    #[inline(never)]
    fn ensure_idx(&mut self, word: usize) {
        let mut target_len = std::cmp::max(2, self.bits.len());
        while word >= target_len {
            target_len *= 2;
        }
        self.bits.resize(target_len, 0);
    }

    #[inline(always)]
    pub fn set(&mut self, idx: usize, val: bool) {
        let word = idx / BITS_PER_WORD;
        let bit = idx % BITS_PER_WORD;
        if val {
            if word >= self.bits.len() {
                self.ensure_idx(word);
            }
            self.bits[word] |= 1 << bit;
        } else {
            if word < self.bits.len() {
                self.bits[word] &= !(1 << bit);
            }
        }
    }

    pub fn assign(&mut self, other: &Self) {
        if other.bits.len() > 0 {
            self.ensure_idx(other.bits.len() - 1);
        }
        for i in 0..other.bits.len() {
            self.bits[i] = other.bits[i];
        }
        for i in other.bits.len()..self.bits.len() {
            self.bits[i] = 0;
        }
    }

    #[inline(always)]
    pub fn get(&mut self, idx: usize) -> bool {
        let word = idx / BITS_PER_WORD;
        let bit = idx % BITS_PER_WORD;
        if word >= self.bits.len() {
            false
        } else {
            (self.bits[word] & (1 << bit)) != 0
        }
    }

    pub fn or(&mut self, other: &Self) -> bool {
        if other.bits.is_empty() {
            return false;
        }
        let last_idx = other.bits.len() - 1;
        self.ensure_idx(last_idx);

        let mut changed = false;
        for (self_word, other_word) in self.bits.iter_mut().zip(other.bits.iter()) {
            if *other_word & !*self_word != 0 {
                changed = true;
            }
            *self_word |= *other_word;
        }
        changed
    }

    pub fn and(&mut self, other: &Self) {
        if other.bits.len() < self.bits.len() {
            self.bits.truncate(other.bits.len());
        }

        for (self_word, other_word) in self.bits.iter_mut().zip(other.bits.iter()) {
            *self_word &= *other_word;
        }
    }

    pub fn iter<'a>(&'a self) -> SetBitsIter<'a> {
        let cur_word = if self.bits.len() > 0 { self.bits[0] } else { 0 };
        SetBitsIter {
            words: &self.bits[..],
            word_idx: 0,
            cur_word,
        }
    }
}

impl std::cmp::PartialEq for BitVec {
    fn eq(&self, other: &Self) -> bool {
        let limit = std::cmp::min(self.bits.len(), other.bits.len());
        for i in 0..limit {
            if self.bits[i] != other.bits[i] {
                return false;
            }
        }
        for i in limit..self.bits.len() {
            if self.bits[i] != 0 {
                return false;
            }
        }
        for i in limit..other.bits.len() {
            if other.bits[i] != 0 {
                return false;
            }
        }
        true
    }
}
impl std::cmp::Eq for BitVec {}

pub struct SetBitsIter<'a> {
    words: &'a [u64],
    word_idx: usize,
    cur_word: u64,
}

impl<'a> Iterator for SetBitsIter<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        while self.cur_word == 0 {
            if self.word_idx + 1 >= self.words.len() {
                return None;
            }
            self.word_idx += 1;
            self.cur_word = self.words[self.word_idx];
        }
        let bitidx = self.cur_word.trailing_zeros();
        self.cur_word &= !(1 << bitidx);
        Some(self.word_idx * BITS_PER_WORD + bitidx as usize)
    }
}

#[cfg(test)]
mod test {
    use super::BitVec;

    #[test]
    fn test_set_bits_iter() {
        let mut vec = BitVec::new();
        let mut sum = 0;
        for i in 0..1024 {
            if i % 17 == 0 {
                vec.set(i, true);
                sum += i;
            }
        }

        let mut checksum = 0;
        for bit in vec.iter() {
            assert!(bit % 17 == 0);
            checksum += bit;
        }

        assert_eq!(sum, checksum);
    }
}
