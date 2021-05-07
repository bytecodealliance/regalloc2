/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

//! Bit vectors.

use fxhash::FxHashMap;

const SMALL_ELEMS: usize = 12;

/// A hybrid large/small-mode sparse mapping from integer indices to elements.
#[derive(Clone, Debug)]
enum AdaptiveMap {
    Small(u32, [u32; SMALL_ELEMS], [u64; SMALL_ELEMS]),
    Large(FxHashMap<u32, u64>),
}

const INVALID: u32 = 0xffff_ffff;

impl AdaptiveMap {
    fn new() -> Self {
        Self::Small(0, [INVALID; SMALL_ELEMS], [0; SMALL_ELEMS])
    }
    #[inline(never)]
    fn expand(&mut self) {
        match self {
            &mut Self::Small(len, ref keys, ref values) => {
                let mut map = FxHashMap::default();
                for i in 0..len {
                    map.insert(keys[i as usize], values[i as usize]);
                }
                *self = Self::Large(map);
            }
            _ => {}
        }
    }
    #[inline(always)]
    fn get_or_insert<'a>(&'a mut self, key: u32) -> &'a mut u64 {
        let needs_expand = match self {
            &mut Self::Small(len, ref keys, ..) => {
                len == SMALL_ELEMS as u32 && !keys.iter().any(|k| *k == key)
            }
            _ => false,
        };
        if needs_expand {
            self.expand();
        }

        match self {
            &mut Self::Small(ref mut len, ref mut keys, ref mut values) => {
                for i in 0..*len {
                    if keys[i as usize] == key {
                        return &mut values[i as usize];
                    }
                }
                assert!(*len < SMALL_ELEMS as u32);
                let idx = *len;
                *len += 1;
                keys[idx as usize] = key;
                values[idx as usize] = 0;
                &mut values[idx as usize]
            }
            &mut Self::Large(ref mut map) => map.entry(key).or_insert(0),
        }
    }
    #[inline(always)]
    fn get_mut(&mut self, key: u32) -> Option<&mut u64> {
        match self {
            &mut Self::Small(len, ref keys, ref mut values) => {
                for i in 0..len {
                    if keys[i as usize] == key {
                        return Some(&mut values[i as usize]);
                    }
                }
                None
            }
            &mut Self::Large(ref mut map) => map.get_mut(&key),
        }
    }
    #[inline(always)]
    fn get(&self, key: u32) -> Option<&u64> {
        match self {
            &Self::Small(len, ref keys, ref values) => {
                for i in 0..len {
                    if keys[i as usize] == key {
                        return Some(&values[i as usize]);
                    }
                }
                None
            }
            &Self::Large(ref map) => map.get(&key),
        }
    }
    fn iter<'a>(&'a self) -> AdaptiveMapIter<'a> {
        match self {
            &Self::Small(len, ref keys, ref values) => {
                AdaptiveMapIter::Small(&keys[0..len as usize], &values[0..len as usize])
            }
            &Self::Large(ref map) => AdaptiveMapIter::Large(map.iter()),
        }
    }
}

enum AdaptiveMapIter<'a> {
    Small(&'a [u32], &'a [u64]),
    Large(std::collections::hash_map::Iter<'a, u32, u64>),
}

impl<'a> std::iter::Iterator for AdaptiveMapIter<'a> {
    type Item = (u32, u64);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            &mut Self::Small(ref mut keys, ref mut values) => {
                if keys.is_empty() {
                    None
                } else {
                    let (k, v) = ((*keys)[0], (*values)[0]);
                    *keys = &(*keys)[1..];
                    *values = &(*values)[1..];
                    Some((k, v))
                }
            }
            &mut Self::Large(ref mut it) => it.next().map(|(&k, &v)| (k, v)),
        }
    }
}

/// A conceptually infinite-length bitvector that allows bitwise operations and
/// iteration over set bits efficiently.
#[derive(Clone, Debug)]
pub struct BitVec {
    elems: AdaptiveMap,
}

const BITS_PER_WORD: usize = 64;

impl BitVec {
    pub fn new() -> Self {
        Self {
            elems: AdaptiveMap::new(),
        }
    }

    #[inline(always)]
    fn elem(&mut self, bit_index: usize) -> &mut u64 {
        let word_index = (bit_index / BITS_PER_WORD) as u32;
        self.elems.get_or_insert(word_index)
    }

    #[inline(always)]
    fn maybe_elem_mut(&mut self, bit_index: usize) -> Option<&mut u64> {
        let word_index = (bit_index / BITS_PER_WORD) as u32;
        self.elems.get_mut(word_index)
    }

    #[inline(always)]
    fn maybe_elem(&self, bit_index: usize) -> Option<&u64> {
        let word_index = (bit_index / BITS_PER_WORD) as u32;
        self.elems.get(word_index)
    }

    #[inline(always)]
    pub fn set(&mut self, idx: usize, val: bool) {
        let bit = idx % BITS_PER_WORD;
        if val {
            *self.elem(idx) |= 1 << bit;
        } else if let Some(word) = self.maybe_elem_mut(idx) {
            *word &= !(1 << bit);
        }
    }

    pub fn assign(&mut self, other: &Self) {
        self.elems = other.elems.clone();
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> bool {
        let bit = idx % BITS_PER_WORD;
        if let Some(word) = self.maybe_elem(idx) {
            (*word & (1 << bit)) != 0
        } else {
            false
        }
    }

    pub fn or(&mut self, other: &Self) -> bool {
        let mut changed = 0;
        for (word_idx, bits) in other.elems.iter() {
            if bits == 0 {
                continue;
            }
            let word_idx = word_idx as usize;
            let self_word = self.elem(word_idx * BITS_PER_WORD);
            changed |= bits & !*self_word;
            *self_word |= bits;
        }
        changed != 0
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
        self.elems.iter().flat_map(|(word_idx, bits)| {
            let word_idx = word_idx as usize;
            set_bits(bits).map(move |i| BITS_PER_WORD * word_idx + i)
        })
    }
}

fn set_bits(bits: u64) -> impl Iterator<Item = usize> {
    let iter = SetBitsIter(bits);
    iter
}

pub struct SetBitsIter(u64);

impl Iterator for SetBitsIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            let bitidx = self.0.trailing_zeros();
            self.0 &= !(1 << bitidx);
            Some(bitidx as usize)
        }
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
