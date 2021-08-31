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

//! Data structures for backtracking allocator.

use crate::cfg::CFGInfo;
use crate::index::ContainerComparator;
use crate::indexset::IndexSet;
use crate::{
    define_index, Allocation, Block, Edit, Function, Inst, MachineEnv, Operand, PReg, ProgPoint,
    RegClass, SpillSlot, VReg,
};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Debug;

/// A range from `from` (inclusive) to `to` (exclusive).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeRange {
    pub from: ProgPoint,
    pub to: ProgPoint,
}

impl CodeRange {
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.from == self.to
    }
    #[inline(always)]
    pub fn contains(&self, other: &Self) -> bool {
        other.from >= self.from && other.to <= self.to
    }
    #[inline(always)]
    pub fn contains_point(&self, other: ProgPoint) -> bool {
        other >= self.from && other < self.to
    }
    #[inline(always)]
    pub fn overlaps(&self, other: &Self) -> bool {
        other.to > self.from && other.from < self.to
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.to.inst().index() - self.from.inst().index()
    }
}

impl std::cmp::PartialOrd for CodeRange {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl std::cmp::Ord for CodeRange {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        if self.to <= other.from {
            Ordering::Less
        } else if self.from >= other.to {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

define_index!(LiveBundleIndex);
define_index!(LiveRangeIndex);
define_index!(SpillSetIndex);
define_index!(UseIndex);
define_index!(VRegIndex);
define_index!(PRegIndex);
define_index!(SpillSlotIndex);

/// Used to carry small sets of bundles, e.g. for conflict sets.
pub type LiveBundleVec = SmallVec<[LiveBundleIndex; 4]>;

#[derive(Clone, Copy, Debug)]
pub struct LiveRangeListEntry {
    pub range: CodeRange,
    pub index: LiveRangeIndex,
}

pub type LiveRangeList = SmallVec<[LiveRangeListEntry; 4]>;
pub type UseList = SmallVec<[Use; 2]>;

#[derive(Clone, Debug)]
pub struct LiveRange {
    pub range: CodeRange,

    pub vreg: VRegIndex,
    pub bundle: LiveBundleIndex,
    pub uses_spill_weight_and_flags: u32,

    pub uses: UseList,

    pub merged_into: LiveRangeIndex,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum LiveRangeFlag {
    StartsAtDef = 1,
}

impl LiveRange {
    #[inline(always)]
    pub fn set_flag(&mut self, flag: LiveRangeFlag) {
        self.uses_spill_weight_and_flags |= (flag as u32) << 29;
    }
    #[inline(always)]
    pub fn clear_flag(&mut self, flag: LiveRangeFlag) {
        self.uses_spill_weight_and_flags &= !((flag as u32) << 29);
    }
    #[inline(always)]
    pub fn assign_flag(&mut self, flag: LiveRangeFlag, val: bool) {
        let bit = if val { (flag as u32) << 29 } else { 0 };
        self.uses_spill_weight_and_flags &= 0xe000_0000;
        self.uses_spill_weight_and_flags |= bit;
    }
    #[inline(always)]
    pub fn has_flag(&self, flag: LiveRangeFlag) -> bool {
        self.uses_spill_weight_and_flags & ((flag as u32) << 29) != 0
    }
    #[inline(always)]
    pub fn flag_word(&self) -> u32 {
        self.uses_spill_weight_and_flags & 0xe000_0000
    }
    #[inline(always)]
    pub fn merge_flags(&mut self, flag_word: u32) {
        self.uses_spill_weight_and_flags |= flag_word;
    }
    #[inline(always)]
    pub fn uses_spill_weight(&self) -> u32 {
        self.uses_spill_weight_and_flags & 0x1fff_ffff
    }
    #[inline(always)]
    pub fn set_uses_spill_weight(&mut self, weight: u32) {
        assert!(weight < (1 << 29));
        self.uses_spill_weight_and_flags =
            (self.uses_spill_weight_and_flags & 0xe000_0000) | weight;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Use {
    pub operand: Operand,
    pub pos: ProgPoint,
    pub slot: u8,
    pub weight: u16,
}

impl Use {
    #[inline(always)]
    pub fn new(operand: Operand, pos: ProgPoint, slot: u8) -> Self {
        Self {
            operand,
            pos,
            slot,
            // Weight is updated on insertion into LR.
            weight: 0,
        }
    }
}

pub const SLOT_NONE: u8 = u8::MAX;

#[derive(Clone, Debug)]
pub struct LiveBundle {
    pub ranges: LiveRangeList,
    pub spillset: SpillSetIndex,
    pub allocation: Allocation,
    pub prio: u32, // recomputed after every bulk update
    pub spill_weight_and_props: u32,
}

impl LiveBundle {
    #[inline(always)]
    pub fn set_cached_spill_weight_and_props(
        &mut self,
        spill_weight: u32,
        minimal: bool,
        fixed: bool,
        stack: bool,
    ) {
        debug_assert!(spill_weight < ((1 << 29) - 1));
        self.spill_weight_and_props = spill_weight
            | (if minimal { 1 << 31 } else { 0 })
            | (if fixed { 1 << 30 } else { 0 })
            | (if stack { 1 << 29 } else { 0 });
    }

    #[inline(always)]
    pub fn cached_minimal(&self) -> bool {
        self.spill_weight_and_props & (1 << 31) != 0
    }

    #[inline(always)]
    pub fn cached_fixed(&self) -> bool {
        self.spill_weight_and_props & (1 << 30) != 0
    }

    #[inline(always)]
    pub fn cached_stack(&self) -> bool {
        self.spill_weight_and_props & (1 << 29) != 0
    }

    #[inline(always)]
    pub fn set_cached_fixed(&mut self) {
        self.spill_weight_and_props |= 1 << 30;
    }

    #[inline(always)]
    pub fn set_cached_stack(&mut self) {
        self.spill_weight_and_props |= 1 << 29;
    }

    #[inline(always)]
    pub fn cached_spill_weight(&self) -> u32 {
        self.spill_weight_and_props & ((1 << 29) - 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BundleProperties {
    pub minimal: bool,
    pub fixed: bool,
}

#[derive(Clone, Debug)]
pub struct SpillSet {
    pub vregs: SmallVec<[VRegIndex; 2]>,
    pub slot: SpillSlotIndex,
    pub reg_hint: PReg,
    pub class: RegClass,
    pub spill_bundle: LiveBundleIndex,
    pub required: bool,
    pub size: u8,
}

#[derive(Clone, Debug)]
pub struct VRegData {
    pub ranges: LiveRangeList,
    pub blockparam: Block,
    pub is_ref: bool,
    pub is_pinned: bool,
}

#[derive(Clone, Debug)]
pub struct PRegData {
    pub reg: PReg,
    pub allocations: LiveRangeSet,
}

#[derive(Clone, Debug)]
pub struct Env<'a, F: Function> {
    pub func: &'a F,
    pub env: &'a MachineEnv,
    pub cfginfo: CFGInfo,
    pub liveins: Vec<IndexSet>,
    pub liveouts: Vec<IndexSet>,
    /// Blockparam outputs: from-vreg, (end of) from-block, (start of)
    /// to-block, to-vreg. The field order is significant: these are sorted so
    /// that a scan over vregs, then blocks in each range, can scan in
    /// order through this (sorted) list and add allocs to the
    /// half-move list.
    pub blockparam_outs: Vec<(VRegIndex, Block, Block, VRegIndex)>,
    /// Blockparam inputs: to-vreg, (start of) to-block, (end of)
    /// from-block. As above for `blockparam_outs`, field order is
    /// significant.
    pub blockparam_ins: Vec<(VRegIndex, Block, Block)>,
    /// Blockparam allocs: block, idx, vreg, alloc. Info to describe
    /// blockparam locations at block entry, for metadata purposes
    /// (e.g. for the checker).
    pub blockparam_allocs: Vec<(Block, u32, VRegIndex, Allocation)>,

    pub ranges: Vec<LiveRange>,
    pub bundles: Vec<LiveBundle>,
    pub spillsets: Vec<SpillSet>,
    pub vregs: Vec<VRegData>,
    pub vreg_regs: Vec<VReg>,
    pub pregs: Vec<PRegData>,
    pub allocation_queue: PrioQueue,
    pub clobbers: Vec<Inst>,   // Sorted list of insts with clobbers.
    pub safepoints: Vec<Inst>, // Sorted list of safepoint insts.
    pub safepoints_per_vreg: HashMap<usize, HashSet<Inst>>,

    pub spilled_bundles: Vec<LiveBundleIndex>,
    pub spillslots: Vec<SpillSlotData>,
    pub slots_by_size: Vec<SpillSlotList>,

    pub extra_spillslot: Vec<Option<Allocation>>,

    // Program moves: these are moves in the provided program that we
    // handle with our internal machinery, in order to avoid the
    // overhead of ordinary operand processing. We expect the client
    // to not generate any code for instructions that return
    // `Some(..)` for `.is_move()`, and instead use the edits that we
    // provide to implement those moves (or some simplified version of
    // them) post-regalloc.
    //
    // (from-vreg, inst, from-alloc), sorted by (from-vreg, inst)
    pub prog_move_srcs: Vec<((VRegIndex, Inst), Allocation)>,
    // (to-vreg, inst, to-alloc), sorted by (to-vreg, inst)
    pub prog_move_dsts: Vec<((VRegIndex, Inst), Allocation)>,
    // (from-vreg, to-vreg) for bundle-merging.
    pub prog_move_merges: Vec<(LiveRangeIndex, LiveRangeIndex)>,

    // When multiple fixed-register constraints are present on a
    // single VReg at a single program point (this can happen for,
    // e.g., call args that use the same value multiple times), we
    // remove all but one of the fixed-register constraints, make a
    // note here, and add a clobber with that PReg instread to keep
    // the register available. When we produce the final edit-list, we
    // will insert a copy from wherever the VReg's primary allocation
    // was to the approprate PReg.
    //
    // (progpoint, copy-from-preg, copy-to-preg, to-slot)
    pub multi_fixed_reg_fixups: Vec<(ProgPoint, PRegIndex, PRegIndex, usize)>,

    pub inserted_moves: Vec<InsertedMove>,

    // Output:
    pub edits: Vec<(u32, InsertMovePrio, Edit)>,
    pub allocs: Vec<Allocation>,
    pub inst_alloc_offsets: Vec<u32>,
    pub num_spillslots: u32,
    pub safepoint_slots: Vec<(ProgPoint, SpillSlot)>,

    pub allocated_bundle_count: usize,

    pub stats: Stats,

    // For debug output only: a list of textual annotations at every
    // ProgPoint to insert into the final allocated program listing.
    pub debug_annotations: std::collections::HashMap<ProgPoint, Vec<String>>,
    pub annotations_enabled: bool,
}

#[derive(Clone, Debug)]
pub struct SpillSlotData {
    pub ranges: LiveRangeSet,
    pub class: RegClass,
    pub alloc: Allocation,
    pub next_spillslot: SpillSlotIndex,
}

#[derive(Clone, Debug)]
pub struct SpillSlotList {
    pub first_spillslot: SpillSlotIndex,
    pub last_spillslot: SpillSlotIndex,
}

#[derive(Clone, Debug)]
pub struct PrioQueue {
    pub heap: std::collections::BinaryHeap<PrioQueueEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PrioQueueEntry {
    pub prio: u32,
    pub bundle: LiveBundleIndex,
    pub reg_hint: PReg,
}

#[derive(Clone, Debug)]
pub struct LiveRangeSet {
    pub btree: BTreeMap<LiveRangeKey, LiveRangeIndex>,
}

#[derive(Clone, Copy, Debug)]
pub struct LiveRangeKey {
    pub from: u32,
    pub to: u32,
}

impl LiveRangeKey {
    #[inline(always)]
    pub fn from_range(range: &CodeRange) -> Self {
        Self {
            from: range.from.to_index(),
            to: range.to.to_index(),
        }
    }

    #[inline(always)]
    pub fn to_range(&self) -> CodeRange {
        CodeRange {
            from: ProgPoint::from_index(self.from),
            to: ProgPoint::from_index(self.to),
        }
    }
}

impl std::cmp::PartialEq for LiveRangeKey {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to > other.from && self.from < other.to
    }
}
impl std::cmp::Eq for LiveRangeKey {}
impl std::cmp::PartialOrd for LiveRangeKey {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl std::cmp::Ord for LiveRangeKey {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.to <= other.from {
            std::cmp::Ordering::Less
        } else if self.from >= other.to {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

pub struct PrioQueueComparator<'a> {
    pub prios: &'a [usize],
}
impl<'a> ContainerComparator for PrioQueueComparator<'a> {
    type Ix = LiveBundleIndex;
    fn compare(&self, a: Self::Ix, b: Self::Ix) -> std::cmp::Ordering {
        self.prios[a.index()].cmp(&self.prios[b.index()])
    }
}

impl PrioQueue {
    pub fn new() -> Self {
        PrioQueue {
            heap: std::collections::BinaryHeap::new(),
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, bundle: LiveBundleIndex, prio: usize, reg_hint: PReg) {
        self.heap.push(PrioQueueEntry {
            prio: prio as u32,
            bundle,
            reg_hint,
        });
    }

    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self.heap.is_empty()
    }

    #[inline(always)]
    pub fn pop(&mut self) -> Option<(LiveBundleIndex, PReg)> {
        self.heap.pop().map(|entry| (entry.bundle, entry.reg_hint))
    }
}

impl LiveRangeSet {
    pub(crate) fn new() -> Self {
        Self {
            btree: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct InsertedMove {
    pub pos: ProgPoint,
    pub prio: InsertMovePrio,
    pub from_alloc: Allocation,
    pub to_alloc: Allocation,
    pub to_vreg: Option<VReg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum InsertMovePrio {
    InEdgeMoves,
    BlockParam,
    Regular,
    PostRegular,
    MultiFixedReg,
    ReusedInput,
    OutEdgeMoves,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub livein_blocks: usize,
    pub livein_iterations: usize,
    pub initial_liverange_count: usize,
    pub merged_bundle_count: usize,
    pub prog_moves: usize,
    pub prog_moves_dead_src: usize,
    pub prog_move_merge_attempt: usize,
    pub prog_move_merge_success: usize,
    pub process_bundle_count: usize,
    pub process_bundle_reg_probes_fixed: usize,
    pub process_bundle_reg_success_fixed: usize,
    pub process_bundle_bounding_range_probe_start_any: usize,
    pub process_bundle_bounding_range_probes_any: usize,
    pub process_bundle_bounding_range_success_any: usize,
    pub process_bundle_reg_probe_start_any: usize,
    pub process_bundle_reg_probes_any: usize,
    pub process_bundle_reg_success_any: usize,
    pub evict_bundle_event: usize,
    pub evict_bundle_count: usize,
    pub splits: usize,
    pub splits_clobbers: usize,
    pub splits_hot: usize,
    pub splits_conflicts: usize,
    pub splits_defs: usize,
    pub splits_all: usize,
    pub final_liverange_count: usize,
    pub final_bundle_count: usize,
    pub spill_bundle_count: usize,
    pub spill_bundle_reg_probes: usize,
    pub spill_bundle_reg_success: usize,
    pub blockparam_ins_count: usize,
    pub blockparam_outs_count: usize,
    pub blockparam_allocs_count: usize,
    pub halfmoves_count: usize,
    pub edits_count: usize,
}
