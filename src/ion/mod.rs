/*
 * The following license applies to this file, which has been largely
 * derived from the files `js/src/jit/BacktrackingAllocator.h` and
 * `js/src/jit/BacktrackingAllocator.cpp` in Mozilla Firefox:
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

//! Backtracking register allocator on SSA code ported from IonMonkey's
//! BacktrackingAllocator.

/*
 * TODO:
 *
 * - tune heuristics:
 *   - splits:
 *     - safepoints?
 *     - split just before uses with fixed regs and/or just after defs
 *       with fixed regs?
 *   - measure average liverange length / number of splits / ...
 *
 * - reused-input reg: don't allocate register for input that is reused.
 *
 * - "Fixed-stack location": negative spillslot numbers?
 *
 * - Rematerialization
 */

/*
   Performance ideas:

   - conflict hints? (note on one bundle that it definitely conflicts
     with another, so avoid probing the other's alloc)

   - partial allocation -- place one LR, split rest off into separate
     bundle, in one pass?

   - coarse-grained "register contention" counters per fixed region;
     randomly sample these, adding up a vector of them, to choose
     register probe order?
 */

#![allow(dead_code, unused_imports)]

use crate::bitvec::BitVec;
use crate::cfg::CFGInfo;
use crate::index::ContainerComparator;
use crate::moves::ParallelMoves;
use crate::{
    define_index, domtree, Allocation, AllocationKind, Block, Edit, Function, Inst, InstPosition,
    MachineEnv, Operand, OperandKind, OperandPolicy, OperandPos, Output, PReg, ProgPoint,
    RegAllocError, RegClass, SpillSlot, VReg,
};
use fxhash::FxHashSet;
use log::debug;
use smallvec::{smallvec, SmallVec};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::fmt::Debug;

/// A range from `from` (inclusive) to `to` (exclusive).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeRange {
    from: ProgPoint,
    to: ProgPoint,
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

type LiveBundleVec = SmallVec<[LiveBundleIndex; 4]>;

#[derive(Clone, Debug)]
struct LiveRangeHot {
    range: CodeRange,
    next_in_bundle: LiveRangeIndex,
}

#[derive(Clone, Debug)]
struct LiveRange {
    vreg: VRegIndex,
    bundle: LiveBundleIndex,
    uses_spill_weight_and_flags: u32,

    first_use: UseIndex,
    last_use: UseIndex,

    next_in_reg: LiveRangeIndex,

    merged_into: LiveRangeIndex,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum LiveRangeFlag {
    Minimal = 1,
    Fixed = 2,
}

impl LiveRange {
    #[inline(always)]
    pub fn set_flag(&mut self, flag: LiveRangeFlag) {
        self.uses_spill_weight_and_flags |= (flag as u32) << 30;
    }
    #[inline(always)]
    pub fn clear_flag(&mut self, flag: LiveRangeFlag) {
        self.uses_spill_weight_and_flags &= !((flag as u32) << 30);
    }
    #[inline(always)]
    pub fn has_flag(&self, flag: LiveRangeFlag) -> bool {
        self.uses_spill_weight_and_flags & ((flag as u32) << 30) != 0
    }
    #[inline(always)]
    pub fn uses_spill_weight(&self) -> u32 {
        self.uses_spill_weight_and_flags & 0x3fff_ffff
    }
    #[inline(always)]
    pub fn set_uses_spill_weight(&mut self, weight: u32) {
        assert!(weight < (1 << 30));
        self.uses_spill_weight_and_flags =
            (self.uses_spill_weight_and_flags & 0xc000_0000) | weight;
    }
}

#[derive(Clone, Debug)]
struct Use {
    operand: Operand,
    pos: ProgPoint,
    next_use_and_slot: u32,
}

impl Use {
    #[inline(always)]
    fn new(operand: Operand, pos: ProgPoint, next_use: UseIndex, slot: u8) -> Self {
        debug_assert!(next_use.is_invalid() || next_use.index() < ((1 << 24) - 1));
        let next_use = (next_use.0 as usize) & 0x00ff_ffff;
        Self {
            operand,
            pos,
            next_use_and_slot: (next_use as u32) | ((slot as u32) << 24),
        }
    }
    #[inline(always)]
    fn next_use(&self) -> UseIndex {
        let val = self.next_use_and_slot & 0x00ff_ffff;
        // Sign-extend 0x00ff_ffff to INVALID (0xffff_ffff).
        let val = ((val as i32) << 8) >> 8;
        UseIndex::new(val as usize)
    }
    #[inline(always)]
    fn slot(&self) -> u8 {
        (self.next_use_and_slot >> 24) as u8
    }
    #[inline(always)]
    fn set_next_use(&mut self, u: UseIndex) {
        debug_assert!(u.is_invalid() || u.index() < ((1 << 24) - 1));
        let u = (u.0 as usize) & 0x00ff_ffff;
        self.next_use_and_slot = (self.next_use_and_slot & 0xff00_0000) | (u as u32);
    }
}

const SLOT_NONE: u8 = u8::MAX;

#[derive(Clone, Debug)]
struct LiveBundle {
    first_range: LiveRangeIndex,
    last_range: LiveRangeIndex,
    spillset: SpillSetIndex,
    allocation: Allocation,
    prio: u32, // recomputed after every bulk update
    spill_weight_and_props: u32,
    range_summary: RangeSummary,
}

impl LiveBundle {
    #[inline(always)]
    fn set_cached_spill_weight_and_props(&mut self, spill_weight: u32, minimal: bool, fixed: bool) {
        debug_assert!(spill_weight < ((1 << 30) - 1));
        self.spill_weight_and_props =
            spill_weight | (if minimal { 1 << 31 } else { 0 }) | (if fixed { 1 << 30 } else { 0 });
    }

    #[inline(always)]
    fn cached_minimal(&self) -> bool {
        self.spill_weight_and_props & (1 << 31) != 0
    }

    #[inline(always)]
    fn cached_fixed(&self) -> bool {
        self.spill_weight_and_props & (1 << 30) != 0
    }

    #[inline(always)]
    fn cached_spill_weight(&self) -> u32 {
        self.spill_weight_and_props & ((1 << 30) - 1)
    }
}

#[derive(Clone, Debug)]
struct RangeSummary {
    /// Indices in `range_ranges` dense array of packed CodeRange structs.
    from: u32,
    to: u32,
    bound: CodeRange,
}

impl RangeSummary {
    fn new() -> Self {
        Self {
            from: 0,
            to: 0,
            bound: CodeRange {
                from: ProgPoint::from_index(0),
                to: ProgPoint::from_index(0),
            },
        }
    }

    fn iter<'a>(&'a self, range_array: &'a [CodeRange]) -> RangeSummaryIter<'a> {
        RangeSummaryIter {
            idx: self.from as usize,
            start: self.from as usize,
            limit: self.to as usize,
            bound: self.bound,
            arr: range_array,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RangeSummaryIter<'a> {
    idx: usize,
    start: usize,
    limit: usize,
    bound: CodeRange,
    arr: &'a [CodeRange],
}

impl<'a> std::iter::Iterator for RangeSummaryIter<'a> {
    type Item = CodeRange;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.limit {
            return None;
        }
        while self.idx < self.limit && self.arr[self.idx].to <= self.bound.from {
            self.idx += 1;
        }
        let mut cur = self.arr[self.idx];
        if cur.from >= self.bound.to {
            self.idx = self.limit;
            return None;
        }

        if cur.from < self.bound.from {
            cur.from = self.bound.from;
        }
        if cur.to > self.bound.to {
            cur.to = self.bound.to;
        }

        self.idx += 1;
        Some(cur)
    }
}

#[derive(Clone, Debug)]
struct SpillSet {
    bundles: SmallVec<[LiveBundleIndex; 2]>,
    slot: SpillSlotIndex,
    reg_hint: PReg,
    class: RegClass,
    size: u8,
}

#[derive(Clone, Debug)]
struct VRegData {
    blockparam: Block,
    first_range: LiveRangeIndex,
    is_ref: bool,
}

#[derive(Clone, Debug)]
struct PRegData {
    reg: PReg,
    allocations: LiveRangeSet,
}

/*
 * Environment setup:
 *
 * We have seven fundamental objects: LiveRange, LiveBundle, SpillSet, Use, VReg, PReg.
 *
 * The relationship is as follows:
 *
 * LiveRange --(vreg)--> shared(VReg)
 * LiveRange --(bundle)--> shared(LiveBundle)
 * LiveRange --(use) --> list(Use)
 *
 * Use --(vreg)--> shared(VReg)
 *
 * LiveBundle --(range)--> list(LiveRange)
 * LiveBundle --(spillset)--> shared(SpillSet)
 * LiveBundle --(parent)--> parent(LiveBundle)
 *
 * SpillSet --(parent)--> parent(SpillSet)
 * SpillSet --(bundles)--> list(LiveBundle)
 *
 * VReg --(range)--> list(LiveRange)
 *
 * PReg --(ranges)--> set(LiveRange)
 */

#[derive(Clone, Debug)]
struct Env<'a, F: Function> {
    func: &'a F,
    env: &'a MachineEnv,
    cfginfo: CFGInfo,
    liveins: Vec<BitVec>,
    liveouts: Vec<BitVec>,
    /// Blockparam outputs: from-vreg, (end of) from-block, (start of)
    /// to-block, to-vreg. The field order is significant: these are sorted so
    /// that a scan over vregs, then blocks in each range, can scan in
    /// order through this (sorted) list and add allocs to the
    /// half-move list.
    blockparam_outs: Vec<(VRegIndex, Block, Block, VRegIndex)>,
    /// Blockparam inputs: to-vreg, (start of) to-block, (end of)
    /// from-block. As above for `blockparam_outs`, field order is
    /// significant.
    blockparam_ins: Vec<(VRegIndex, Block, Block)>,
    /// Blockparam allocs: block, idx, vreg, alloc. Info to describe
    /// blockparam locations at block entry, for metadata purposes
    /// (e.g. for the checker).
    blockparam_allocs: Vec<(Block, u32, VRegIndex, Allocation)>,

    ranges: Vec<LiveRange>,
    ranges_hot: Vec<LiveRangeHot>,
    range_ranges: Vec<CodeRange>,
    bundles: Vec<LiveBundle>,
    spillsets: Vec<SpillSet>,
    uses: Vec<Use>,
    vregs: Vec<VRegData>,
    vreg_regs: Vec<VReg>,
    pregs: Vec<PRegData>,
    allocation_queue: PrioQueue,
    hot_code: LiveRangeSet,
    clobbers: Vec<Inst>,   // Sorted list of insts with clobbers.
    safepoints: Vec<Inst>, // Sorted list of safepoint insts.
    safepoints_per_vreg: HashMap<usize, HashSet<Inst>>,

    spilled_bundles: Vec<LiveBundleIndex>,
    spillslots: Vec<SpillSlotData>,
    slots_by_size: Vec<SpillSlotList>,

    // Program moves: these are moves in the provided program that we
    // handle with our internal machinery, in order to avoid the
    // overhead of ordinary operand processing. We expect the client
    // to not generate any code for instructions that return
    // `Some(..)` for `.is_move()`, and instead use the edits that we
    // provide to implement those moves (or some simplified version of
    // them) post-regalloc.
    //
    // (from-vreg, inst, from-alloc), sorted by (from-vreg, inst)
    prog_move_srcs: Vec<((VRegIndex, Inst), Allocation)>,
    // (to-vreg, inst, to-alloc), sorted by (to-vreg, inst)
    prog_move_dsts: Vec<((VRegIndex, Inst), Allocation)>,
    // (from-vreg, to-vreg) for bundle-merging.
    prog_move_merges: Vec<(LiveRangeIndex, LiveRangeIndex)>,

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
    multi_fixed_reg_fixups: Vec<(ProgPoint, PRegIndex, PRegIndex, usize)>,

    inserted_moves: Vec<InsertedMove>,

    // Output:
    edits: Vec<(u32, InsertMovePrio, Edit)>,
    allocs: Vec<Allocation>,
    inst_alloc_offsets: Vec<u32>,
    num_spillslots: u32,
    safepoint_slots: Vec<(ProgPoint, SpillSlot)>,

    stats: Stats,

    // For debug output only: a list of textual annotations at every
    // ProgPoint to insert into the final allocated program listing.
    debug_annotations: std::collections::HashMap<ProgPoint, Vec<String>>,
}

#[derive(Clone, Debug)]
struct SpillSlotData {
    ranges: LiveRangeSet,
    class: RegClass,
    size: u32,
    alloc: Allocation,
    next_spillslot: SpillSlotIndex,
}

#[derive(Clone, Debug)]
struct SpillSlotList {
    first_spillslot: SpillSlotIndex,
    last_spillslot: SpillSlotIndex,
}

#[derive(Clone, Debug)]
struct PrioQueue {
    heap: std::collections::BinaryHeap<PrioQueueEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PrioQueueEntry {
    prio: u32,
    bundle: LiveBundleIndex,
}

#[derive(Clone, Debug)]
struct LiveRangeSet {
    btree: BTreeMap<LiveRangeKey, LiveRangeIndex>,
}

#[derive(Clone, Copy, Debug)]
struct LiveRangeKey {
    from: u32,
    to: u32,
}

impl LiveRangeKey {
    #[inline(always)]
    fn from_range(range: &CodeRange) -> Self {
        Self {
            from: range.from.to_index(),
            to: range.to.to_index(),
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

struct PrioQueueComparator<'a> {
    prios: &'a [usize],
}
impl<'a> ContainerComparator for PrioQueueComparator<'a> {
    type Ix = LiveBundleIndex;
    fn compare(&self, a: Self::Ix, b: Self::Ix) -> std::cmp::Ordering {
        self.prios[a.index()].cmp(&self.prios[b.index()])
    }
}

impl PrioQueue {
    fn new() -> Self {
        PrioQueue {
            heap: std::collections::BinaryHeap::new(),
        }
    }

    fn insert(&mut self, bundle: LiveBundleIndex, prio: usize) {
        self.heap.push(PrioQueueEntry {
            prio: prio as u32,
            bundle,
        });
    }

    fn is_empty(self) -> bool {
        self.heap.is_empty()
    }

    fn pop(&mut self) -> Option<LiveBundleIndex> {
        self.heap.pop().map(|entry| entry.bundle)
    }
}

impl LiveRangeSet {
    pub(crate) fn new() -> Self {
        Self {
            btree: BTreeMap::new(),
        }
    }
}

#[inline(always)]
fn spill_weight_from_policy(policy: OperandPolicy) -> u32 {
    match policy {
        OperandPolicy::Any => 1000,
        OperandPolicy::Reg | OperandPolicy::FixedReg(_) => 2000,
        _ => 0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Requirement {
    Fixed(PReg),
    Register(RegClass),
    Stack(RegClass),
    Any(RegClass),
}
impl Requirement {
    #[inline(always)]
    fn class(self) -> RegClass {
        match self {
            Requirement::Fixed(preg) => preg.class(),
            Requirement::Register(class) | Requirement::Any(class) | Requirement::Stack(class) => {
                class
            }
        }
    }
    #[inline(always)]
    fn merge(self, other: Requirement) -> Option<Requirement> {
        if self.class() != other.class() {
            return None;
        }
        match (self, other) {
            (other, Requirement::Any(_)) | (Requirement::Any(_), other) => Some(other),
            (Requirement::Stack(_), Requirement::Stack(_)) => Some(self),
            (Requirement::Register(_), Requirement::Fixed(preg))
            | (Requirement::Fixed(preg), Requirement::Register(_)) => {
                Some(Requirement::Fixed(preg))
            }
            (Requirement::Register(_), Requirement::Register(_)) => Some(self),
            (Requirement::Fixed(a), Requirement::Fixed(b)) if a == b => Some(self),
            _ => None,
        }
    }
    #[inline(always)]
    fn from_operand(op: Operand) -> Requirement {
        match op.policy() {
            OperandPolicy::FixedReg(preg) => Requirement::Fixed(preg),
            OperandPolicy::Reg | OperandPolicy::Reuse(_) => Requirement::Register(op.class()),
            OperandPolicy::Stack => Requirement::Stack(op.class()),
            _ => Requirement::Any(op.class()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AllocRegResult {
    Allocated(Allocation),
    Conflict(LiveBundleVec),
    ConflictWithFixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BundleProperties {
    minimal: bool,
    fixed: bool,
}

#[derive(Clone, Debug)]
struct InsertedMove {
    pos: ProgPoint,
    prio: InsertMovePrio,
    from_alloc: Allocation,
    to_alloc: Allocation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum InsertMovePrio {
    InEdgeMoves,
    BlockParam,
    Regular,
    MultiFixedReg,
    ReusedInput,
    OutEdgeMoves,
    ProgramMove,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    livein_blocks: usize,
    livein_iterations: usize,
    initial_liverange_count: usize,
    merged_bundle_count: usize,
    process_bundle_count: usize,
    process_bundle_reg_probes_fixed: usize,
    process_bundle_reg_success_fixed: usize,
    process_bundle_bounding_range_probe_start_any: usize,
    process_bundle_bounding_range_probes_any: usize,
    process_bundle_bounding_range_success_any: usize,
    process_bundle_reg_probe_start_any: usize,
    process_bundle_reg_probes_any: usize,
    process_bundle_reg_success_any: usize,
    evict_bundle_event: usize,
    evict_bundle_count: usize,
    splits: usize,
    splits_clobbers: usize,
    splits_hot: usize,
    splits_conflicts: usize,
    splits_all: usize,
    final_liverange_count: usize,
    final_bundle_count: usize,
    spill_bundle_count: usize,
    spill_bundle_reg_probes: usize,
    spill_bundle_reg_success: usize,
    blockparam_ins_count: usize,
    blockparam_outs_count: usize,
    blockparam_allocs_count: usize,
    halfmoves_count: usize,
    edits_count: usize,
}

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
struct RegTraversalIter<'a> {
    env: &'a MachineEnv,
    class: usize,
    hints: [Option<PReg>; 2],
    hint_idx: usize,
    pref_idx: usize,
    non_pref_idx: usize,
    offset: usize,
}

impl<'a> RegTraversalIter<'a> {
    pub fn new(
        env: &'a MachineEnv,
        class: RegClass,
        hint_reg: PReg,
        hint2_reg: PReg,
        offset: usize,
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
        Self {
            env,
            class: class as u8 as usize,
            hints,
            hint_idx: 0,
            pref_idx: 0,
            non_pref_idx: 0,
            offset,
        }
    }
}

impl<'a> std::iter::Iterator for RegTraversalIter<'a> {
    type Item = PReg;

    fn next(&mut self) -> Option<PReg> {
        if self.hint_idx < 2 && self.hints[self.hint_idx].is_some() {
            let h = self.hints[self.hint_idx];
            self.hint_idx += 1;
            return h;
        }
        while self.pref_idx < self.env.preferred_regs_by_class[self.class].len() {
            let arr = &self.env.preferred_regs_by_class[self.class][..];
            let r = arr[(self.pref_idx + self.offset) % arr.len()];
            self.pref_idx += 1;
            if Some(r) == self.hints[0] || Some(r) == self.hints[1] {
                continue;
            }
            return Some(r);
        }
        while self.non_pref_idx < self.env.non_preferred_regs_by_class[self.class].len() {
            let arr = &self.env.non_preferred_regs_by_class[self.class][..];
            let r = arr[(self.non_pref_idx + self.offset) % arr.len()];
            self.non_pref_idx += 1;
            if Some(r) == self.hints[0] || Some(r) == self.hints[1] {
                continue;
            }
            return Some(r);
        }
        None
    }
}

impl<'a, F: Function> Env<'a, F> {
    pub(crate) fn new(func: &'a F, env: &'a MachineEnv, cfginfo: CFGInfo) -> Self {
        let n = func.insts();
        Self {
            func,
            env,
            cfginfo,

            liveins: Vec::with_capacity(func.blocks()),
            liveouts: Vec::with_capacity(func.blocks()),
            blockparam_outs: vec![],
            blockparam_ins: vec![],
            blockparam_allocs: vec![],
            bundles: Vec::with_capacity(n),
            ranges: Vec::with_capacity(4 * n),
            ranges_hot: Vec::with_capacity(4 * n),
            range_ranges: Vec::with_capacity(4 * n),
            spillsets: Vec::with_capacity(n),
            uses: Vec::with_capacity(4 * n),
            vregs: Vec::with_capacity(n),
            vreg_regs: Vec::with_capacity(n),
            pregs: vec![],
            allocation_queue: PrioQueue::new(),
            clobbers: vec![],
            safepoints: vec![],
            safepoints_per_vreg: HashMap::new(),
            hot_code: LiveRangeSet::new(),
            spilled_bundles: vec![],
            spillslots: vec![],
            slots_by_size: vec![],

            prog_move_srcs: Vec::with_capacity(n / 2),
            prog_move_dsts: Vec::with_capacity(n / 2),
            prog_move_merges: Vec::with_capacity(n / 2),

            multi_fixed_reg_fixups: vec![],
            inserted_moves: vec![],
            edits: Vec::with_capacity(n),
            allocs: Vec::with_capacity(4 * n),
            inst_alloc_offsets: vec![],
            num_spillslots: 0,
            safepoint_slots: vec![],

            stats: Stats::default(),

            debug_annotations: std::collections::HashMap::new(),
        }
    }

    fn create_pregs_and_vregs(&mut self) {
        // Create PRegs from the env.
        self.pregs.resize(
            PReg::MAX_INDEX,
            PRegData {
                reg: PReg::invalid(),
                allocations: LiveRangeSet::new(),
            },
        );
        for &preg in &self.env.regs {
            self.pregs[preg.index()].reg = preg;
        }
        // Create VRegs from the vreg count.
        for idx in 0..self.func.num_vregs() {
            // We'll fill in the real details when we see the def.
            let reg = VReg::new(idx, RegClass::Int);
            self.add_vreg(
                reg,
                VRegData {
                    first_range: LiveRangeIndex::invalid(),
                    blockparam: Block::invalid(),
                    is_ref: false,
                },
            );
        }
        for v in self.func.reftype_vregs() {
            self.vregs[v.vreg()].is_ref = true;
        }
        // Create allocations too.
        for inst in 0..self.func.insts() {
            let start = self.allocs.len() as u32;
            self.inst_alloc_offsets.push(start);
            for _ in 0..self.func.inst_operands(Inst::new(inst)).len() {
                self.allocs.push(Allocation::none());
            }
        }
    }

    fn add_vreg(&mut self, reg: VReg, data: VRegData) -> VRegIndex {
        let idx = self.vregs.len();
        self.vregs.push(data);
        self.vreg_regs.push(reg);
        VRegIndex::new(idx)
    }

    fn create_liverange(&mut self, range: CodeRange) -> LiveRangeIndex {
        let idx = self.ranges.len();
        self.ranges_hot.push(LiveRangeHot {
            range,
            next_in_bundle: LiveRangeIndex::invalid(),
        });

        self.ranges.push(LiveRange {
            vreg: VRegIndex::invalid(),
            bundle: LiveBundleIndex::invalid(),
            uses_spill_weight_and_flags: 0,

            first_use: UseIndex::invalid(),
            last_use: UseIndex::invalid(),

            next_in_reg: LiveRangeIndex::invalid(),

            merged_into: LiveRangeIndex::invalid(),
        });

        LiveRangeIndex::new(idx)
    }

    /// Mark `range` as live for the given `vreg`. `num_ranges` is used to prevent
    /// excessive coalescing on pathological inputs.
    ///
    /// Returns the liverange that contains the given range.
    fn add_liverange_to_vreg(
        &mut self,
        vreg: VRegIndex,
        range: CodeRange,
        num_ranges: &mut usize,
    ) -> LiveRangeIndex {
        log::debug!("add_liverange_to_vreg: vreg {:?} range {:?}", vreg, range);
        const COALESCE_LIMIT: usize = 100_000;

        // Look for a single or contiguous sequence of existing live ranges that overlap with the
        // given range.

        let mut insert_after = LiveRangeIndex::invalid();
        let mut merged = LiveRangeIndex::invalid();
        let mut iter = self.vregs[vreg.index()].first_range;
        let mut prev = LiveRangeIndex::invalid();
        while iter.is_valid() {
            log::debug!(" -> existing range: {:?}", self.ranges[iter.index()]);
            if range.from >= self.ranges_hot[iter.index()].range.to && *num_ranges < COALESCE_LIMIT
            {
                // New range comes fully after this one -- record it as a lower bound.
                insert_after = iter;
                prev = iter;
                iter = self.ranges[iter.index()].next_in_reg;
                log::debug!("    -> lower bound");
                continue;
            }
            if range.to <= self.ranges_hot[iter.index()].range.from {
                // New range comes fully before this one -- we're found our spot.
                log::debug!("    -> upper bound (break search loop)");
                break;
            }
            // If we're here, then we overlap with at least one endpoint of the range.
            log::debug!("    -> must overlap");
            debug_assert!(range.overlaps(&self.ranges_hot[iter.index()].range));
            if merged.is_invalid() {
                // This is the first overlapping range. Extend to simply cover the new range.
                merged = iter;
                if range.from < self.ranges_hot[iter.index()].range.from {
                    self.ranges_hot[iter.index()].range.from = range.from;
                }
                if range.to > self.ranges_hot[iter.index()].range.to {
                    self.ranges_hot[iter.index()].range.to = range.to;
                }
                log::debug!(
                    "    -> extended range of existing range to {:?}",
                    self.ranges_hot[iter.index()].range
                );
                // Continue; there may be more ranges to merge with.
                prev = iter;
                iter = self.ranges[iter.index()].next_in_reg;
                continue;
            }
            // We overlap but we've already extended the first overlapping existing liverange, so
            // we need to do a true merge instead.
            log::debug!("    -> merging {:?} into {:?}", iter, merged);
            log::debug!(
                "    -> before: merged {:?}: {:?}",
                merged,
                self.ranges[merged.index()]
            );
            debug_assert!(
                self.ranges_hot[iter.index()].range.from
                    >= self.ranges_hot[merged.index()].range.from
            ); // Because we see LRs in order.
            if self.ranges_hot[iter.index()].range.to > self.ranges_hot[merged.index()].range.to {
                self.ranges_hot[merged.index()].range.to = self.ranges_hot[iter.index()].range.to;
            }
            self.distribute_liverange_uses(iter, merged);
            log::debug!(
                "    -> after: merged {:?}: {:?}",
                merged,
                self.ranges[merged.index()]
            );

            // Remove from list of liveranges for this vreg.
            let next = self.ranges[iter.index()].next_in_reg;
            if prev.is_valid() {
                self.ranges[prev.index()].next_in_reg = next;
            } else {
                self.vregs[vreg.index()].first_range = next;
            }
            // `prev` remains the same (we deleted current range).
            iter = next;
        }

        // If we get here and did not merge into an existing liverange or liveranges, then we need
        // to create a new one.
        if merged.is_invalid() {
            let lr = self.create_liverange(range);
            self.ranges[lr.index()].vreg = vreg;
            if insert_after.is_valid() {
                let next = self.ranges[insert_after.index()].next_in_reg;
                self.ranges[lr.index()].next_in_reg = next;
                self.ranges[insert_after.index()].next_in_reg = lr;
            } else {
                self.ranges[lr.index()].next_in_reg = self.vregs[vreg.index()].first_range;
                self.vregs[vreg.index()].first_range = lr;
            }
            *num_ranges += 1;
            lr
        } else {
            merged
        }
    }

    fn distribute_liverange_uses(&mut self, from: LiveRangeIndex, into: LiveRangeIndex) {
        log::debug!("distribute from {:?} to {:?}", from, into);
        assert_eq!(
            self.ranges[from.index()].vreg,
            self.ranges[into.index()].vreg
        );
        let into_range = self.ranges_hot[into.index()].range;
        // For every use in `from`...
        let mut prev = UseIndex::invalid();
        let mut iter = self.ranges[from.index()].first_use;
        while iter.is_valid() {
            let usedata = &mut self.uses[iter.index()];
            // If we have already passed `into`, we're done.
            if usedata.pos >= into_range.to {
                break;
            }
            // If this use is within the range of `into`, move it over.
            if into_range.contains_point(usedata.pos) {
                log::debug!(" -> moving {:?}", iter);
                let next = usedata.next_use();
                if prev.is_valid() {
                    self.uses[prev.index()].set_next_use(next);
                } else {
                    self.ranges[from.index()].first_use = next;
                }
                if iter == self.ranges[from.index()].last_use {
                    self.ranges[from.index()].last_use = prev;
                }
                // `prev` remains the same.
                self.update_liverange_stats_on_remove_use(from, iter);
                // This may look inefficient but because we are always merging
                // non-overlapping LiveRanges, all uses will be at the beginning
                // or end of the existing use-list; both cases are optimized.
                self.insert_use_into_liverange_and_update_stats(into, iter);
                iter = next;
            } else {
                prev = iter;
                iter = usedata.next_use();
            }
        }
        self.ranges[from.index()].merged_into = into;
    }

    fn update_liverange_stats_on_remove_use(&mut self, from: LiveRangeIndex, u: UseIndex) {
        log::debug!("remove use {:?} from lr {:?}", u, from);
        debug_assert!(u.is_valid());
        let usedata = &self.uses[u.index()];
        let lrdata = &mut self.ranges[from.index()];
        log::debug!(
            "  -> subtract {} from uses_spill_weight {}; now {}",
            spill_weight_from_policy(usedata.operand.policy()),
            lrdata.uses_spill_weight(),
            lrdata.uses_spill_weight() - spill_weight_from_policy(usedata.operand.policy()),
        );

        lrdata.uses_spill_weight_and_flags -= spill_weight_from_policy(usedata.operand.policy());
        if usedata.operand.kind() != OperandKind::Use {
            lrdata.uses_spill_weight_and_flags -= 2000;
        }
    }

    fn insert_use_into_liverange_and_update_stats(&mut self, into: LiveRangeIndex, u: UseIndex) {
        let insert_pos = self.uses[u.index()].pos;
        let first = self.ranges[into.index()].first_use;
        self.uses[u.index()].set_next_use(UseIndex::invalid());
        if first.is_invalid() {
            // Empty list.
            self.ranges[into.index()].first_use = u;
            self.ranges[into.index()].last_use = u;
        } else if insert_pos > self.uses[self.ranges[into.index()].last_use.index()].pos {
            // After tail.
            let tail = self.ranges[into.index()].last_use;
            self.uses[tail.index()].set_next_use(u);
            self.ranges[into.index()].last_use = u;
        } else {
            // Otherwise, scan linearly to find insertion position.
            let mut prev = UseIndex::invalid();
            let mut iter = first;
            while iter.is_valid() {
                if self.uses[iter.index()].pos > insert_pos {
                    break;
                }
                prev = iter;
                iter = self.uses[iter.index()].next_use();
            }
            self.uses[u.index()].set_next_use(iter);
            if prev.is_valid() {
                self.uses[prev.index()].set_next_use(u);
            } else {
                self.ranges[into.index()].first_use = u;
            }
            if iter.is_invalid() {
                self.ranges[into.index()].last_use = u;
            }
        }

        // Update stats.
        let policy = self.uses[u.index()].operand.policy();
        log::debug!(
            "insert use {:?} into lr {:?} with weight {}",
            u,
            into,
            spill_weight_from_policy(policy)
        );
        self.ranges[into.index()].uses_spill_weight_and_flags += spill_weight_from_policy(policy);
        if self.uses[u.index()].operand.kind() != OperandKind::Use {
            self.ranges[into.index()].uses_spill_weight_and_flags += 2000;
        }
        log::debug!("  -> now {}", self.ranges[into.index()].uses_spill_weight());
    }

    fn find_vreg_liverange_for_pos(
        &self,
        vreg: VRegIndex,
        pos: ProgPoint,
    ) -> Option<LiveRangeIndex> {
        let mut range = self.vregs[vreg.index()].first_range;
        while range.is_valid() {
            if self.ranges_hot[range.index()].range.contains_point(pos) {
                return Some(range);
            }
            range = self.ranges[range.index()].next_in_reg;
        }
        None
    }

    fn add_liverange_to_preg(&mut self, range: CodeRange, reg: PReg) {
        log::debug!("adding liverange to preg: {:?} to {}", range, reg);
        let preg_idx = PRegIndex::new(reg.index());
        let lr = self.create_liverange(range);
        self.pregs[preg_idx.index()]
            .allocations
            .btree
            .insert(LiveRangeKey::from_range(&range), lr);
    }

    fn is_live_in(&mut self, block: Block, vreg: VRegIndex) -> bool {
        self.liveins[block.index()].get(vreg.index())
    }

    fn compute_liveness(&mut self) {
        // Create initial LiveIn and LiveOut bitsets.
        for _ in 0..self.func.blocks() {
            self.liveins.push(BitVec::new());
            self.liveouts.push(BitVec::new());
        }

        // Run a worklist algorithm to precisely compute liveins and
        // liveouts.
        let mut workqueue = VecDeque::new();
        let mut workqueue_set = FxHashSet::default();
        // Initialize workqueue with postorder traversal.
        for &block in &self.cfginfo.postorder[..] {
            workqueue.push_back(block);
            workqueue_set.insert(block);
        }

        while !workqueue.is_empty() {
            let block = workqueue.pop_front().unwrap();
            workqueue_set.remove(&block);

            log::debug!("computing liveins for block{}", block.index());

            self.stats.livein_iterations += 1;

            let mut live = self.liveouts[block.index()].clone();
            for inst in self.func.block_insns(block).rev().iter() {
                if let Some((src, dst)) = self.func.is_move(inst) {
                    live.set(dst.vreg(), false);
                    live.set(src.vreg(), true);
                }
                for pos in &[OperandPos::After, OperandPos::Before] {
                    for op in self.func.inst_operands(inst) {
                        if op.pos() == *pos {
                            match op.kind() {
                                OperandKind::Use | OperandKind::Mod => {
                                    live.set(op.vreg().vreg(), true);
                                }
                                OperandKind::Def => {
                                    live.set(op.vreg().vreg(), false);
                                }
                            }
                        }
                    }
                }
            }
            for &blockparam in self.func.block_params(block) {
                live.set(blockparam.vreg(), false);
            }

            for &pred in self.func.block_preds(block) {
                if self.liveouts[pred.index()].or(&live) {
                    if !workqueue_set.contains(&pred) {
                        workqueue_set.insert(pred);
                        workqueue.push_back(pred);
                    }
                }
            }

            log::debug!("computed liveins at block{}: {:?}", block.index(), live);
            self.liveins[block.index()] = live;
        }

        let mut num_ranges = 0;

        for &vreg in self.func.reftype_vregs() {
            self.safepoints_per_vreg.insert(vreg.vreg(), HashSet::new());
        }

        // Create Uses and Defs referring to VRegs, and place the Uses
        // in LiveRanges.
        //
        // We already computed precise liveouts and liveins for every
        // block above, so we don't need to run an iterative algorithm
        // here; instead, every block's computation is purely local,
        // from end to start.

        // Track current LiveRange for each vreg.
        //
        // Invariant: a stale range may be present here; ranges are
        // only valid if `live.get(vreg)` is true.
        let mut vreg_ranges: Vec<LiveRangeIndex> =
            vec![LiveRangeIndex::invalid(); self.func.num_vregs()];

        for i in (0..self.func.blocks()).rev() {
            let block = Block::new(i);

            self.stats.livein_blocks += 1;

            // Init our local live-in set.
            let mut live = self.liveouts[block.index()].clone();

            // Initially, registers are assumed live for the whole block.
            for vreg in live.iter() {
                let range = CodeRange {
                    from: self.cfginfo.block_entry[block.index()],
                    to: self.cfginfo.block_exit[block.index()].next(),
                };
                log::debug!(
                    "vreg {:?} live at end of block --> create range {:?}",
                    VRegIndex::new(vreg),
                    range
                );
                let lr = self.add_liverange_to_vreg(VRegIndex::new(vreg), range, &mut num_ranges);
                vreg_ranges[vreg] = lr;
            }

            // Create vreg data for blockparams.
            for param in self.func.block_params(block) {
                self.vreg_regs[param.vreg()] = *param;
                self.vregs[param.vreg()].blockparam = block;
            }

            let insns = self.func.block_insns(block);

            // If the last instruction is a branch (rather than
            // return), create blockparam_out entries.
            if self.func.is_branch(insns.last()) {
                let operands = self.func.inst_operands(insns.last());
                let mut i = self.func.branch_blockparam_arg_offset(block, insns.last());
                for &succ in self.func.block_succs(block) {
                    for &blockparam in self.func.block_params(succ) {
                        let from_vreg = VRegIndex::new(operands[i].vreg().vreg());
                        let blockparam_vreg = VRegIndex::new(blockparam.vreg());
                        self.blockparam_outs
                            .push((from_vreg, block, succ, blockparam_vreg));
                        i += 1;
                    }
                }
            }

            // For each instruction, in reverse order, process
            // operands and clobbers.
            for inst in insns.rev().iter() {
                if self.func.inst_clobbers(inst).len() > 0 {
                    self.clobbers.push(inst);
                }

                // Mark clobbers with CodeRanges on PRegs.
                for i in 0..self.func.inst_clobbers(inst).len() {
                    // don't borrow `self`
                    let clobber = self.func.inst_clobbers(inst)[i];
                    // Clobber range is at After point only: an
                    // instruction can still take an input in a reg
                    // that it later clobbers. (In other words, the
                    // clobber is like a normal def that never gets
                    // used.)
                    let range = CodeRange {
                        from: ProgPoint::after(inst),
                        to: ProgPoint::before(inst.next()),
                    };
                    self.add_liverange_to_preg(range, clobber);
                }

                // Does the instruction have any input-reusing
                // outputs? This is important below to establish
                // proper interference wrt other inputs.
                let mut reused_input = None;
                for op in self.func.inst_operands(inst) {
                    if let OperandPolicy::Reuse(i) = op.policy() {
                        reused_input = Some(i);
                        break;
                    }
                }

                // If this is a move, handle specially.
                if let Some((src, dst)) = self.func.is_move(inst) {
                    if src != dst {
                        log::debug!(" -> move inst{}: src {} -> dst {}", inst.index(), src, dst);
                        assert_eq!(src.class(), dst.class());

                        // Handle the def w.r.t. liveranges: trim the
                        // start of the range and mark it dead at this
                        // point in our backward scan.
                        let pos = ProgPoint::before(inst); // See note below re: pos of use.
                        let mut dst_lr = vreg_ranges[dst.vreg()];
                        // If there was no liverange (dead def), create a trivial one.
                        if !live.get(dst.vreg()) {
                            dst_lr = self.add_liverange_to_vreg(
                                VRegIndex::new(dst.vreg()),
                                CodeRange {
                                    from: pos,
                                    to: pos.next().next(),
                                },
                                &mut num_ranges,
                            );
                            log::debug!(" -> invalid; created {:?}", dst_lr);
                        } else {
                            log::debug!(" -> has existing LR {:?}", dst_lr);
                        }
                        if self.ranges_hot[dst_lr.index()].range.from
                            == self.cfginfo.block_entry[block.index()]
                        {
                            log::debug!(" -> started at block start; trimming to {:?}", pos);
                            self.ranges_hot[dst_lr.index()].range.from = pos;
                        }
                        live.set(dst.vreg(), false);
                        vreg_ranges[dst.vreg()] = LiveRangeIndex::invalid();
                        self.vreg_regs[dst.vreg()] = dst;

                        // Handle the use w.r.t. liveranges: make it live
                        // and create an initial LR back to the start of
                        // the block.
                        let pos = ProgPoint::before(inst);
                        let range = CodeRange {
                            from: self.cfginfo.block_entry[block.index()],
                            // Live up to end of previous inst. Because
                            // the move isn't actually reading the
                            // value as part of the inst, all we need
                            // to do is to decide where to join the
                            // LRs; and we want this to be at an inst
                            // boundary, not in the middle, so that
                            // the move-insertion logic remains happy.
                            to: pos,
                        };
                        let src_lr = self.add_liverange_to_vreg(
                            VRegIndex::new(src.vreg()),
                            range,
                            &mut num_ranges,
                        );
                        vreg_ranges[src.vreg()] = src_lr;

                        log::debug!(" -> src LR {:?}", src_lr);

                        // Add to live-set.
                        let src_is_dead_after_move = !live.get(src.vreg());
                        live.set(src.vreg(), true);

                        // Add to program-moves lists.
                        self.prog_move_srcs
                            .push(((VRegIndex::new(src.vreg()), inst), Allocation::none()));
                        self.prog_move_dsts
                            .push(((VRegIndex::new(dst.vreg()), inst), Allocation::none()));
                        if src_is_dead_after_move {
                            self.prog_move_merges.push((src_lr, dst_lr));
                        }

                        continue;
                    }
                }

                // Process defs and uses.
                for &cur_pos in &[InstPosition::After, InstPosition::Before] {
                    for i in 0..self.func.inst_operands(inst).len() {
                        // don't borrow `self`
                        let operand = self.func.inst_operands(inst)[i];
                        let pos = match (operand.kind(), operand.pos()) {
                            (OperandKind::Mod, _) => ProgPoint::before(inst),
                            (OperandKind::Def, OperandPos::Before) => ProgPoint::before(inst),
                            (OperandKind::Def, OperandPos::After) => ProgPoint::after(inst),
                            (OperandKind::Use, OperandPos::After) => ProgPoint::after(inst),
                            // If this is a branch, extend `pos` to
                            // the end of the block. (Branch uses are
                            // blockparams and need to be live at the
                            // end of the block.)
                            (OperandKind::Use, _) if self.func.is_branch(inst) => {
                                self.cfginfo.block_exit[block.index()]
                            }
                            // If there are any reused inputs in this
                            // instruction, and this is *not* the
                            // reused input, force `pos` to
                            // `After`. (See note below for why; it's
                            // very subtle!)
                            (OperandKind::Use, OperandPos::Before)
                                if reused_input.is_some() && reused_input.unwrap() != i =>
                            {
                                ProgPoint::after(inst)
                            }
                            (OperandKind::Use, OperandPos::Before) => ProgPoint::before(inst),
                        };

                        if pos.pos() != cur_pos {
                            continue;
                        }

                        log::debug!(
                            "processing inst{} operand at {:?}: {:?}",
                            inst.index(),
                            pos,
                            operand
                        );

                        match operand.kind() {
                            OperandKind::Def | OperandKind::Mod => {
                                // Create the use object.
                                let u = UseIndex::new(self.uses.len());
                                self.uses.push(Use::new(
                                    operand,
                                    pos,
                                    UseIndex::invalid(),
                                    i as u8,
                                ));

                                log::debug!("Def of {} at {:?}", operand.vreg(), pos);

                                // Fill in vreg's actual data.
                                self.vreg_regs[operand.vreg().vreg()] = operand.vreg();

                                let mut lr = vreg_ranges[operand.vreg().vreg()];
                                log::debug!(" -> has existing LR {:?}", lr);
                                // If there was no liverange (dead def), create a trivial one.
                                if !live.get(operand.vreg().vreg()) {
                                    let from = match operand.kind() {
                                        OperandKind::Def => pos,
                                        OperandKind::Mod => self.cfginfo.block_entry[block.index()],
                                        _ => unreachable!(),
                                    };
                                    let to = match operand.kind() {
                                        OperandKind::Def => pos.next(),
                                        OperandKind::Mod => pos.next().next(), // both Before and After positions
                                        _ => unreachable!(),
                                    };
                                    lr = self.add_liverange_to_vreg(
                                        VRegIndex::new(operand.vreg().vreg()),
                                        CodeRange { from, to },
                                        &mut num_ranges,
                                    );
                                    log::debug!(" -> invalid; created {:?}", lr);
                                }
                                self.insert_use_into_liverange_and_update_stats(lr, u);

                                if operand.kind() == OperandKind::Def {
                                    // Trim the range for this vreg to start
                                    // at `pos` if it previously ended at the
                                    // start of this block (i.e. was not
                                    // merged into some larger LiveRange due
                                    // to out-of-order blocks).
                                    if self.ranges_hot[lr.index()].range.from
                                        == self.cfginfo.block_entry[block.index()]
                                    {
                                        log::debug!(
                                            " -> started at block start; trimming to {:?}",
                                            pos
                                        );
                                        self.ranges_hot[lr.index()].range.from = pos;
                                    }

                                    // Remove from live-set.
                                    live.set(operand.vreg().vreg(), false);
                                    vreg_ranges[operand.vreg().vreg()] = LiveRangeIndex::invalid();
                                }
                            }
                            OperandKind::Use => {
                                // Create the use object.
                                let u = UseIndex::new(self.uses.len());
                                self.uses.push(Use::new(
                                    operand,
                                    pos,
                                    UseIndex::invalid(),
                                    i as u8,
                                ));

                                // Create/extend the LiveRange if it
                                // doesn't already exist, and add the use
                                // to the range.
                                let mut lr = vreg_ranges[operand.vreg().vreg()];
                                if !live.get(operand.vreg().vreg()) {
                                    let range = CodeRange {
                                        from: self.cfginfo.block_entry[block.index()],
                                        to: pos.next(),
                                    };
                                    lr = self.add_liverange_to_vreg(
                                        VRegIndex::new(operand.vreg().vreg()),
                                        range,
                                        &mut num_ranges,
                                    );
                                    vreg_ranges[operand.vreg().vreg()] = lr;
                                }
                                assert!(lr.is_valid());

                                log::debug!(
                                    "Use of {:?} at {:?} -> {:?} -> {:?}",
                                    operand,
                                    pos,
                                    u,
                                    lr
                                );

                                self.insert_use_into_liverange_and_update_stats(lr, u);

                                // Add to live-set.
                                live.set(operand.vreg().vreg(), true);
                            }
                        }
                    }
                }

                if self.func.is_safepoint(inst) {
                    self.safepoints.push(inst);
                    for vreg in live.iter() {
                        if let Some(safepoints) = self.safepoints_per_vreg.get_mut(&vreg) {
                            safepoints.insert(inst);
                        }
                    }
                }
            }

            // Block parameters define vregs at the very beginning of
            // the block. Remove their live vregs from the live set
            // here.
            for vreg in self.func.block_params(block) {
                if live.get(vreg.vreg()) {
                    live.set(vreg.vreg(), false);
                } else {
                    // Create trivial liverange if blockparam is dead.
                    let start = self.cfginfo.block_entry[block.index()];
                    self.add_liverange_to_vreg(
                        VRegIndex::new(vreg.vreg()),
                        CodeRange {
                            from: start,
                            to: start.next(),
                        },
                        &mut num_ranges,
                    );
                }
                // add `blockparam_ins` entries.
                let vreg_idx = VRegIndex::new(vreg.vreg());
                for &pred in self.func.block_preds(block) {
                    self.blockparam_ins.push((vreg_idx, block, pred));
                }
            }
        }

        self.safepoints.sort_unstable();

        // Insert safepoint virtual stack uses, if needed.
        for vreg in self.func.reftype_vregs() {
            let vreg = VRegIndex::new(vreg.vreg());
            let mut iter = self.vregs[vreg.index()].first_range;
            let mut safepoint_idx = 0;
            while iter.is_valid() {
                let range = self.ranges_hot[iter.index()].range;
                while safepoint_idx < self.safepoints.len()
                    && ProgPoint::before(self.safepoints[safepoint_idx]) < range.from
                {
                    safepoint_idx += 1;
                }
                while safepoint_idx < self.safepoints.len()
                    && range.contains_point(ProgPoint::before(self.safepoints[safepoint_idx]))
                {
                    // Create a virtual use.
                    let pos = ProgPoint::before(self.safepoints[safepoint_idx]);
                    let operand = Operand::new(
                        self.vreg_regs[vreg.index()],
                        OperandPolicy::Stack,
                        OperandKind::Use,
                        OperandPos::Before,
                    );

                    // Create the actual use object.
                    let u = UseIndex::new(self.uses.len());
                    self.uses
                        .push(Use::new(operand, pos, UseIndex::invalid(), SLOT_NONE));

                    // Create/extend the LiveRange and add the use to the range.
                    let range = CodeRange {
                        from: pos,
                        to: pos.next(),
                    };
                    let lr = self.add_liverange_to_vreg(
                        VRegIndex::new(operand.vreg().vreg()),
                        range,
                        &mut num_ranges,
                    );
                    vreg_ranges[operand.vreg().vreg()] = lr;

                    log::debug!(
                        "Safepoint-induced stack use of {:?} at {:?} -> {:?} -> {:?}",
                        operand,
                        pos,
                        u,
                        lr
                    );

                    self.insert_use_into_liverange_and_update_stats(lr, u);
                    safepoint_idx += 1;
                }
                if safepoint_idx >= self.safepoints.len() {
                    break;
                }
                iter = self.ranges[iter.index()].next_in_reg;
            }
        }

        // Do a fixed-reg cleanup pass: if there are any LiveRanges with
        // multiple uses (or defs) at the same ProgPoint and there is
        // more than one FixedReg constraint at that ProgPoint, we
        // need to record all but one of them in a special fixup list
        // and handle them later; otherwise, bundle-splitting to
        // create minimal bundles becomes much more complex (we would
        // have to split the multiple uses at the same progpoint into
        // different bundles, which breaks invariants related to
        // disjoint ranges and bundles).
        let mut seen_fixed_for_vreg: SmallVec<[VReg; 16]> = smallvec![];
        let mut first_preg: SmallVec<[PRegIndex; 16]> = smallvec![];
        let mut extra_clobbers: SmallVec<[(PReg, Inst); 8]> = smallvec![];
        for vreg in 0..self.vregs.len() {
            let mut iter = self.vregs[vreg].first_range;
            while iter.is_valid() {
                log::debug!(
                    "multi-fixed-reg cleanup: vreg {:?} range {:?}",
                    VRegIndex::new(vreg),
                    iter
                );
                let mut last_point = None;
                let mut fixup_multi_fixed_vregs = |pos: ProgPoint,
                                                   slot: usize,
                                                   op: &mut Operand,
                                                   fixups: &mut Vec<(
                    ProgPoint,
                    PRegIndex,
                    PRegIndex,
                    usize,
                )>| {
                    if last_point.is_some() && Some(pos) != last_point {
                        seen_fixed_for_vreg.clear();
                        first_preg.clear();
                    }
                    last_point = Some(pos);

                    if let OperandPolicy::FixedReg(preg) = op.policy() {
                        let vreg_idx = VRegIndex::new(op.vreg().vreg());
                        let preg_idx = PRegIndex::new(preg.index());
                        log::debug!(
                            "at pos {:?}, vreg {:?} has fixed constraint to preg {:?}",
                            pos,
                            vreg_idx,
                            preg_idx
                        );
                        if let Some(idx) = seen_fixed_for_vreg.iter().position(|r| *r == op.vreg())
                        {
                            let orig_preg = first_preg[idx];
                            if orig_preg != preg_idx {
                                log::debug!(" -> duplicate; switching to policy Reg");
                                fixups.push((pos, orig_preg, preg_idx, slot));
                                *op = Operand::new(
                                    op.vreg(),
                                    OperandPolicy::Reg,
                                    op.kind(),
                                    op.pos(),
                                );
                                log::debug!(
                                    " -> extra clobber {} at inst{}",
                                    preg,
                                    pos.inst().index()
                                );
                                extra_clobbers.push((preg, pos.inst()));
                            }
                        } else {
                            seen_fixed_for_vreg.push(op.vreg());
                            first_preg.push(preg_idx);
                        }
                    }
                };

                let mut use_iter = self.ranges[iter.index()].first_use;
                while use_iter.is_valid() {
                    let pos = self.uses[use_iter.index()].pos;
                    let slot = self.uses[use_iter.index()].slot() as usize;
                    fixup_multi_fixed_vregs(
                        pos,
                        slot,
                        &mut self.uses[use_iter.index()].operand,
                        &mut self.multi_fixed_reg_fixups,
                    );
                    use_iter = self.uses[use_iter.index()].next_use();
                }

                for &(clobber, inst) in &extra_clobbers {
                    let range = CodeRange {
                        from: ProgPoint::before(inst),
                        to: ProgPoint::before(inst.next()),
                    };
                    self.add_liverange_to_preg(range, clobber);
                }

                extra_clobbers.clear();
                first_preg.clear();
                seen_fixed_for_vreg.clear();

                iter = self.ranges[iter.index()].next_in_reg;
            }
        }

        self.clobbers.sort_unstable();
        self.blockparam_ins.sort_unstable();
        self.blockparam_outs.sort_unstable();
        self.prog_move_srcs.sort_unstable_by_key(|(pos, _)| *pos);
        self.prog_move_dsts.sort_unstable_by_key(|(pos, _)| *pos);

        log::debug!("prog_move_srcs = {:?}", self.prog_move_srcs);
        log::debug!("prog_move_dsts = {:?}", self.prog_move_dsts);

        self.stats.initial_liverange_count = self.ranges.len();
        self.stats.blockparam_ins_count = self.blockparam_ins.len();
        self.stats.blockparam_outs_count = self.blockparam_outs.len();
    }

    fn compute_hot_code(&mut self) {
        // Initialize hot_code to contain inner loops only.
        let mut header = Block::invalid();
        let mut backedge = Block::invalid();
        for block in 0..self.func.blocks() {
            let block = Block::new(block);
            let max_backedge = self
                .func
                .block_preds(block)
                .iter()
                .filter(|b| b.index() >= block.index())
                .max();
            if let Some(&b) = max_backedge {
                header = block;
                backedge = b;
            }
            if block == backedge {
                // We've traversed a loop body without finding a deeper loop. Mark the whole body
                // as hot.
                let from = self.cfginfo.block_entry[header.index()];
                let to = self.cfginfo.block_exit[backedge.index()].next();
                let range = CodeRange { from, to };
                let lr = self.create_liverange(range);
                self.hot_code
                    .btree
                    .insert(LiveRangeKey::from_range(&range), lr);
            }
        }
    }

    fn create_bundle(&mut self) -> LiveBundleIndex {
        let bundle = self.bundles.len();
        self.bundles.push(LiveBundle {
            allocation: Allocation::none(),
            first_range: LiveRangeIndex::invalid(),
            last_range: LiveRangeIndex::invalid(),
            spillset: SpillSetIndex::invalid(),
            prio: 0,
            spill_weight_and_props: 0,
            range_summary: RangeSummary::new(),
        });
        LiveBundleIndex::new(bundle)
    }

    fn merge_bundles(&mut self, from: LiveBundleIndex, to: LiveBundleIndex) -> bool {
        if from == to {
            // Merge bundle into self -- trivial merge.
            return true;
        }
        log::debug!(
            "merging from bundle{} to bundle{}",
            from.index(),
            to.index()
        );

        let vreg_from = self.ranges[self.bundles[from.index()].first_range.index()].vreg;
        let vreg_to = self.ranges[self.bundles[to.index()].first_range.index()].vreg;
        // Both bundles must deal with the same RegClass. All vregs in a bundle
        // have to have the same regclass (because bundles start with one vreg
        // and all merging happens here) so we can just sample the first vreg of
        // each bundle.
        let rc = self.vreg_regs[vreg_from.index()].class();
        if rc != self.vreg_regs[vreg_to.index()].class() {
            return false;
        }

        // If either bundle is already assigned (due to a pinned vreg), don't merge.
        if !self.bundles[from.index()].allocation.is_none()
            || !self.bundles[to.index()].allocation.is_none()
        {
            return false;
        }

        #[cfg(debug)]
        {
            // Sanity check: both bundles should contain only ranges with appropriate VReg classes.
            let mut iter = self.bundles[from.index()].first_range;
            while iter.is_valid() {
                let vreg = self.ranges[iter.index()].vreg;
                assert_eq!(rc, self.vregs[vreg.index()].reg.class());
                iter = self.ranges_hot[iter.index()].next_in_bundle;
            }
            let mut iter = self.bundles[to.index()].first_range;
            while iter.is_valid() {
                let vreg = self.ranges[iter.index()].vreg;
                assert_eq!(rc, self.vregs[vreg.index()].reg.class());
                iter = self.ranges_hot[iter.index()].next_in_bundle;
            }
        }

        // Check for overlap in LiveRanges.
        let mut iter0 = self.bundles[from.index()].first_range;
        let mut iter1 = self.bundles[to.index()].first_range;
        let mut range_count = 0;
        while iter0.is_valid() && iter1.is_valid() {
            range_count += 1;
            if range_count > 200 {
                // Limit merge complexity.
                return false;
            }

            if self.ranges_hot[iter0.index()].range.from >= self.ranges_hot[iter1.index()].range.to
            {
                iter1 = self.ranges_hot[iter1.index()].next_in_bundle;
            } else if self.ranges_hot[iter1.index()].range.from
                >= self.ranges_hot[iter0.index()].range.to
            {
                iter0 = self.ranges_hot[iter0.index()].next_in_bundle;
            } else {
                // Overlap -- cannot merge.
                return false;
            }
        }

        // If we reach here, then the bundles do not overlap -- merge them!
        // We do this with a merge-sort-like scan over both chains, removing
        // from `to` (`iter1`) and inserting into `from` (`iter0`).
        let mut iter0 = self.bundles[from.index()].first_range;
        let mut iter1 = self.bundles[to.index()].first_range;
        if iter0.is_invalid() {
            // `from` bundle is empty -- trivial merge.
            return true;
        }
        if iter1.is_invalid() {
            // `to` bundle is empty -- just move head/tail pointers over from
            // `from` and set `bundle` up-link on all ranges.
            let head = self.bundles[from.index()].first_range;
            let tail = self.bundles[from.index()].last_range;
            self.bundles[to.index()].first_range = head;
            self.bundles[to.index()].last_range = tail;
            self.bundles[from.index()].first_range = LiveRangeIndex::invalid();
            self.bundles[from.index()].last_range = LiveRangeIndex::invalid();
            while iter0.is_valid() {
                self.ranges[iter0.index()].bundle = from;
                iter0 = self.ranges_hot[iter0.index()].next_in_bundle;
            }
            return true;
        }

        // Two non-empty chains of LiveRanges: traverse both simultaneously and
        // merge links into `from`.
        let mut prev = LiveRangeIndex::invalid();
        while iter0.is_valid() || iter1.is_valid() {
            // Pick the next range.
            let next_range_iter = if iter0.is_valid() {
                if iter1.is_valid() {
                    if self.ranges_hot[iter0.index()].range.from
                        <= self.ranges_hot[iter1.index()].range.from
                    {
                        &mut iter0
                    } else {
                        &mut iter1
                    }
                } else {
                    &mut iter0
                }
            } else {
                &mut iter1
            };
            let next = *next_range_iter;
            *next_range_iter = self.ranges_hot[next.index()].next_in_bundle;

            // link from prev.
            if prev.is_valid() {
                self.ranges_hot[prev.index()].next_in_bundle = next;
            } else {
                self.bundles[to.index()].first_range = next;
            }
            self.bundles[to.index()].last_range = next;
            self.ranges[next.index()].bundle = to;
            prev = next;
        }
        self.bundles[from.index()].first_range = LiveRangeIndex::invalid();
        self.bundles[from.index()].last_range = LiveRangeIndex::invalid();

        true
    }

    fn insert_liverange_into_bundle(&mut self, bundle: LiveBundleIndex, lr: LiveRangeIndex) {
        log::debug!(
            "insert_liverange_into_bundle: lr {:?} bundle {:?}",
            lr,
            bundle
        );
        self.ranges_hot[lr.index()].next_in_bundle = LiveRangeIndex::invalid();
        self.ranges[lr.index()].bundle = bundle;
        if self.bundles[bundle.index()].first_range.is_invalid() {
            // Empty bundle.
            self.bundles[bundle.index()].first_range = lr;
            self.bundles[bundle.index()].last_range = lr;
        } else if self.ranges_hot[self.bundles[bundle.index()].first_range.index()]
            .range
            .to
            <= self.ranges_hot[lr.index()].range.from
        {
            // After last range in bundle.
            let last = self.bundles[bundle.index()].last_range;
            self.ranges_hot[last.index()].next_in_bundle = lr;
            self.bundles[bundle.index()].last_range = lr;
        } else {
            // Find location to insert.
            let mut iter = self.bundles[bundle.index()].first_range;
            let mut insert_after = LiveRangeIndex::invalid();
            let insert_range = self.ranges_hot[lr.index()].range;
            while iter.is_valid() {
                debug_assert!(!self.ranges_hot[iter.index()].range.overlaps(&insert_range));
                if self.ranges_hot[iter.index()].range.to <= insert_range.from {
                    break;
                }
                insert_after = iter;
                iter = self.ranges_hot[iter.index()].next_in_bundle;
            }
            if insert_after.is_valid() {
                self.ranges_hot[insert_after.index()].next_in_bundle = lr;
                if self.bundles[bundle.index()].last_range == insert_after {
                    self.bundles[bundle.index()].last_range = lr;
                }
            } else {
                let next = self.bundles[bundle.index()].first_range;
                self.ranges_hot[lr.index()].next_in_bundle = next;
                self.bundles[bundle.index()].first_range = lr;
            }
        }
    }

    fn merge_vreg_bundles(&mut self) {
        // Create a bundle for every vreg, initially.
        log::debug!("merge_vreg_bundles: creating vreg bundles");
        for vreg in 0..self.vregs.len() {
            let vreg = VRegIndex::new(vreg);
            if self.vregs[vreg.index()].first_range.is_invalid() {
                continue;
            }
            let bundle = self.create_bundle();
            let mut range = self.vregs[vreg.index()].first_range;
            while range.is_valid() {
                self.insert_liverange_into_bundle(bundle, range);
                range = self.ranges[range.index()].next_in_reg;
            }
            log::debug!("vreg v{} gets bundle{}", vreg.index(), bundle.index());

            // If this vreg is pinned, assign the allocation and block the PRegs.
            if let Some(preg) = self.func.is_pinned_vreg(self.vreg_regs[vreg.index()]) {
                self.bundles[bundle.index()].allocation = Allocation::reg(preg);

                let mut iter = self.bundles[bundle.index()].first_range;
                while iter.is_valid() {
                    let range = self.ranges_hot[iter.index()].range;
                    // Create a new LiveRange for the PReg
                    // reservation, unaffiliated with the VReg, to
                    // reserve it (like a clobber) without the
                    // possibility of eviction.
                    self.add_liverange_to_preg(range, preg);
                    iter = self.ranges_hot[iter.index()].next_in_bundle;
                }
                continue;
            }

            // Otherwise, create a spillslot for it.
            let ssidx = SpillSetIndex::new(self.spillsets.len());
            let reg = self.vreg_regs[vreg.index()];
            let size = self.func.spillslot_size(reg.class(), reg) as u8;
            self.spillsets.push(SpillSet {
                bundles: smallvec![],
                slot: SpillSlotIndex::invalid(),
                size,
                class: reg.class(),
                reg_hint: PReg::invalid(),
            });
            self.bundles[bundle.index()].spillset = ssidx;
        }

        for inst in 0..self.func.insts() {
            let inst = Inst::new(inst);

            // Attempt to merge Reuse-policy operand outputs with the
            // corresponding inputs.
            for op in self.func.inst_operands(inst) {
                if let OperandPolicy::Reuse(reuse_idx) = op.policy() {
                    let src_vreg = op.vreg();
                    let dst_vreg = self.func.inst_operands(inst)[reuse_idx].vreg();
                    log::debug!(
                        "trying to merge reused-input def: src {} to dst {}",
                        src_vreg,
                        dst_vreg
                    );
                    let src_bundle =
                        self.ranges[self.vregs[src_vreg.vreg()].first_range.index()].bundle;
                    assert!(src_bundle.is_valid());
                    let dest_bundle =
                        self.ranges[self.vregs[dst_vreg.vreg()].first_range.index()].bundle;
                    assert!(dest_bundle.is_valid());
                    self.merge_bundles(/* from */ dest_bundle, /* to */ src_bundle);
                }
            }
        }

        // Attempt to merge blockparams with their inputs.
        for i in 0..self.blockparam_outs.len() {
            let (from_vreg, _, _, to_vreg) = self.blockparam_outs[i];
            log::debug!(
                "trying to merge blockparam v{} with input v{}",
                to_vreg.index(),
                from_vreg.index()
            );
            let to_bundle = self.ranges[self.vregs[to_vreg.index()].first_range.index()].bundle;
            assert!(to_bundle.is_valid());
            let from_bundle = self.ranges[self.vregs[from_vreg.index()].first_range.index()].bundle;
            assert!(from_bundle.is_valid());
            log::debug!(
                " -> from bundle{} to bundle{}",
                from_bundle.index(),
                to_bundle.index()
            );
            self.merge_bundles(from_bundle, to_bundle);
        }

        // Attempt to merge move srcs/dsts.
        for i in 0..self.prog_move_merges.len() {
            let (src, dst) = self.prog_move_merges[i];
            log::debug!("trying to merge move src LR {:?} to dst LR {:?}", src, dst);
            let src = self.resolve_merged_lr(src);
            let dst = self.resolve_merged_lr(dst);
            log::debug!(
                "resolved LR-construction merging chains: move-merge is now src LR {:?} to dst LR {:?}",
                src,
                dst
            );
            let src_bundle = self.ranges[src.index()].bundle;
            assert!(src_bundle.is_valid());
            let dest_bundle = self.ranges[dst.index()].bundle;
            assert!(dest_bundle.is_valid());
            self.merge_bundles(/* from */ dest_bundle, /* to */ src_bundle);
        }

        // Now create range summaries for all bundles.
        for bundle in 0..self.bundles.len() {
            let bundle = LiveBundleIndex::new(bundle);
            let mut iter = self.bundles[bundle.index()].first_range;
            let start_idx = self.range_ranges.len();
            let start_pos = if iter.is_valid() {
                self.ranges_hot[iter.index()].range.from
            } else {
                ProgPoint::from_index(0)
            };
            let mut end_pos = start_pos;
            while iter.is_valid() {
                let range = self.ranges_hot[iter.index()].range;
                end_pos = range.to;
                self.range_ranges.push(range);
                iter = self.ranges_hot[iter.index()].next_in_bundle;
            }
            let end_idx = self.range_ranges.len();
            let bound = CodeRange {
                from: start_pos,
                to: end_pos,
            };
            self.bundles[bundle.index()].range_summary = RangeSummary {
                from: start_idx as u32,
                to: end_idx as u32,
                bound,
            };
        }

        log::debug!("done merging bundles");
    }

    fn resolve_merged_lr(&self, mut lr: LiveRangeIndex) -> LiveRangeIndex {
        let mut iter = 0;
        while iter < 100 && self.ranges[lr.index()].merged_into.is_valid() {
            lr = self.ranges[lr.index()].merged_into;
            iter += 1;
        }
        lr
    }

    fn compute_bundle_prio(&self, bundle: LiveBundleIndex) -> u32 {
        // The priority is simply the total "length" -- the number of
        // instructions covered by all LiveRanges.
        let mut iter = self.bundles[bundle.index()].first_range;
        let mut total = 0;
        while iter.is_valid() {
            total += self.ranges_hot[iter.index()].range.len() as u32;
            iter = self.ranges_hot[iter.index()].next_in_bundle;
        }
        total
    }

    fn queue_bundles(&mut self) {
        for bundle in 0..self.bundles.len() {
            if self.bundles[bundle].first_range.is_invalid() {
                continue;
            }
            if !self.bundles[bundle].allocation.is_none() {
                continue;
            }
            let bundle = LiveBundleIndex::new(bundle);
            let prio = self.compute_bundle_prio(bundle);
            self.bundles[bundle.index()].prio = prio;
            self.recompute_bundle_properties(bundle);
            self.allocation_queue.insert(bundle, prio as usize);
        }
        self.stats.merged_bundle_count = self.allocation_queue.heap.len();
    }

    fn process_bundles(&mut self) {
        let mut count = 0;
        while let Some(bundle) = self.allocation_queue.pop() {
            self.stats.process_bundle_count += 1;
            self.process_bundle(bundle);
            count += 1;
            if count > self.func.insts() * 50 {
                self.dump_state();
                panic!("Infinite loop!");
            }
        }
        self.stats.final_liverange_count = self.ranges.len();
        self.stats.final_bundle_count = self.bundles.len();
        self.stats.spill_bundle_count = self.spilled_bundles.len();
    }

    fn dump_state(&self) {
        log::debug!("Bundles:");
        for (i, b) in self.bundles.iter().enumerate() {
            log::debug!(
                "bundle{}: first_range={:?} last_range={:?} spillset={:?} alloc={:?}",
                i,
                b.first_range,
                b.last_range,
                b.spillset,
                b.allocation
            );
        }
        log::debug!("VRegs:");
        for (i, v) in self.vregs.iter().enumerate() {
            log::debug!("vreg{}: first_range={:?}", i, v.first_range,);
        }
        log::debug!("Ranges:");
        for (i, (r, rh)) in self.ranges.iter().zip(self.ranges_hot.iter()).enumerate() {
            log::debug!(
                concat!(
                    "range{}: range={:?} vreg={:?} bundle={:?} ",
                    "weight={} first_use={:?} last_use={:?} ",
                    "next_in_bundle={:?} next_in_reg={:?}"
                ),
                i,
                rh.range,
                r.vreg,
                r.bundle,
                r.uses_spill_weight(),
                r.first_use,
                r.last_use,
                rh.next_in_bundle,
                r.next_in_reg
            );
        }
        log::debug!("Uses:");
        for (i, u) in self.uses.iter().enumerate() {
            log::debug!(
                "use{}: op={:?} pos={:?} slot={} next_use={:?}",
                i,
                u.operand,
                u.pos,
                u.slot(),
                u.next_use(),
            );
        }
    }

    fn compute_requirement(&self, bundle: LiveBundleIndex) -> Option<Requirement> {
        let init_vreg = self.vreg_regs[self.ranges
            [self.bundles[bundle.index()].first_range.index()]
        .vreg
        .index()];
        let class = init_vreg.class();
        let mut needed = Requirement::Any(class);

        log::debug!(
            "compute_requirement: bundle {:?} class {:?} (from vreg {:?})",
            bundle,
            class,
            init_vreg
        );

        let mut iter = self.bundles[bundle.index()].first_range;
        while iter.is_valid() {
            let range_hot = &self.ranges_hot[iter.index()];
            let range = &self.ranges[iter.index()];
            log::debug!(" -> range {:?}", range_hot.range);
            let mut use_iter = range.first_use;
            while use_iter.is_valid() {
                let usedata = &self.uses[use_iter.index()];
                let use_op = usedata.operand;
                let use_req = Requirement::from_operand(use_op);
                log::debug!(" -> use {:?} op {:?} req {:?}", use_iter, use_op, use_req);
                needed = needed.merge(use_req)?;
                log::debug!("   -> needed {:?}", needed);
                use_iter = usedata.next_use();
            }
            iter = range_hot.next_in_bundle;
        }

        log::debug!(" -> final needed: {:?}", needed);
        Some(needed)
    }

    fn bundle_bounding_range_if_multiple(&self, bundle: LiveBundleIndex) -> Option<CodeRange> {
        let first_range = self.bundles[bundle.index()].first_range;
        let last_range = self.bundles[bundle.index()].last_range;
        if first_range.is_invalid() || first_range == last_range {
            return None;
        }
        Some(CodeRange {
            from: self.ranges_hot[first_range.index()].range.from,
            to: self.ranges_hot[last_range.index()].range.to,
        })
    }

    fn range_definitely_fits_in_reg(&self, range: CodeRange, reg: PRegIndex) -> bool {
        self.pregs[reg.index()]
            .allocations
            .btree
            .get(&LiveRangeKey::from_range(&range))
            .is_none()
    }

    fn try_to_allocate_bundle_to_reg(
        &mut self,
        bundle: LiveBundleIndex,
        reg: PRegIndex,
    ) -> AllocRegResult {
        log::debug!("try_to_allocate_bundle_to_reg: {:?} -> {:?}", bundle, reg);
        let mut conflicts = smallvec![];
        // Use the range-summary array; this allows fast streaming
        // access to CodeRanges (which are just two u32s packed
        // together) which is important for this hot loop.
        let iter = self.bundles[bundle.index()]
            .range_summary
            .iter(&self.range_ranges[..]);
        for range in iter {
            log::debug!(" -> range {:?}", range);
            // Note that the comparator function here tests for *overlap*, so we
            // are checking whether the BTree contains any preg range that
            // *overlaps* with range `range`, not literally the range `range`.
            if let Some(preg_range) = self.pregs[reg.index()]
                .allocations
                .btree
                .get(&LiveRangeKey::from_range(&range))
            {
                log::debug!(" -> btree contains range {:?} that overlaps", preg_range);
                if self.ranges[preg_range.index()].vreg.is_valid() {
                    log::debug!("   -> from vreg {:?}", self.ranges[preg_range.index()].vreg);
                    // range from an allocated bundle: find the bundle and add to
                    // conflicts list.
                    let conflict_bundle = self.ranges[preg_range.index()].bundle;
                    log::debug!("   -> conflict bundle {:?}", conflict_bundle);
                    if !conflicts.iter().any(|b| *b == conflict_bundle) {
                        conflicts.push(conflict_bundle);
                    }

                    // Empirically, it seems to be essentially as good
                    // to return only one conflicting bundle as all of
                    // them; it is very rare that the combination of
                    // all conflicting bundles yields a maximum spill
                    // weight that is enough to keep them in place
                    // when a single conflict does not. It is also a
                    // quite significant compile-time win to *stop
                    // scanning* as soon as we have a conflict. To
                    // experiment with this, however, just remove this
                    // `break`; the rest of the code will do the right
                    // thing.
                    break;
                } else {
                    log::debug!("   -> conflict with fixed reservation");
                    // range from a direct use of the PReg (due to clobber).
                    return AllocRegResult::ConflictWithFixed;
                }
            }
        }

        if conflicts.len() > 0 {
            return AllocRegResult::Conflict(conflicts);
        }

        // We can allocate! Add our ranges to the preg's BTree.
        let preg = self.pregs[reg.index()].reg;
        log::debug!("  -> bundle {:?} assigned to preg {:?}", bundle, preg);
        self.bundles[bundle.index()].allocation = Allocation::reg(preg);
        let mut iter = self.bundles[bundle.index()].first_range;
        while iter.is_valid() {
            let range = &self.ranges_hot[iter.index()];
            self.pregs[reg.index()]
                .allocations
                .btree
                .insert(LiveRangeKey::from_range(&range.range), iter);
            iter = range.next_in_bundle;
        }

        AllocRegResult::Allocated(Allocation::reg(preg))
    }

    fn evict_bundle(&mut self, bundle: LiveBundleIndex) {
        log::debug!(
            "evicting bundle {:?}: alloc {:?}",
            bundle,
            self.bundles[bundle.index()].allocation
        );
        let preg = match self.bundles[bundle.index()].allocation.as_reg() {
            Some(preg) => preg,
            None => {
                log::debug!(
                    "  -> has no allocation! {:?}",
                    self.bundles[bundle.index()].allocation
                );
                return;
            }
        };
        let preg_idx = PRegIndex::new(preg.index());
        self.bundles[bundle.index()].allocation = Allocation::none();
        let mut iter = self.bundles[bundle.index()].first_range;
        while iter.is_valid() {
            log::debug!(" -> removing LR {:?} from reg {:?}", iter, preg_idx);
            self.pregs[preg_idx.index()]
                .allocations
                .btree
                .remove(&LiveRangeKey::from_range(
                    &self.ranges_hot[iter.index()].range,
                ));
            iter = self.ranges_hot[iter.index()].next_in_bundle;
        }
        let prio = self.bundles[bundle.index()].prio;
        log::debug!(" -> prio {}; back into queue", prio);
        self.allocation_queue.insert(bundle, prio as usize);
    }

    fn bundle_spill_weight(&self, bundle: LiveBundleIndex) -> u32 {
        self.bundles[bundle.index()].cached_spill_weight()
    }

    fn maximum_spill_weight_in_bundle_set(&self, bundles: &LiveBundleVec) -> u32 {
        log::debug!("maximum_spill_weight_in_bundle_set: {:?}", bundles);
        let m = bundles
            .iter()
            .map(|&b| {
                let w = self.bundles[b.index()].cached_spill_weight();
                log::debug!("bundle{}: {}", b.index(), w);
                w
            })
            .max()
            .unwrap_or(0);
        log::debug!(" -> max: {}", m);
        m
    }

    fn recompute_bundle_properties(&mut self, bundle: LiveBundleIndex) {
        let minimal;
        let mut fixed = false;
        let bundledata = &self.bundles[bundle.index()];
        let first_range = &self.ranges[bundledata.first_range.index()];
        let first_range_hot = &self.ranges_hot[bundledata.first_range.index()];

        log::debug!("recompute bundle properties: bundle {:?}", bundle);

        if first_range.vreg.is_invalid() {
            log::debug!("  -> no vreg; minimal and fixed");
            minimal = true;
            fixed = true;
        } else {
            let mut use_iter = first_range.first_use;
            while use_iter.is_valid() {
                let use_data = &self.uses[use_iter.index()];
                if let OperandPolicy::FixedReg(_) = use_data.operand.policy() {
                    log::debug!("  -> fixed use {:?}", use_iter);
                    fixed = true;
                    break;
                }
                use_iter = use_data.next_use();
            }
            // Minimal if this is the only range in the bundle, and if
            // the range covers only one instruction. Note that it
            // could cover just one ProgPoint, i.e. X.Before..X.After,
            // or two ProgPoints, i.e. X.Before..X+1.Before.
            log::debug!("  -> first range has range {:?}", first_range_hot.range);
            log::debug!(
                "  -> first range has next in bundle {:?}",
                first_range_hot.next_in_bundle
            );
            minimal = first_range_hot.next_in_bundle.is_invalid()
                && first_range_hot.range.from.inst() == first_range_hot.range.to.prev().inst();
            log::debug!("  -> minimal: {}", minimal);
        }

        let spill_weight = if minimal {
            if fixed {
                log::debug!("  -> fixed and minimal: spill weight 2000000");
                2_000_000
            } else {
                log::debug!("  -> non-fixed and minimal: spill weight 1000000");
                1_000_000
            }
        } else {
            let mut total = 0;
            let mut range = self.bundles[bundle.index()].first_range;
            while range.is_valid() {
                let range_data = &self.ranges[range.index()];
                log::debug!(
                    "  -> uses spill weight: +{}",
                    range_data.uses_spill_weight()
                );
                total += range_data.uses_spill_weight();
                range = self.ranges_hot[range.index()].next_in_bundle;
            }

            if self.bundles[bundle.index()].prio > 0 {
                log::debug!(
                    " -> dividing by prio {}; final weight {}",
                    self.bundles[bundle.index()].prio,
                    total / self.bundles[bundle.index()].prio
                );
                total / self.bundles[bundle.index()].prio
            } else {
                0
            }
        };

        self.bundles[bundle.index()].set_cached_spill_weight_and_props(
            spill_weight,
            minimal,
            fixed,
        );
    }

    fn minimal_bundle(&mut self, bundle: LiveBundleIndex) -> bool {
        self.bundles[bundle.index()].cached_minimal()
    }

    fn find_split_points(
        &mut self,
        bundle: LiveBundleIndex,
        conflicting: LiveBundleIndex,
    ) -> SmallVec<[ProgPoint; 4]> {
        // Scan the bundle's ranges once. We want to record:
        // - Does the bundle contain any ranges in "hot" code and/or "cold" code?
        //   If so, record the transition points that are fully included in
        //   `bundle`: the first ProgPoint in a hot range if the prior cold
        //   point is also in the bundle; and the first ProgPoint in a cold
        //   range if the prior hot point is also in the bundle.
        // - Does the bundle cross any clobbering insts?
        //   If so, record the ProgPoint before each such instruction.
        // - Is there a register use before the conflicting bundle?
        //   If so, record the ProgPoint just after the last one.
        // - Is there a register use after the conflicting bundle?
        //   If so, record the ProgPoint just before the last one.
        //
        // Then choose one of the above kinds of splits, in priority order.

        let mut cold_hot_splits: SmallVec<[ProgPoint; 4]> = smallvec![];
        let mut clobber_splits: SmallVec<[ProgPoint; 4]> = smallvec![];
        let mut last_before_conflict: Option<ProgPoint> = None;
        let mut first_after_conflict: Option<ProgPoint> = None;

        log::debug!(
            "find_split_points: bundle {:?} conflicting {:?}",
            bundle,
            conflicting
        );

        // We simultaneously scan the sorted list of LiveRanges in our bundle
        // and the sorted list of call instruction locations. We also take the
        // total range (start of first range to end of last range) of the
        // conflicting bundle, if any, so we can find the last use before it and
        // first use after it. Each loop iteration handles one range in our
        // bundle. Calls are scanned up until they advance past the current
        // range.
        let mut our_iter = self.bundles[bundle.index()].first_range;
        let (conflict_from, conflict_to) = if conflicting.is_valid() {
            (
                Some(
                    self.ranges_hot[self.bundles[conflicting.index()].first_range.index()]
                        .range
                        .from,
                ),
                Some(
                    self.ranges_hot[self.bundles[conflicting.index()].last_range.index()]
                        .range
                        .to,
                ),
            )
        } else {
            (None, None)
        };

        let bundle_start = if self.bundles[bundle.index()].first_range.is_valid() {
            self.ranges_hot[self.bundles[bundle.index()].first_range.index()]
                .range
                .from
        } else {
            ProgPoint::before(Inst::new(0))
        };
        let bundle_end = if self.bundles[bundle.index()].last_range.is_valid() {
            self.ranges_hot[self.bundles[bundle.index()].last_range.index()]
                .range
                .to
        } else {
            ProgPoint::before(Inst::new(self.func.insts()))
        };

        log::debug!(" -> conflict from {:?} to {:?}", conflict_from, conflict_to);
        let mut clobberidx = 0;
        while our_iter.is_valid() {
            // Probe the hot-code tree.
            let our_range = self.ranges_hot[our_iter.index()].range;
            log::debug!(" -> range {:?}", our_range);
            if let Some(hot_range_idx) = self
                .hot_code
                .btree
                .get(&LiveRangeKey::from_range(&our_range))
            {
                // `hot_range_idx` is a range that *overlaps* with our range.

                // There may be cold code in our range on either side of the hot
                // range. Record the transition points if so.
                let hot_range = self.ranges_hot[hot_range_idx.index()].range;
                log::debug!("   -> overlaps with hot-code range {:?}", hot_range);
                let start_cold = our_range.from < hot_range.from;
                let end_cold = our_range.to > hot_range.to;
                if start_cold {
                    log::debug!(
                        "    -> our start is cold; potential split at cold->hot transition {:?}",
                        hot_range.from,
                    );
                    // First ProgPoint in hot range.
                    cold_hot_splits.push(hot_range.from);
                }
                if end_cold {
                    log::debug!(
                        "    -> our end is cold; potential split at hot->cold transition {:?}",
                        hot_range.to,
                    );
                    // First ProgPoint in cold range (after hot range).
                    cold_hot_splits.push(hot_range.to);
                }
            }

            // Scan through clobber-insts from last left-off position until the first
            // clobbering inst past this range. Record all clobber sites as potential
            // splits.
            while clobberidx < self.clobbers.len() {
                let cur_clobber = self.clobbers[clobberidx];
                let pos = ProgPoint::before(cur_clobber);
                if pos >= our_range.to {
                    break;
                }
                clobberidx += 1;
                if pos < our_range.from {
                    continue;
                }
                if pos > bundle_start {
                    log::debug!("   -> potential clobber split at {:?}", pos);
                    clobber_splits.push(pos);
                }
            }

            // Update last-before-conflict and first-before-conflict positions.

            let mut update_with_pos = |pos: ProgPoint| {
                let before_inst = ProgPoint::before(pos.inst());
                let before_next_inst = before_inst.next().next();
                if before_inst > bundle_start
                    && (conflict_from.is_none() || before_inst < conflict_from.unwrap())
                    && (last_before_conflict.is_none()
                        || before_inst > last_before_conflict.unwrap())
                {
                    last_before_conflict = Some(before_inst);
                }
                if before_next_inst < bundle_end
                    && (conflict_to.is_none() || pos >= conflict_to.unwrap())
                    && (first_after_conflict.is_none() || pos > first_after_conflict.unwrap())
                {
                    first_after_conflict = Some(ProgPoint::before(pos.inst().next()));
                }
            };

            let mut use_idx = self.ranges[our_iter.index()].first_use;
            while use_idx.is_valid() {
                let use_data = &self.uses[use_idx.index()];
                log::debug!("   -> range has use at {:?}", use_data.pos);
                update_with_pos(use_data.pos);
                use_idx = use_data.next_use();
            }

            our_iter = self.ranges_hot[our_iter.index()].next_in_bundle;
        }
        log::debug!(
            "  -> first use/def after conflict range: {:?}",
            first_after_conflict,
        );
        log::debug!(
            "  -> last use/def before conflict range: {:?}",
            last_before_conflict,
        );

        // Based on the above, we can determine which split strategy we are taking at this
        // iteration:
        // - If we span both hot and cold code, split into separate "hot" and "cold" bundles.
        // - Otherwise, if we span any calls, split just before every call instruction.
        // - Otherwise, if there is a register use after the conflicting bundle,
        //   split at that use-point ("split before first use").
        // - Otherwise, if there is a register use before the conflicting
        //   bundle, split at that use-point ("split after last use").
        // - Otherwise, split at every use, to form minimal bundles.

        if cold_hot_splits.len() > 0 {
            log::debug!(" going with cold/hot splits: {:?}", cold_hot_splits);
            self.stats.splits_hot += 1;
            cold_hot_splits
        } else if clobber_splits.len() > 0 {
            log::debug!(" going with clobber splits: {:?}", clobber_splits);
            self.stats.splits_clobbers += 1;
            clobber_splits
        } else if first_after_conflict.is_some() {
            self.stats.splits_conflicts += 1;
            log::debug!(" going with first after conflict");
            smallvec![first_after_conflict.unwrap()]
        } else if last_before_conflict.is_some() {
            self.stats.splits_conflicts += 1;
            log::debug!(" going with last before conflict");
            smallvec![last_before_conflict.unwrap()]
        } else {
            self.stats.splits_all += 1;
            log::debug!(" splitting at all uses");
            self.find_all_use_split_points(bundle)
        }
    }

    fn find_all_use_split_points(&self, bundle: LiveBundleIndex) -> SmallVec<[ProgPoint; 4]> {
        let mut splits = smallvec![];
        let mut iter = self.bundles[bundle.index()].first_range;
        log::debug!("finding all use/def splits for {:?}", bundle);
        let bundle_start = if iter.is_valid() {
            self.ranges_hot[iter.index()].range.from
        } else {
            ProgPoint::before(Inst::new(0))
        };
        // N.B.: a minimal bundle must include only ProgPoints in a
        // single instruction, but can include both (can include two
        // ProgPoints). We split here, taking care to never split *in
        // the middle* of an instruction, because we would not be able
        // to insert moves to reify such an assignment.
        while iter.is_valid() {
            log::debug!(
                " -> range {:?}: {:?}",
                iter,
                self.ranges_hot[iter.index()].range
            );
            let mut use_idx = self.ranges[iter.index()].first_use;
            while use_idx.is_valid() {
                let use_data = &self.uses[use_idx.index()];
                log::debug!("  -> use: {:?}", use_data);
                let before_use_inst = if use_data.operand.kind() == OperandKind::Def {
                    // For a def, split *at* the def -- this may be an
                    // After point, but the value cannot be live into
                    // the def so we don't need to insert a move.
                    use_data.pos
                } else {
                    // For an use or mod, split before the instruction
                    // -- this allows us to insert a move if
                    // necessary.
                    ProgPoint::before(use_data.pos.inst())
                };
                let after_use_inst = ProgPoint::before(use_data.pos.inst().next());
                log::debug!(
                    "  -> splitting before and after use: {:?} and {:?}",
                    before_use_inst,
                    after_use_inst,
                );
                if before_use_inst > bundle_start {
                    splits.push(before_use_inst);
                }
                splits.push(after_use_inst);
                use_idx = use_data.next_use();
            }

            iter = self.ranges_hot[iter.index()].next_in_bundle;
        }
        splits.sort_unstable();
        log::debug!(" -> final splits: {:?}", splits);
        splits
    }

    fn split_and_requeue_bundle(
        &mut self,
        bundle: LiveBundleIndex,
        first_conflicting_bundle: LiveBundleIndex,
    ) {
        self.stats.splits += 1;
        // Try splitting: (i) across hot code; (ii) across all calls,
        // if we had a fixed-reg conflict; (iii) before first reg use;
        // (iv) after reg use; (v) around all register uses.  After
        // each type of split, check for conflict with conflicting
        // bundle(s); stop when no conflicts. In all cases, re-queue
        // the split bundles on the allocation queue.
        //
        // The critical property here is that we must eventually split
        // down to minimal bundles, which consist just of live ranges
        // around each individual def/use (this is step (v)
        // above). This ensures termination eventually.

        let split_points = self.find_split_points(bundle, first_conflicting_bundle);
        log::debug!(
            "split bundle {:?} (conflict {:?}): split points {:?}",
            bundle,
            first_conflicting_bundle,
            split_points
        );

        // Split `bundle` at every ProgPoint in `split_points`,
        // creating new LiveRanges and bundles (and updating vregs'
        // linked lists appropriately), and enqueue the new bundles.
        //
        // We uphold several basic invariants here:
        // - The LiveRanges in every vreg, and in every bundle, are disjoint
        // - Every bundle for a given vreg is disjoint
        //
        // To do so, we make one scan in program order: all ranges in
        // the bundle, and the def/all uses in each range. We track
        // the currently active bundle. For each range, we distribute
        // its uses among one or more ranges, depending on whether it
        // crosses any split points. If we had to split a range, then
        // we need to insert the new subparts in its vreg as
        // well. N.B.: to avoid the need to *remove* ranges from vregs
        // (which we could not do without a lookup, since we use
        // singly-linked lists and the bundle may contain multiple
        // vregs so we cannot simply scan a single vreg simultaneously
        // to the main scan), we instead *trim* the existing range
        // into its first subpart, and then create the new
        // subparts. Note that shrinking a LiveRange is always legal
        // (as long as one replaces the shrunk space with new
        // LiveRanges).
        //
        // Note that the original IonMonkey splitting code is quite a
        // bit more complex and has some subtle invariants. We stick
        // to the above invariants to keep this code maintainable.

        let mut split_idx = 0;

        // Fast-forward past any splits that occur before or exactly
        // at the start of the first range in the bundle.
        let first_range = self.bundles[bundle.index()].first_range;
        let bundle_start = if first_range.is_valid() {
            self.ranges_hot[first_range.index()].range.from
        } else {
            ProgPoint::before(Inst::new(0))
        };
        while split_idx < split_points.len() && split_points[split_idx] <= bundle_start {
            split_idx += 1;
        }

        let mut new_bundles: LiveBundleVec = smallvec![];
        let mut cur_bundle = bundle;
        let mut iter = self.bundles[bundle.index()].first_range;
        self.bundles[bundle.index()].first_range = LiveRangeIndex::invalid();
        self.bundles[bundle.index()].last_range = LiveRangeIndex::invalid();
        let mut range_summary_idx = self.bundles[bundle.index()].range_summary.from;
        while iter.is_valid() {
            // Read `next` link now and then clear it -- we rebuild the list below.
            let next = self.ranges_hot[iter.index()].next_in_bundle;
            self.ranges_hot[iter.index()].next_in_bundle = LiveRangeIndex::invalid();

            let mut range = self.ranges_hot[iter.index()].range;
            log::debug!(" -> has range {:?} (LR {:?})", range, iter);

            // If any splits occur before this range, create a new
            // bundle, then advance to the first split within the
            // range.
            if split_idx < split_points.len() && split_points[split_idx] <= range.from {
                cur_bundle = self.create_bundle();
                log::debug!(
                    "  -> split before a range; creating new bundle {:?}",
                    cur_bundle
                );
                self.bundles[cur_bundle.index()].spillset = self.bundles[bundle.index()].spillset;
                new_bundles.push(cur_bundle);
                split_idx += 1;
                self.bundles[cur_bundle.index()].range_summary.from = range_summary_idx;
            }
            while split_idx < split_points.len() && split_points[split_idx] <= range.from {
                split_idx += 1;
            }

            // Link into current bundle.
            self.ranges[iter.index()].bundle = cur_bundle;
            if self.bundles[cur_bundle.index()].first_range.is_valid() {
                self.ranges_hot[self.bundles[cur_bundle.index()].last_range.index()]
                    .next_in_bundle = iter;
            } else {
                self.bundles[cur_bundle.index()].first_range = iter;
            }
            self.bundles[cur_bundle.index()].last_range = iter;

            // While the next split point is beyond the start of the
            // range and before the end, shorten the current LiveRange
            // (this is always legal) and create a new Bundle and
            // LiveRange for the remainder. Truncate the old bundle
            // (set last_range). Insert the LiveRange into the vreg
            // and into the new bundle. Then move the use-chain over,
            // splitting at the appropriate point.
            //
            // We accumulate the use stats (fixed-use count and spill
            // weight) as we scan through uses, recomputing the values
            // for the truncated initial LiveRange and taking the
            // remainders for the split "rest" LiveRange.

            while split_idx < split_points.len() && split_points[split_idx] < range.to {
                let split_point = split_points[split_idx];
                split_idx += 1;

                // Skip forward to the current range.
                if split_point <= range.from {
                    continue;
                }

                log::debug!(
                    " -> processing split point {:?} with iter {:?}",
                    split_point,
                    iter
                );

                // We split into `first` and `rest`. `rest` may be
                // further subdivided in subsequent iterations; we
                // only do one split per iteration.
                debug_assert!(range.from < split_point && split_point < range.to);
                let rest_range = CodeRange {
                    from: split_point,
                    to: self.ranges_hot[iter.index()].range.to,
                };
                self.ranges_hot[iter.index()].range.to = split_point;
                range = rest_range;
                log::debug!(
                    " -> range of {:?} now {:?}",
                    iter,
                    self.ranges_hot[iter.index()].range
                );

                // Create the rest-range and insert it into the vreg's
                // range list. (Note that the vreg does not keep a
                // tail-pointer so we do not need to update that.)
                let rest_lr = self.create_liverange(rest_range);
                self.ranges[rest_lr.index()].vreg = self.ranges[iter.index()].vreg;
                self.ranges[rest_lr.index()].next_in_reg = self.ranges[iter.index()].next_in_reg;
                self.ranges[iter.index()].next_in_reg = rest_lr;

                log::debug!(
                    " -> split tail to new LR {:?} with range {:?}",
                    rest_lr,
                    rest_range
                );

                // Scan over uses, accumulating stats for those that
                // stay in the first range, finding the first use that
                // moves to the rest range.
                let mut last_use_in_first_range = UseIndex::invalid();
                let mut use_iter = self.ranges[iter.index()].first_use;
                let mut uses_spill_weight = 0;
                while use_iter.is_valid() {
                    if self.uses[use_iter.index()].pos >= split_point {
                        break;
                    }
                    last_use_in_first_range = use_iter;
                    let policy = self.uses[use_iter.index()].operand.policy();
                    log::debug!(
                        " -> use {:?} before split point; policy {:?}",
                        use_iter,
                        policy
                    );
                    uses_spill_weight += spill_weight_from_policy(policy);
                    log::debug!("   -> use {:?} remains in orig", use_iter);
                    use_iter = self.uses[use_iter.index()].next_use();
                }

                // Move over `rest`'s uses and update stats on first
                // and rest LRs.
                if use_iter.is_valid() {
                    log::debug!(
                        "   -> moving uses over the split starting at {:?}",
                        use_iter
                    );
                    self.ranges[rest_lr.index()].first_use = use_iter;
                    self.ranges[rest_lr.index()].last_use = self.ranges[iter.index()].last_use;

                    self.ranges[iter.index()].last_use = last_use_in_first_range;
                    if last_use_in_first_range.is_valid() {
                        self.uses[last_use_in_first_range.index()]
                            .set_next_use(UseIndex::invalid());
                    } else {
                        self.ranges[iter.index()].first_use = UseIndex::invalid();
                    }

                    let new_spill_weight =
                        self.ranges[iter.index()].uses_spill_weight() - uses_spill_weight;
                    self.ranges[rest_lr.index()].set_uses_spill_weight(new_spill_weight);
                    self.ranges[iter.index()].set_uses_spill_weight(uses_spill_weight);
                }

                log::debug!(
                    " -> range {:?} next-in-bundle is {:?}",
                    iter,
                    self.ranges_hot[iter.index()].next_in_bundle
                );

                // Create a new bundle to hold the rest-range.
                let rest_bundle = self.create_bundle();
                self.bundles[cur_bundle.index()].range_summary.to = range_summary_idx + 1;
                cur_bundle = rest_bundle;
                self.bundles[cur_bundle.index()].range_summary.from = range_summary_idx;
                self.bundles[cur_bundle.index()].range_summary.to = range_summary_idx + 1;
                new_bundles.push(rest_bundle);
                self.bundles[rest_bundle.index()].first_range = rest_lr;
                self.bundles[rest_bundle.index()].last_range = rest_lr;
                self.bundles[rest_bundle.index()].spillset = self.bundles[bundle.index()].spillset;
                self.ranges[rest_lr.index()].bundle = rest_bundle;
                log::debug!(" -> new bundle {:?} for LR {:?}", rest_bundle, rest_lr);

                iter = rest_lr;
            }

            iter = next;
            range_summary_idx += 1;
            self.bundles[cur_bundle.index()].range_summary.to = range_summary_idx;
        }

        self.fixup_range_summary_bound(bundle);
        for &b in &new_bundles {
            self.fixup_range_summary_bound(b);
        }

        // Enqueue all split-bundles on the allocation queue.
        let prio = self.compute_bundle_prio(bundle);
        self.bundles[bundle.index()].prio = prio;
        self.recompute_bundle_properties(bundle);
        self.allocation_queue.insert(bundle, prio as usize);
        for &b in &new_bundles {
            let prio = self.compute_bundle_prio(b);
            self.bundles[b.index()].prio = prio;
            self.recompute_bundle_properties(b);
            self.allocation_queue.insert(b, prio as usize);
        }
    }

    fn fixup_range_summary_bound(&mut self, bundle: LiveBundleIndex) {
        let bundledata = &mut self.bundles[bundle.index()];
        let from = if bundledata.first_range.is_valid() {
            self.ranges_hot[bundledata.first_range.index()].range.from
        } else {
            ProgPoint::from_index(0)
        };
        let to = if bundledata.last_range.is_valid() {
            self.ranges_hot[bundledata.last_range.index()].range.to
        } else {
            ProgPoint::from_index(0)
        };
        bundledata.range_summary.bound = CodeRange { from, to };

        #[cfg(debug_assertions)]
        {
            // Sanity check: ensure that ranges returned by the range
            // summary correspond to actual ranges.
            let mut iter = self.bundles[bundle.index()].first_range;
            let mut summary_iter = self.bundles[bundle.index()]
                .range_summary
                .iter(&self.range_ranges[..]);
            while iter.is_valid() {
                assert_eq!(
                    summary_iter.next(),
                    Some(self.ranges_hot[iter.index()].range)
                );
                iter = self.ranges_hot[iter.index()].next_in_bundle;
            }
            assert_eq!(summary_iter.next(), None);
        }
    }

    fn process_bundle(&mut self, bundle: LiveBundleIndex) {
        // Find any requirements: for every LR, for every def/use, gather
        // requirements (fixed-reg, any-reg, any) and merge them.
        let req = self.compute_requirement(bundle);
        // Grab a hint from our spillset, if any.
        let hint_reg = self.spillsets[self.bundles[bundle.index()].spillset.index()].reg_hint;
        log::debug!(
            "process_bundle: bundle {:?} requirement {:?} hint {:?}",
            bundle,
            req,
            hint_reg,
        );

        // Try to allocate!
        let mut attempts = 0;
        let mut first_conflicting_bundle;
        loop {
            attempts += 1;
            log::debug!("attempt {}, req {:?}", attempts, req);
            debug_assert!(attempts < 100 * self.func.insts());
            first_conflicting_bundle = None;
            let req = match req {
                Some(r) => r,
                // `None` means conflicting requirements, hence impossible to
                // allocate.
                None => break,
            };

            let conflicting_bundles = match req {
                Requirement::Fixed(preg) => {
                    let preg_idx = PRegIndex::new(preg.index());
                    self.stats.process_bundle_reg_probes_fixed += 1;
                    log::debug!("trying fixed reg {:?}", preg_idx);
                    match self.try_to_allocate_bundle_to_reg(bundle, preg_idx) {
                        AllocRegResult::Allocated(alloc) => {
                            self.stats.process_bundle_reg_success_fixed += 1;
                            log::debug!(" -> allocated to fixed {:?}", preg_idx);
                            self.spillsets[self.bundles[bundle.index()].spillset.index()]
                                .reg_hint = alloc.as_reg().unwrap();
                            return;
                        }
                        AllocRegResult::Conflict(bundles) => {
                            log::debug!(" -> conflict with bundles {:?}", bundles);
                            bundles
                        }
                        AllocRegResult::ConflictWithFixed => {
                            log::debug!(" -> conflict with fixed alloc");
                            // Empty conflicts set: there's nothing we can
                            // evict, because fixed conflicts cannot be moved.
                            smallvec![]
                        }
                    }
                }
                Requirement::Register(class) => {
                    // Scan all pregs and attempt to allocate.
                    let mut lowest_cost_conflict_set: Option<LiveBundleVec> = None;

                    // Heuristic: start the scan for an available
                    // register at an offset influenced both by our
                    // location in the code and by the bundle we're
                    // considering. This has the effect of spreading
                    // demand more evenly across registers.
                    let scan_offset = self.ranges_hot
                        [self.bundles[bundle.index()].first_range.index()]
                    .range
                    .from
                    .inst()
                    .index()
                        + bundle.index();

                    // If the bundle is more than one range, see if we
                    // can find a reg that the bounding range fits
                    // completely in first. Use that if so. Otherwise,
                    // do a detailed (liverange-by-liverange) probe of
                    // each reg in preference order.
                    let bounding_range = self.bundle_bounding_range_if_multiple(bundle);
                    if let Some(bounding_range) = bounding_range {
                        log::debug!("initial scan with bounding range {:?}", bounding_range);
                        self.stats.process_bundle_bounding_range_probe_start_any += 1;
                        for preg in RegTraversalIter::new(
                            self.env,
                            class,
                            hint_reg,
                            PReg::invalid(),
                            scan_offset,
                        ) {
                            let preg_idx = PRegIndex::new(preg.index());
                            log::debug!("trying preg {:?}", preg_idx);
                            self.stats.process_bundle_bounding_range_probes_any += 1;
                            if self.range_definitely_fits_in_reg(bounding_range, preg_idx) {
                                let result = self.try_to_allocate_bundle_to_reg(bundle, preg_idx);
                                self.stats.process_bundle_bounding_range_success_any += 1;
                                let alloc = match result {
                                    AllocRegResult::Allocated(alloc) => alloc,
                                    _ => panic!("Impossible result: {:?}", result),
                                };
                                self.spillsets[self.bundles[bundle.index()].spillset.index()]
                                    .reg_hint = alloc.as_reg().unwrap();
                                log::debug!(" -> definitely fits; assigning");
                                return;
                            }
                        }
                    }

                    self.stats.process_bundle_reg_probe_start_any += 1;
                    for preg in RegTraversalIter::new(
                        self.env,
                        class,
                        hint_reg,
                        PReg::invalid(),
                        scan_offset,
                    ) {
                        self.stats.process_bundle_reg_probes_any += 1;
                        let preg_idx = PRegIndex::new(preg.index());
                        log::debug!("trying preg {:?}", preg_idx);
                        match self.try_to_allocate_bundle_to_reg(bundle, preg_idx) {
                            AllocRegResult::Allocated(alloc) => {
                                self.stats.process_bundle_reg_success_any += 1;
                                log::debug!(" -> allocated to any {:?}", preg_idx);
                                self.spillsets[self.bundles[bundle.index()].spillset.index()]
                                    .reg_hint = alloc.as_reg().unwrap();
                                return;
                            }
                            AllocRegResult::Conflict(bundles) => {
                                log::debug!(" -> conflict with bundles {:?}", bundles);
                                if lowest_cost_conflict_set.is_none() {
                                    lowest_cost_conflict_set = Some(bundles);
                                } else if self.maximum_spill_weight_in_bundle_set(&bundles)
                                    < self.maximum_spill_weight_in_bundle_set(
                                        lowest_cost_conflict_set.as_ref().unwrap(),
                                    )
                                {
                                    lowest_cost_conflict_set = Some(bundles);
                                }
                            }
                            AllocRegResult::ConflictWithFixed => {
                                log::debug!(" -> conflict with fixed alloc");
                                // Simply don't consider as an option.
                            }
                        }
                    }

                    // Otherwise, we *require* a register, but didn't fit into
                    // any with current bundle assignments. Hence, we will need
                    // to either split or attempt to evict some bundles. Return
                    // the conflicting bundles to evict and retry. Empty list
                    // means nothing to try (due to fixed conflict) so we must
                    // split instead.
                    lowest_cost_conflict_set.unwrap_or(smallvec![])
                }

                Requirement::Stack(_) => {
                    // If we must be on the stack, put ourselves on
                    // the spillset's list immediately.
                    self.spillsets[self.bundles[bundle.index()].spillset.index()]
                        .bundles
                        .push(bundle);
                    return;
                }

                Requirement::Any(_) => {
                    // If a register is not *required*, spill now (we'll retry
                    // allocation on spilled bundles later).
                    log::debug!("spilling bundle {:?} to spilled_bundles list", bundle);
                    self.spilled_bundles.push(bundle);
                    return;
                }
            };

            log::debug!(" -> conflict set {:?}", conflicting_bundles);

            // If we have already tried evictions once before and are
            // still unsuccessful, give up and move on to splitting as
            // long as this is not a minimal bundle.
            if attempts >= 2 && !self.minimal_bundle(bundle) {
                break;
            }

            // If we hit a fixed conflict, give up and move on to splitting.
            if conflicting_bundles.is_empty() {
                break;
            }

            first_conflicting_bundle = Some(conflicting_bundles[0]);

            // If the maximum spill weight in the conflicting-bundles set is >= this bundle's spill
            // weight, then don't evict.
            let max_spill_weight = self.maximum_spill_weight_in_bundle_set(&conflicting_bundles);
            log::debug!(
                " -> max_spill_weight = {}; our spill weight {}",
                max_spill_weight,
                self.bundle_spill_weight(bundle)
            );
            if max_spill_weight >= self.bundle_spill_weight(bundle) {
                log::debug!(" -> we're already the cheapest bundle to spill -- going to split");
                break;
            }

            // Evict all bundles in `conflicting bundles` and try again.
            self.stats.evict_bundle_event += 1;
            for &bundle in &conflicting_bundles {
                log::debug!(" -> evicting {:?}", bundle);
                self.evict_bundle(bundle);
                self.stats.evict_bundle_count += 1;
            }
        }

        // A minimal bundle cannot be split.
        if self.minimal_bundle(bundle) {
            self.dump_state();
        }
        assert!(!self.minimal_bundle(bundle));

        self.split_and_requeue_bundle(
            bundle,
            first_conflicting_bundle.unwrap_or(LiveBundleIndex::invalid()),
        );
    }

    fn try_allocating_regs_for_spilled_bundles(&mut self) {
        for i in 0..self.spilled_bundles.len() {
            let bundle = self.spilled_bundles[i]; // don't borrow self
            let any_vreg = self.vreg_regs[self.ranges
                [self.bundles[bundle.index()].first_range.index()]
            .vreg
            .index()];
            let class = any_vreg.class();
            let mut success = false;
            self.stats.spill_bundle_reg_probes += 1;
            for preg in RegTraversalIter::new(
                self.env,
                class,
                PReg::invalid(),
                PReg::invalid(),
                bundle.index(),
            ) {
                let preg_idx = PRegIndex::new(preg.index());
                if let AllocRegResult::Allocated(_) =
                    self.try_to_allocate_bundle_to_reg(bundle, preg_idx)
                {
                    self.stats.spill_bundle_reg_success += 1;
                    success = true;
                    break;
                }
            }
            if !success {
                log::debug!(
                    "spilling bundle {:?} to spillset bundle list {:?}",
                    bundle,
                    self.bundles[bundle.index()].spillset
                );
                self.spillsets[self.bundles[bundle.index()].spillset.index()]
                    .bundles
                    .push(bundle);
            }
        }
    }

    fn spillslot_can_fit_spillset(
        &mut self,
        spillslot: SpillSlotIndex,
        spillset: SpillSetIndex,
    ) -> bool {
        for &bundle in &self.spillsets[spillset.index()].bundles {
            let mut iter = self.bundles[bundle.index()].first_range;
            while iter.is_valid() {
                let range = self.ranges_hot[iter.index()].range;
                if self.spillslots[spillslot.index()]
                    .ranges
                    .btree
                    .contains_key(&LiveRangeKey::from_range(&range))
                {
                    return false;
                }
                iter = self.ranges_hot[iter.index()].next_in_bundle;
            }
        }
        true
    }

    fn allocate_spillset_to_spillslot(
        &mut self,
        spillset: SpillSetIndex,
        spillslot: SpillSlotIndex,
    ) {
        self.spillsets[spillset.index()].slot = spillslot;
        for i in 0..self.spillsets[spillset.index()].bundles.len() {
            // don't borrow self
            let bundle = self.spillsets[spillset.index()].bundles[i];
            log::debug!(
                "spillslot {:?} alloc'ed to spillset {:?}: bundle {:?}",
                spillslot,
                spillset,
                bundle
            );
            let mut iter = self.bundles[bundle.index()].first_range;
            while iter.is_valid() {
                log::debug!(
                    "spillslot {:?} getting range {:?} from bundle {:?}: {:?}",
                    spillslot,
                    iter,
                    bundle,
                    self.ranges_hot[iter.index()].range
                );
                let range = self.ranges_hot[iter.index()].range;
                self.spillslots[spillslot.index()]
                    .ranges
                    .btree
                    .insert(LiveRangeKey::from_range(&range), iter);
                iter = self.ranges_hot[iter.index()].next_in_bundle;
            }
        }
    }

    fn allocate_spillslots(&mut self) {
        for spillset in 0..self.spillsets.len() {
            log::debug!("allocate spillslot: {}", spillset);
            let spillset = SpillSetIndex::new(spillset);
            if self.spillsets[spillset.index()].bundles.is_empty() {
                continue;
            }
            // Get or create the spillslot list for this size.
            let size = self.spillsets[spillset.index()].size as usize;
            if size >= self.slots_by_size.len() {
                self.slots_by_size.resize(
                    size + 1,
                    SpillSlotList {
                        first_spillslot: SpillSlotIndex::invalid(),
                        last_spillslot: SpillSlotIndex::invalid(),
                    },
                );
            }
            // Try a few existing spillslots.
            let mut spillslot_iter = self.slots_by_size[size].first_spillslot;
            let mut first_slot = SpillSlotIndex::invalid();
            let mut prev = SpillSlotIndex::invalid();
            let mut success = false;
            for _attempt in 0..10 {
                if spillslot_iter.is_invalid() {
                    break;
                }
                if spillslot_iter == first_slot {
                    // We've started looking at slots we placed at the end; end search.
                    break;
                }
                if first_slot.is_invalid() {
                    first_slot = spillslot_iter;
                }

                if self.spillslot_can_fit_spillset(spillslot_iter, spillset) {
                    self.allocate_spillset_to_spillslot(spillset, spillslot_iter);
                    success = true;
                    break;
                }
                // Remove the slot and place it at the end of the respective list.
                let next = self.spillslots[spillslot_iter.index()].next_spillslot;
                if prev.is_valid() {
                    self.spillslots[prev.index()].next_spillslot = next;
                } else {
                    self.slots_by_size[size].first_spillslot = next;
                }
                if !next.is_valid() {
                    self.slots_by_size[size].last_spillslot = prev;
                }

                let last = self.slots_by_size[size].last_spillslot;
                if last.is_valid() {
                    self.spillslots[last.index()].next_spillslot = spillslot_iter;
                } else {
                    self.slots_by_size[size].first_spillslot = spillslot_iter;
                }
                self.slots_by_size[size].last_spillslot = spillslot_iter;

                prev = spillslot_iter;
                spillslot_iter = next;
            }

            if !success {
                // Allocate a new spillslot.
                let spillslot = SpillSlotIndex::new(self.spillslots.len());
                let next = self.slots_by_size[size].first_spillslot;
                self.spillslots.push(SpillSlotData {
                    ranges: LiveRangeSet::new(),
                    next_spillslot: next,
                    size: size as u32,
                    alloc: Allocation::none(),
                    class: self.spillsets[spillset.index()].class,
                });
                self.slots_by_size[size].first_spillslot = spillslot;
                if !next.is_valid() {
                    self.slots_by_size[size].last_spillslot = spillslot;
                }

                self.allocate_spillset_to_spillslot(spillset, spillslot);
            }
        }

        // Assign actual slot indices to spillslots.
        let mut offset: u32 = 0;
        for data in &mut self.spillslots {
            // Align up to `size`.
            debug_assert!(data.size.is_power_of_two());
            offset = (offset + data.size - 1) & !(data.size - 1);
            let slot = if self.func.multi_spillslot_named_by_last_slot() {
                offset + data.size - 1
            } else {
                offset
            };
            data.alloc = Allocation::stack(SpillSlot::new(slot as usize, data.class));
            offset += data.size;
        }
        self.num_spillslots = offset;

        log::debug!("spillslot allocator done");
    }

    fn is_start_of_block(&self, pos: ProgPoint) -> bool {
        let block = self.cfginfo.insn_block[pos.inst().index()];
        pos == self.cfginfo.block_entry[block.index()]
    }
    fn is_end_of_block(&self, pos: ProgPoint) -> bool {
        let block = self.cfginfo.insn_block[pos.inst().index()];
        pos == self.cfginfo.block_exit[block.index()]
    }

    fn insert_move(
        &mut self,
        pos: ProgPoint,
        prio: InsertMovePrio,
        from_alloc: Allocation,
        to_alloc: Allocation,
    ) {
        debug!(
            "insert_move: pos {:?} prio {:?} from_alloc {:?} to_alloc {:?}",
            pos, prio, from_alloc, to_alloc
        );
        match (from_alloc.as_reg(), to_alloc.as_reg()) {
            (Some(from), Some(to)) => {
                assert_eq!(from.class(), to.class());
            }
            _ => {}
        }
        self.inserted_moves.push(InsertedMove {
            pos,
            prio,
            from_alloc,
            to_alloc,
        });
    }

    fn get_alloc(&self, inst: Inst, slot: usize) -> Allocation {
        let inst_allocs = &self.allocs[self.inst_alloc_offsets[inst.index()] as usize..];
        inst_allocs[slot]
    }

    fn set_alloc(&mut self, inst: Inst, slot: usize, alloc: Allocation) {
        let inst_allocs = &mut self.allocs[self.inst_alloc_offsets[inst.index()] as usize..];
        inst_allocs[slot] = alloc;
    }

    fn get_alloc_for_range(&self, range: LiveRangeIndex) -> Allocation {
        let bundledata = &self.bundles[self.ranges[range.index()].bundle.index()];
        if bundledata.allocation != Allocation::none() {
            bundledata.allocation
        } else {
            self.spillslots[self.spillsets[bundledata.spillset.index()].slot.index()].alloc
        }
    }

    fn apply_allocations_and_insert_moves(&mut self) {
        log::debug!("blockparam_ins: {:?}", self.blockparam_ins);
        log::debug!("blockparam_outs: {:?}", self.blockparam_outs);

        /// We create "half-moves" in order to allow a single-scan
        /// strategy with a subsequent sort. Basically, the key idea
        /// is that as our single scan through a range for a vreg hits
        /// upon the source or destination of an edge-move, we emit a
        /// "half-move". These half-moves are carefully keyed in a
        /// particular sort order (the field order below is
        /// significant!) so that all half-moves on a given (from, to)
        /// block-edge appear contiguously, and then all moves from a
        /// given vreg appear contiguously. Within a given from-vreg,
        /// pick the first `Source` (there should only be one, but
        /// imprecision in liveranges due to loop handling sometimes
        /// means that a blockparam-out is also recognized as a normal-out),
        /// and then for each `Dest`, copy the source-alloc to that
        /// dest-alloc.
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct HalfMove {
            key: u64,
            alloc: Allocation,
        }
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
        #[repr(u8)]
        enum HalfMoveKind {
            Source = 0,
            Dest = 1,
        }
        fn half_move_key(
            from_block: Block,
            to_block: Block,
            to_vreg: VRegIndex,
            kind: HalfMoveKind,
        ) -> u64 {
            assert!(from_block.index() < 1 << 21);
            assert!(to_block.index() < 1 << 21);
            assert!(to_vreg.index() < 1 << 21);
            ((from_block.index() as u64) << 43)
                | ((to_block.index() as u64) << 22)
                | ((to_vreg.index() as u64) << 1)
                | (kind as u8 as u64)
        }
        impl HalfMove {
            fn from_block(&self) -> Block {
                Block::new(((self.key >> 43) & ((1 << 21) - 1)) as usize)
            }
            fn to_block(&self) -> Block {
                Block::new(((self.key >> 22) & ((1 << 21) - 1)) as usize)
            }
            fn to_vreg(&self) -> VRegIndex {
                VRegIndex::new(((self.key >> 1) & ((1 << 21) - 1)) as usize)
            }
            fn kind(&self) -> HalfMoveKind {
                if self.key & 1 == 1 {
                    HalfMoveKind::Dest
                } else {
                    HalfMoveKind::Source
                }
            }
        }

        let mut half_moves: Vec<HalfMove> = Vec::with_capacity(6 * self.func.insts());
        let mut reuse_input_insts = Vec::with_capacity(self.func.insts() / 2);

        let mut blockparam_in_idx = 0;
        let mut blockparam_out_idx = 0;
        let mut prog_move_src_idx = 0;
        let mut prog_move_dst_idx = 0;
        for vreg in 0..self.vregs.len() {
            let vreg = VRegIndex::new(vreg);

            // For each range in each vreg, insert moves or
            // half-moves.  We also scan over `blockparam_ins` and
            // `blockparam_outs`, which are sorted by (block, vreg),
            // and over program-move srcs/dsts to fill in allocations.
            let mut iter = self.vregs[vreg.index()].first_range;
            let mut prev = LiveRangeIndex::invalid();
            while iter.is_valid() {
                let alloc = self.get_alloc_for_range(iter);
                let range = self.ranges_hot[iter.index()].range;
                log::debug!(
                    "apply_allocations: vreg {:?} LR {:?} with range {:?} has alloc {:?}",
                    vreg,
                    iter,
                    range,
                    alloc
                );
                debug_assert!(alloc != Allocation::none());

                if log::log_enabled!(log::Level::Debug) {
                    self.annotate(
                        range.from,
                        format!(
                            " <<< start v{} in {} (LR {})",
                            vreg.index(),
                            alloc,
                            iter.index()
                        ),
                    );
                    self.annotate(
                        range.to,
                        format!(
                            "     end   v{} in {} (LR {}) >>>",
                            vreg.index(),
                            alloc,
                            iter.index()
                        ),
                    );
                }

                // Does this range follow immediately after a prior
                // range in the same block? If so, insert a move (if
                // the allocs differ). We do this directly rather than
                // with half-moves because we eagerly know both sides
                // already (and also, half-moves are specific to
                // inter-block transfers).
                //
                // Note that we do *not* do this if there is also a
                // def as the first use in the new range: it's
                // possible that an old liverange covers the Before
                // pos of an inst, a new liverange covers the After
                // pos, and the def also happens at After. In this
                // case we don't want to an insert a move after the
                // instruction copying the old liverange.
                //
                // Note also that we assert that the new range has to
                // start at the Before-point of an instruction; we
                // can't insert a move that logically happens just
                // before After (i.e. in the middle of a single
                // instruction).
                if prev.is_valid() {
                    let prev_alloc = self.get_alloc_for_range(prev);
                    let prev_range = self.ranges_hot[prev.index()].range;
                    let first_use = self.ranges[iter.index()].first_use;
                    let first_is_def = if first_use.is_valid() {
                        self.uses[first_use.index()].operand.kind() == OperandKind::Def
                    } else {
                        false
                    };
                    debug_assert!(prev_alloc != Allocation::none());
                    if prev_range.to == range.from
                        && !self.is_start_of_block(range.from)
                        && !first_is_def
                    {
                        log::debug!(
                            "prev LR {} abuts LR {} in same block; moving {} -> {} for v{}",
                            prev.index(),
                            iter.index(),
                            prev_alloc,
                            alloc,
                            vreg.index()
                        );
                        assert_eq!(range.from.pos(), InstPosition::Before);
                        self.insert_move(range.from, InsertMovePrio::Regular, prev_alloc, alloc);
                    }
                }

                // Scan over blocks whose ends are covered by this
                // range. For each, for each successor that is not
                // already in this range (hence guaranteed to have the
                // same allocation) and if the vreg is live, add a
                // Source half-move.
                let mut block = self.cfginfo.insn_block[range.from.inst().index()];
                while block.is_valid() && block.index() < self.func.blocks() {
                    if range.to < self.cfginfo.block_exit[block.index()].next() {
                        break;
                    }
                    log::debug!("examining block with end in range: block{}", block.index());
                    for &succ in self.func.block_succs(block) {
                        log::debug!(
                            " -> has succ block {} with entry {:?}",
                            succ.index(),
                            self.cfginfo.block_entry[succ.index()]
                        );
                        if range.contains_point(self.cfginfo.block_entry[succ.index()]) {
                            continue;
                        }
                        log::debug!(" -> out of this range, requires half-move if live");
                        if self.is_live_in(succ, vreg) {
                            log::debug!("  -> live at input to succ, adding halfmove");
                            half_moves.push(HalfMove {
                                key: half_move_key(block, succ, vreg, HalfMoveKind::Source),
                                alloc,
                            });
                        }
                    }

                    // Scan forward in `blockparam_outs`, adding all
                    // half-moves for outgoing values to blockparams
                    // in succs.
                    log::debug!(
                        "scanning blockparam_outs for v{} block{}: blockparam_out_idx = {}",
                        vreg.index(),
                        block.index(),
                        blockparam_out_idx,
                    );
                    while blockparam_out_idx < self.blockparam_outs.len() {
                        let (from_vreg, from_block, to_block, to_vreg) =
                            self.blockparam_outs[blockparam_out_idx];
                        if (from_vreg, from_block) > (vreg, block) {
                            break;
                        }
                        if (from_vreg, from_block) == (vreg, block) {
                            log::debug!(
                                " -> found: from v{} block{} to v{} block{}",
                                from_vreg.index(),
                                from_block.index(),
                                to_vreg.index(),
                                to_vreg.index()
                            );
                            half_moves.push(HalfMove {
                                key: half_move_key(
                                    from_block,
                                    to_block,
                                    to_vreg,
                                    HalfMoveKind::Source,
                                ),
                                alloc,
                            });
                            if log::log_enabled!(log::Level::Debug) {
                                self.annotate(
                                    self.cfginfo.block_exit[block.index()],
                                    format!(
                                        "blockparam-out: block{} to block{}: v{} to v{} in {}",
                                        from_block.index(),
                                        to_block.index(),
                                        from_vreg.index(),
                                        to_vreg.index(),
                                        alloc
                                    ),
                                );
                            }
                        }
                        blockparam_out_idx += 1;
                    }

                    block = block.next();
                }

                // Scan over blocks whose beginnings are covered by
                // this range and for which the vreg is live at the
                // start of the block. For each, for each predecessor,
                // add a Dest half-move.
                let mut block = self.cfginfo.insn_block[range.from.inst().index()];
                if self.cfginfo.block_entry[block.index()] < range.from {
                    block = block.next();
                }
                while block.is_valid() && block.index() < self.func.blocks() {
                    if self.cfginfo.block_entry[block.index()] >= range.to {
                        break;
                    }

                    // Add half-moves for blockparam inputs.
                    log::debug!(
                        "scanning blockparam_ins at vreg {} block {}: blockparam_in_idx = {}",
                        vreg.index(),
                        block.index(),
                        blockparam_in_idx
                    );
                    while blockparam_in_idx < self.blockparam_ins.len() {
                        let (to_vreg, to_block, from_block) =
                            self.blockparam_ins[blockparam_in_idx];
                        if (to_vreg, to_block) > (vreg, block) {
                            break;
                        }
                        if (to_vreg, to_block) == (vreg, block) {
                            half_moves.push(HalfMove {
                                key: half_move_key(
                                    from_block,
                                    to_block,
                                    to_vreg,
                                    HalfMoveKind::Dest,
                                ),
                                alloc,
                            });
                            log::debug!(
                                "match: blockparam_in: v{} in block{} from block{} into {}",
                                to_vreg.index(),
                                to_block.index(),
                                from_block.index(),
                                alloc,
                            );
                            if log::log_enabled!(log::Level::Debug) {
                                self.annotate(
                                    self.cfginfo.block_entry[block.index()],
                                    format!(
                                        "blockparam-in: block{} to block{}:into v{} in {}",
                                        from_block.index(),
                                        to_block.index(),
                                        to_vreg.index(),
                                        alloc
                                    ),
                                );
                            }
                        }
                        blockparam_in_idx += 1;
                    }

                    if !self.is_live_in(block, vreg) {
                        block = block.next();
                        continue;
                    }

                    log::debug!(
                        "scanning preds at vreg {} block {} for ends outside the range",
                        vreg.index(),
                        block.index()
                    );

                    // Now find any preds whose ends are not in the
                    // same range, and insert appropriate moves.
                    for &pred in self.func.block_preds(block) {
                        log::debug!(
                            "pred block {} has exit {:?}",
                            pred.index(),
                            self.cfginfo.block_exit[pred.index()]
                        );
                        if range.contains_point(self.cfginfo.block_exit[pred.index()]) {
                            continue;
                        }
                        log::debug!(" -> requires half-move");
                        half_moves.push(HalfMove {
                            key: half_move_key(pred, block, vreg, HalfMoveKind::Dest),
                            alloc,
                        });
                    }

                    block = block.next();
                }

                // If this is a blockparam vreg and the start of block
                // is in this range, add to blockparam_allocs.
                let (blockparam_block, blockparam_idx) =
                    self.cfginfo.vreg_def_blockparam[vreg.index()];
                if blockparam_block.is_valid()
                    && range.contains_point(self.cfginfo.block_entry[blockparam_block.index()])
                {
                    self.blockparam_allocs
                        .push((blockparam_block, blockparam_idx, vreg, alloc));
                }

                // Scan over def/uses and apply allocations.
                let mut use_iter = self.ranges[iter.index()].first_use;
                while use_iter.is_valid() {
                    let usedata = &self.uses[use_iter.index()];
                    log::debug!("applying to use: {:?}", usedata);
                    debug_assert!(range.contains_point(usedata.pos));
                    let inst = usedata.pos.inst();
                    let slot = usedata.slot();
                    let operand = usedata.operand;
                    // Safepoints add virtual uses with no slots;
                    // avoid these.
                    if slot != SLOT_NONE {
                        self.set_alloc(inst, slot as usize, alloc);
                    }
                    if let OperandPolicy::Reuse(_) = operand.policy() {
                        reuse_input_insts.push(inst);
                    }
                    use_iter = self.uses[use_iter.index()].next_use();
                }

                // Scan over program move srcs/dsts to fill in allocations.
                let move_src_start = if range.from.pos() == InstPosition::Before {
                    (vreg, range.from.inst())
                } else {
                    (vreg, range.from.inst().next())
                };
                let move_src_end = (vreg, range.to.inst().next());
                log::debug!(
                    "vreg {:?} range {:?}: looking for program-move sources from {:?} to {:?}",
                    vreg,
                    range,
                    move_src_start,
                    move_src_end
                );
                while prog_move_src_idx < self.prog_move_srcs.len()
                    && self.prog_move_srcs[prog_move_src_idx].0 < move_src_start
                {
                    log::debug!(" -> skipping idx {}", prog_move_src_idx);
                    prog_move_src_idx += 1;
                }
                while prog_move_src_idx < self.prog_move_srcs.len()
                    && self.prog_move_srcs[prog_move_src_idx].0 < move_src_end
                {
                    log::debug!(
                        " -> setting idx {} ({:?}) to alloc {:?}",
                        prog_move_src_idx,
                        self.prog_move_srcs[prog_move_src_idx].0,
                        alloc
                    );
                    self.prog_move_srcs[prog_move_src_idx].1 = alloc;
                    prog_move_src_idx += 1;
                }

                let move_dst_start = (vreg, range.from.inst());
                let move_dst_end = (vreg, range.to.inst());
                log::debug!(
                    "vreg {:?} range {:?}: looking for program-move dests from {:?} to {:?}",
                    vreg,
                    range,
                    move_dst_start,
                    move_dst_end
                );
                while prog_move_dst_idx < self.prog_move_dsts.len()
                    && self.prog_move_dsts[prog_move_dst_idx].0 < move_dst_start
                {
                    log::debug!(" -> skipping idx {}", prog_move_dst_idx);
                    prog_move_dst_idx += 1;
                }
                while prog_move_dst_idx < self.prog_move_dsts.len()
                    && self.prog_move_dsts[prog_move_dst_idx].0 < move_dst_end
                {
                    log::debug!(
                        " -> setting idx {} ({:?}) to alloc {:?}",
                        prog_move_dst_idx,
                        self.prog_move_dsts[prog_move_dst_idx].0,
                        alloc
                    );
                    self.prog_move_dsts[prog_move_dst_idx].1 = alloc;
                    prog_move_dst_idx += 1;
                }

                prev = iter;
                iter = self.ranges[iter.index()].next_in_reg;
            }
        }

        // Sort the half-moves list. For each (from, to,
        // from-vreg) tuple, find the from-alloc and all the
        // to-allocs, and insert moves on the block edge.
        half_moves.sort_unstable_by_key(|h| h.key);
        log::debug!("halfmoves: {:?}", half_moves);
        self.stats.halfmoves_count = half_moves.len();

        let mut i = 0;
        while i < half_moves.len() {
            // Find a Source.
            while i < half_moves.len() && half_moves[i].kind() != HalfMoveKind::Source {
                i += 1;
            }
            if i >= half_moves.len() {
                break;
            }
            let src = &half_moves[i];
            i += 1;

            // Find all Dests.
            let dest_key = src.key | 1;
            let first_dest = i;
            while i < half_moves.len() && half_moves[i].key == dest_key {
                i += 1;
            }
            let last_dest = i;

            log::debug!(
                "halfmove match: src {:?} dests {:?}",
                src,
                &half_moves[first_dest..last_dest]
            );

            // Determine the ProgPoint where moves on this (from, to)
            // edge should go:
            // - If there is more than one in-edge to `to`, then
            //   `from` must have only one out-edge; moves go at tail of
            //   `from` just before last Branch/Ret.
            // - Otherwise, there must be at most one in-edge to `to`,
            //   and moves go at start of `to`.
            let from_last_insn = self.func.block_insns(src.from_block()).last();
            let to_first_insn = self.func.block_insns(src.to_block()).first();
            let from_is_ret = self.func.is_ret(from_last_insn);
            let to_is_entry = self.func.entry_block() == src.to_block();
            let from_outs =
                self.func.block_succs(src.from_block()).len() + if from_is_ret { 1 } else { 0 };
            let to_ins =
                self.func.block_preds(src.to_block()).len() + if to_is_entry { 1 } else { 0 };

            let (insertion_point, prio) = if to_ins > 1 && from_outs <= 1 {
                (
                    // N.B.: "after" the branch should be interpreted
                    // by the user as happening before the actual
                    // branching action, but after the branch reads
                    // all necessary inputs. It's necessary to do this
                    // rather than to place the moves before the
                    // branch because the branch may have other
                    // actions than just the control-flow transfer,
                    // and these other actions may require other
                    // inputs (which should be read before the "edge"
                    // moves).
                    //
                    // Edits will only appear after the last (branch)
                    // instruction if the block has only a single
                    // successor; we do not expect the user to somehow
                    // duplicate or predicate these.
                    ProgPoint::after(from_last_insn),
                    InsertMovePrio::OutEdgeMoves,
                )
            } else if to_ins <= 1 {
                (
                    ProgPoint::before(to_first_insn),
                    InsertMovePrio::InEdgeMoves,
                )
            } else {
                panic!(
                    "Critical edge: can't insert moves between blocks {:?} and {:?}",
                    src.from_block(),
                    src.to_block()
                );
            };

            let mut last = None;
            for dest in first_dest..last_dest {
                let dest = &half_moves[dest];
                if last == Some(dest.alloc) {
                    continue;
                }
                self.insert_move(insertion_point, prio, src.alloc, dest.alloc);
                last = Some(dest.alloc);
            }
        }

        // Handle multi-fixed-reg constraints by copying.
        for (progpoint, from_preg, to_preg, slot) in
            std::mem::replace(&mut self.multi_fixed_reg_fixups, vec![])
        {
            log::debug!(
                "multi-fixed-move constraint at {:?} from p{} to p{}",
                progpoint,
                from_preg.index(),
                to_preg.index()
            );
            self.insert_move(
                progpoint,
                InsertMovePrio::MultiFixedReg,
                Allocation::reg(self.pregs[from_preg.index()].reg),
                Allocation::reg(self.pregs[to_preg.index()].reg),
            );
            self.set_alloc(
                progpoint.inst(),
                slot,
                Allocation::reg(self.pregs[to_preg.index()].reg),
            );
        }

        // Handle outputs that reuse inputs: copy beforehand, then set
        // input's alloc to output's.
        //
        // Note that the output's allocation may not *actually* be
        // valid until InstPosition::After, but the reused input may
        // occur at InstPosition::Before. This may appear incorrect,
        // but we make it work by ensuring that all *other* inputs are
        // extended to InstPosition::After so that the def will not
        // interfere. (The liveness computation code does this -- we
        // do not require the user to do so.)
        //
        // One might ask: why not insist that input-reusing defs occur
        // at InstPosition::Before? this would be correct, but would
        // mean that the reused input and the reusing output
        // interfere, *guaranteeing* that every such case would
        // require a move. This is really bad on ISAs (like x86) where
        // reused inputs are ubiquitous.
        //
        // Another approach might be to put the def at Before, and
        // trim the reused input's liverange back to the previous
        // instruction's After. This is kind of OK until (i) a block
        // boundary occurs between the prior inst and this one, or
        // (ii) any moves/spills/reloads occur between the two
        // instructions. We really do need the input to be live at
        // this inst's Before.
        //
        // In principle what we really need is a "BeforeBefore"
        // program point, but we don't want to introduce that
        // everywhere and pay the cost of twice as many ProgPoints
        // throughout the allocator.
        //
        // Or we could introduce a separate move instruction -- this
        // is the approach that regalloc.rs takes with "mod" operands
        // -- but that is also costly.
        //
        // So we take this approach (invented by IonMonkey -- somewhat
        // hard to discern, though see [0] for a comment that makes
        // this slightly less unclear) to avoid interference between
        // the actual reused input and reusing output, ensure
        // interference (hence no incorrectness) between other inputs
        // and the reusing output, and not require a separate explicit
        // move instruction.
        //
        // [0] https://searchfox.org/mozilla-central/rev/3a798ef9252896fb389679f06dd3203169565af0/js/src/jit/shared/Lowering-shared-inl.h#108-110
        for inst in reuse_input_insts {
            let mut input_reused: SmallVec<[usize; 4]> = smallvec![];
            for output_idx in 0..self.func.inst_operands(inst).len() {
                let operand = self.func.inst_operands(inst)[output_idx];
                if let OperandPolicy::Reuse(input_idx) = operand.policy() {
                    debug_assert!(!input_reused.contains(&input_idx));
                    debug_assert_eq!(operand.pos(), OperandPos::After);
                    input_reused.push(input_idx);
                    let input_alloc = self.get_alloc(inst, input_idx);
                    let output_alloc = self.get_alloc(inst, output_idx);
                    log::debug!(
                        "reuse-input inst {:?}: output {} has alloc {:?}, input {} has alloc {:?}",
                        inst,
                        output_idx,
                        output_alloc,
                        input_idx,
                        input_alloc
                    );
                    if input_alloc != output_alloc {
                        if log::log_enabled!(log::Level::Debug) {
                            self.annotate(
                                ProgPoint::before(inst),
                                format!(" reuse-input-copy: {} -> {}", input_alloc, output_alloc),
                            );
                        }
                        self.insert_move(
                            ProgPoint::before(inst),
                            InsertMovePrio::ReusedInput,
                            input_alloc,
                            output_alloc,
                        );
                        self.set_alloc(inst, input_idx, output_alloc);
                    }
                }
            }
        }

        // Sort the prog-moves lists and insert moves to reify the
        // input program's move operations.
        self.prog_move_srcs
            .sort_unstable_by_key(|((_, inst), _)| *inst);
        self.prog_move_dsts
            .sort_unstable_by_key(|((_, inst), _)| *inst);
        let prog_move_srcs = std::mem::replace(&mut self.prog_move_srcs, vec![]);
        let prog_move_dsts = std::mem::replace(&mut self.prog_move_dsts, vec![]);
        assert_eq!(prog_move_srcs.len(), prog_move_dsts.len());
        for (&((_, from_inst), from_alloc), &((_, to_inst), to_alloc)) in
            prog_move_srcs.iter().zip(prog_move_dsts.iter())
        {
            log::debug!(
                "program move at inst {:?}: alloc {:?} -> {:?}",
                from_inst,
                from_alloc,
                to_alloc
            );
            assert!(!from_alloc.is_none());
            assert!(!to_alloc.is_none());
            assert_eq!(from_inst, to_inst);
            self.insert_move(
                ProgPoint::before(from_inst),
                InsertMovePrio::ProgramMove,
                from_alloc,
                to_alloc,
            );
        }
    }

    fn resolve_inserted_moves(&mut self) {
        // For each program point, gather all moves together. Then
        // resolve (see cases below).
        let mut i = 0;
        self.inserted_moves
            .sort_unstable_by_key(|m| (m.pos.to_index(), m.prio));
        while i < self.inserted_moves.len() {
            let start = i;
            let pos = self.inserted_moves[i].pos;
            let prio = self.inserted_moves[i].prio;
            while i < self.inserted_moves.len()
                && self.inserted_moves[i].pos == pos
                && self.inserted_moves[i].prio == prio
            {
                i += 1;
            }
            let moves = &self.inserted_moves[start..i];

            // Gather all the moves with Int class and Float class
            // separately. These cannot interact, so it is safe to
            // have two separate ParallelMove instances. They need to
            // be separate because moves between the two classes are
            // impossible. (We could enhance ParallelMoves to
            // understand register classes and take multiple scratch
            // regs, but this seems simpler.)
            let mut int_moves: SmallVec<[InsertedMove; 8]> = smallvec![];
            let mut float_moves: SmallVec<[InsertedMove; 8]> = smallvec![];

            for m in moves {
                assert_eq!(m.from_alloc.class(), m.to_alloc.class());
                match m.from_alloc.class() {
                    RegClass::Int => {
                        int_moves.push(m.clone());
                    }
                    RegClass::Float => {
                        float_moves.push(m.clone());
                    }
                }
            }

            for &(regclass, moves) in
                &[(RegClass::Int, &int_moves), (RegClass::Float, &float_moves)]
            {
                // All moves in `moves` semantically happen in
                // parallel. Let's resolve these to a sequence of moves
                // that can be done one at a time.
                let mut parallel_moves = ParallelMoves::new(Allocation::reg(
                    self.env.scratch_by_class[regclass as u8 as usize],
                ));
                log::debug!("parallel moves at pos {:?} prio {:?}", pos, prio);
                for m in moves {
                    if m.from_alloc != m.to_alloc {
                        log::debug!(" {} -> {}", m.from_alloc, m.to_alloc,);
                        parallel_moves.add(m.from_alloc, m.to_alloc);
                    }
                }

                let resolved = parallel_moves.resolve();

                for (src, dst) in resolved {
                    log::debug!("  resolved: {} -> {}", src, dst);
                    self.add_edit(pos, prio, Edit::Move { from: src, to: dst });
                }
            }
        }

        // Add edits to describe blockparam locations too. This is
        // required by the checker. This comes after any edge-moves.
        self.blockparam_allocs
            .sort_unstable_by_key(|&(block, idx, _, _)| (block, idx));
        self.stats.blockparam_allocs_count = self.blockparam_allocs.len();
        let mut i = 0;
        while i < self.blockparam_allocs.len() {
            let start = i;
            let block = self.blockparam_allocs[i].0;
            while i < self.blockparam_allocs.len() && self.blockparam_allocs[i].0 == block {
                i += 1;
            }
            let params = &self.blockparam_allocs[start..i];
            let vregs = params
                .iter()
                .map(|(_, _, vreg_idx, _)| self.vreg_regs[vreg_idx.index()])
                .collect::<Vec<_>>();
            let allocs = params
                .iter()
                .map(|(_, _, _, alloc)| *alloc)
                .collect::<Vec<_>>();
            assert_eq!(vregs.len(), self.func.block_params(block).len());
            assert_eq!(allocs.len(), self.func.block_params(block).len());
            self.add_edit(
                self.cfginfo.block_entry[block.index()],
                InsertMovePrio::BlockParam,
                Edit::BlockParams { vregs, allocs },
            );
        }

        // Ensure edits are in sorted ProgPoint order. N.B.: this must
        // be a stable sort! We have to keep the order produced by the
        // parallel-move resolver for all moves within a single sort
        // key.
        self.edits.sort_by_key(|&(pos, prio, _)| (pos, prio));
        self.stats.edits_count = self.edits.len();

        // Add debug annotations.
        if log::log_enabled!(log::Level::Debug) {
            for i in 0..self.edits.len() {
                let &(pos, _, ref edit) = &self.edits[i];
                match edit {
                    &Edit::Move { from, to } => {
                        self.annotate(
                            ProgPoint::from_index(pos),
                            format!("move {} -> {}", from, to),
                        );
                    }
                    &Edit::BlockParams {
                        ref vregs,
                        ref allocs,
                    } => {
                        let s = format!("blockparams vregs:{:?} allocs:{:?}", vregs, allocs);
                        self.annotate(ProgPoint::from_index(pos), s);
                    }
                }
            }
        }
    }

    fn add_edit(&mut self, pos: ProgPoint, prio: InsertMovePrio, edit: Edit) {
        match &edit {
            &Edit::Move { from, to } if from == to => return,
            &Edit::Move { from, to } if from.is_reg() && to.is_reg() => {
                assert_eq!(from.as_reg().unwrap().class(), to.as_reg().unwrap().class());
            }
            _ => {}
        }

        self.edits.push((pos.to_index(), prio, edit));
    }

    fn compute_stackmaps(&mut self) {
        // For each ref-typed vreg, iterate through ranges and find
        // safepoints in-range. Add the SpillSlot to the stackmap.

        if self.func.reftype_vregs().is_empty() {
            return;
        }

        // Given `safepoints_per_vreg` from the liveness computation,
        // all we have to do is, for each vreg in this map, step
        // through the LiveRanges along with a sorted list of
        // safepoints; and for each safepoint in the current range,
        // emit the allocation into the `safepoint_slots` list.

        log::debug!("safepoints_per_vreg = {:?}", self.safepoints_per_vreg);

        for vreg in self.func.reftype_vregs() {
            log::debug!("generating safepoint info for vreg {}", vreg);
            let vreg = VRegIndex::new(vreg.vreg());
            let mut safepoints: Vec<ProgPoint> = self
                .safepoints_per_vreg
                .get(&vreg.index())
                .unwrap()
                .iter()
                .map(|&inst| ProgPoint::before(inst))
                .collect();
            safepoints.sort_unstable();
            log::debug!(" -> live over safepoints: {:?}", safepoints);

            let mut safepoint_idx = 0;
            let mut iter = self.vregs[vreg.index()].first_range;
            while iter.is_valid() {
                let range = self.ranges_hot[iter.index()].range;
                let alloc = self.get_alloc_for_range(iter);
                log::debug!(" -> range {:?}: alloc {}", range, alloc);
                while safepoint_idx < safepoints.len() && safepoints[safepoint_idx] < range.to {
                    if safepoints[safepoint_idx] < range.from {
                        safepoint_idx += 1;
                        continue;
                    }
                    log::debug!("    -> covers safepoint {:?}", safepoints[safepoint_idx]);

                    let slot = alloc
                        .as_stack()
                        .expect("Reference-typed value not in spillslot at safepoint");
                    self.safepoint_slots.push((safepoints[safepoint_idx], slot));
                    safepoint_idx += 1;
                }
                iter = self.ranges[iter.index()].next_in_reg;
            }
        }

        self.safepoint_slots.sort_unstable();
        log::debug!("final safepoint slots info: {:?}", self.safepoint_slots);
    }

    pub(crate) fn init(&mut self) -> Result<(), RegAllocError> {
        self.create_pregs_and_vregs();
        self.compute_liveness();
        self.compute_hot_code();
        self.merge_vreg_bundles();
        self.queue_bundles();
        if log::log_enabled!(log::Level::Debug) {
            self.dump_state();
        }
        Ok(())
    }

    pub(crate) fn run(&mut self) -> Result<(), RegAllocError> {
        self.process_bundles();
        self.try_allocating_regs_for_spilled_bundles();
        self.allocate_spillslots();
        self.apply_allocations_and_insert_moves();
        self.resolve_inserted_moves();
        self.compute_stackmaps();
        Ok(())
    }

    fn annotate(&mut self, progpoint: ProgPoint, s: String) {
        if log::log_enabled!(log::Level::Debug) {
            self.debug_annotations
                .entry(progpoint)
                .or_insert_with(|| vec![])
                .push(s);
        }
    }

    fn dump_results(&self) {
        log::debug!("=== REGALLOC RESULTS ===");
        for block in 0..self.func.blocks() {
            let block = Block::new(block);
            log::debug!(
                "block{}: [succs {:?} preds {:?}]",
                block.index(),
                self.func
                    .block_succs(block)
                    .iter()
                    .map(|b| b.index())
                    .collect::<Vec<_>>(),
                self.func
                    .block_preds(block)
                    .iter()
                    .map(|b| b.index())
                    .collect::<Vec<_>>()
            );
            for inst in self.func.block_insns(block).iter() {
                for annotation in self
                    .debug_annotations
                    .get(&ProgPoint::before(inst))
                    .map(|v| &v[..])
                    .unwrap_or(&[])
                {
                    log::debug!("  inst{}-pre: {}", inst.index(), annotation);
                }
                let ops = self
                    .func
                    .inst_operands(inst)
                    .iter()
                    .map(|op| format!("{}", op))
                    .collect::<Vec<_>>();
                let clobbers = self
                    .func
                    .inst_clobbers(inst)
                    .iter()
                    .map(|preg| format!("{}", preg))
                    .collect::<Vec<_>>();
                let allocs = (0..ops.len())
                    .map(|i| format!("{}", self.get_alloc(inst, i)))
                    .collect::<Vec<_>>();
                let opname = if self.func.is_branch(inst) {
                    "br"
                } else if self.func.is_call(inst) {
                    "call"
                } else if self.func.is_ret(inst) {
                    "ret"
                } else {
                    "op"
                };
                let args = ops
                    .iter()
                    .zip(allocs.iter())
                    .map(|(op, alloc)| format!("{} [{}]", op, alloc))
                    .collect::<Vec<_>>();
                let clobbers = if clobbers.is_empty() {
                    "".to_string()
                } else {
                    format!(" [clobber: {}]", clobbers.join(", "))
                };
                log::debug!(
                    "  inst{}: {} {}{}",
                    inst.index(),
                    opname,
                    args.join(", "),
                    clobbers
                );
                for annotation in self
                    .debug_annotations
                    .get(&ProgPoint::after(inst))
                    .map(|v| &v[..])
                    .unwrap_or(&[])
                {
                    log::debug!("  inst{}-post: {}", inst.index(), annotation);
                }
            }
        }
    }
}

pub fn run<F: Function>(func: &F, mach_env: &MachineEnv) -> Result<Output, RegAllocError> {
    let cfginfo = CFGInfo::new(func);

    let mut env = Env::new(func, mach_env, cfginfo);
    env.init()?;

    env.run()?;

    if log::log_enabled!(log::Level::Debug) {
        env.dump_results();
    }

    Ok(Output {
        edits: env
            .edits
            .into_iter()
            .map(|(pos, _, edit)| (ProgPoint::from_index(pos), edit))
            .collect(),
        allocs: env.allocs,
        inst_alloc_offsets: env.inst_alloc_offsets,
        num_spillslots: env.num_spillslots as usize,
        debug_locations: vec![],
        safepoint_slots: env.safepoint_slots,
        stats: env.stats,
    })
}
