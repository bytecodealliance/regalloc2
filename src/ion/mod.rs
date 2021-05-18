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
use std::convert::TryFrom;
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

/// Used to carry small sets of bundles, e.g. for conflict sets.
type LiveBundleVec = SmallVec<[LiveBundleIndex; 4]>;

#[derive(Clone, Copy, Debug)]
struct LiveRangeListEntry {
    range: CodeRange,
    index: LiveRangeIndex,
}

type LiveRangeList = SmallVec<[LiveRangeListEntry; 4]>;
type UseList = SmallVec<[Use; 4]>;

#[derive(Clone, Debug)]
struct LiveRange {
    range: CodeRange,

    vreg: VRegIndex,
    bundle: LiveBundleIndex,
    uses_spill_weight_and_flags: u32,

    uses: UseList,

    merged_into: LiveRangeIndex,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum LiveRangeFlag {
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
    pub fn has_flag(&self, flag: LiveRangeFlag) -> bool {
        self.uses_spill_weight_and_flags & ((flag as u32) << 29) != 0
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
struct Use {
    operand: Operand,
    pos: ProgPoint,
    slot: u8,
    weight: u16,
}

impl Use {
    #[inline(always)]
    fn new(operand: Operand, pos: ProgPoint, slot: u8) -> Self {
        Self {
            operand,
            pos,
            slot,
            // Weight is updated on insertion into LR.
            weight: 0,
        }
    }
}

const SLOT_NONE: u8 = u8::MAX;

#[derive(Clone, Debug)]
struct LiveBundle {
    ranges: LiveRangeList,
    spillset: SpillSetIndex,
    allocation: Allocation,
    prio: u32, // recomputed after every bulk update
    spill_weight_and_props: u32,
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
struct SpillSet {
    bundles: SmallVec<[LiveBundleIndex; 2]>,
    slot: SpillSlotIndex,
    reg_hint: PReg,
    class: RegClass,
    size: u8,
}

#[derive(Clone, Debug)]
struct VRegData {
    ranges: LiveRangeList,
    blockparam: Block,
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
    bundles: Vec<LiveBundle>,
    spillsets: Vec<SpillSet>,
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
fn spill_weight_from_policy(policy: OperandPolicy, is_hot: bool, is_def: bool) -> u32 {
    let hot_bonus = if is_hot { 10000 } else { 0 };
    let def_bonus = if is_def { 2000 } else { 0 };
    let policy_bonus = match policy {
        OperandPolicy::Any => 1000,
        OperandPolicy::Reg | OperandPolicy::FixedReg(_) => 2000,
        _ => 0,
    };
    hot_bonus + def_bonus + policy_bonus
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
    to_vreg: Option<VReg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum InsertMovePrio {
    InEdgeMoves,
    BlockParam,
    Regular,
    MultiFixedReg,
    ReusedInput,
    OutEdgeMoves,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    livein_blocks: usize,
    livein_iterations: usize,
    initial_liverange_count: usize,
    merged_bundle_count: usize,
    prog_moves: usize,
    prog_moves_dead_src: usize,
    prog_move_merge_attempt: usize,
    prog_move_merge_success: usize,
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
    splits_defs: usize,
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
            spillsets: Vec::with_capacity(n),
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
                    ranges: smallvec![],
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

        self.ranges.push(LiveRange {
            range,
            vreg: VRegIndex::invalid(),
            bundle: LiveBundleIndex::invalid(),
            uses_spill_weight_and_flags: 0,

            uses: smallvec![],

            merged_into: LiveRangeIndex::invalid(),
        });

        LiveRangeIndex::new(idx)
    }

    /// Mark `range` as live for the given `vreg`.
    ///
    /// Returns the liverange that contains the given range.
    fn add_liverange_to_vreg(&mut self, vreg: VRegIndex, range: CodeRange) -> LiveRangeIndex {
        log::debug!("add_liverange_to_vreg: vreg {:?} range {:?}", vreg, range);

        // When we use this function, the LR lists in VRegs are not
        // yet sorted. We can extend an existing LR if we happen to
        // see that one abuts the new range -- we check the end,
        // because this one should be the earliest given how we build
        // liveness (but we don't claim or uphold this as an
        // invariant) -- or we can just append to the end. After we
        // add all ranges, we will sort the lists.

        if let Some(last) = self.vregs[vreg.index()].ranges.last_mut() {
            if last.range.from == range.to {
                log::debug!(" -> abuts existing range {:?}, extending", last.index);
                last.range.from = range.from;
                self.ranges[last.index.index()].range.from = range.from;
                return last.index;
            }
        }

        // If we get here and did not merge into an existing liverange or liveranges, then we need
        // to create a new one.
        let lr = self.create_liverange(range);
        self.ranges[lr.index()].vreg = vreg;
        self.vregs[vreg.index()]
            .ranges
            .push(LiveRangeListEntry { range, index: lr });
        lr
    }

    fn insert_use_into_liverange(&mut self, into: LiveRangeIndex, mut u: Use) {
        let insert_pos = u.pos;
        let operand = u.operand;
        let policy = operand.policy();
        let is_hot = self
            .hot_code
            .btree
            .contains_key(&LiveRangeKey::from_range(&CodeRange {
                from: insert_pos,
                to: insert_pos.next(),
            }));
        let weight = spill_weight_from_policy(policy, is_hot, operand.kind() != OperandKind::Use);
        u.weight = u16::try_from(weight).expect("weight too large for u16 field");

        log::debug!(
            "insert use {:?} into lr {:?} with weight {}",
            u,
            into,
            weight,
        );

        self.ranges[into.index()].uses.push(u);

        // Update stats.
        self.ranges[into.index()].uses_spill_weight_and_flags += weight;
        log::debug!("  -> now {}", self.ranges[into.index()].uses_spill_weight());
    }

    fn find_vreg_liverange_for_pos(
        &self,
        vreg: VRegIndex,
        pos: ProgPoint,
    ) -> Option<LiveRangeIndex> {
        for entry in &self.vregs[vreg.index()].ranges {
            if entry.range.contains_point(pos) {
                return Some(entry.index);
            }
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

    fn compute_liveness(&mut self) -> Result<(), RegAllocError> {
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
                    live.set(dst.vreg().vreg(), false);
                    live.set(src.vreg().vreg(), true);
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

        // Check that there are no liveins to the entry block. (The
        // client should create a virtual intsruction that defines any
        // PReg liveins if necessary.)
        if self.liveins[self.func.entry_block().index()]
            .iter()
            .next()
            .is_some()
        {
            return Err(RegAllocError::EntryLivein);
        }

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
                let lr = self.add_liverange_to_vreg(VRegIndex::new(vreg), range);
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
                    // We can completely skip the move if it is
                    // trivial (vreg to same vreg) or its output is
                    // dead.
                    if src.vreg() != dst.vreg() {
                        log::debug!(" -> move inst{}: src {} -> dst {}", inst.index(), src, dst);

                        assert_eq!(src.class(), dst.class());
                        assert_eq!(src.kind(), OperandKind::Use);
                        assert_eq!(src.pos(), OperandPos::Before);
                        assert_eq!(dst.kind(), OperandKind::Def);
                        assert_eq!(dst.pos(), OperandPos::After);

                        // Redefine src and dst operands to have
                        // positions of After and Before respectively
                        // (see note below), and to have Any
                        // constraints if they were originally Reg.
                        let src_policy = match src.policy() {
                            OperandPolicy::Reg => OperandPolicy::Any,
                            x => x,
                        };
                        let dst_policy = match dst.policy() {
                            OperandPolicy::Reg => OperandPolicy::Any,
                            x => x,
                        };
                        let src = Operand::new(
                            src.vreg(),
                            src_policy,
                            OperandKind::Use,
                            OperandPos::After,
                        );
                        let dst = Operand::new(
                            dst.vreg(),
                            dst_policy,
                            OperandKind::Def,
                            OperandPos::Before,
                        );

                        if log::log_enabled!(log::Level::Debug) {
                            self.annotate(
                                ProgPoint::after(inst),
                                format!(
                                    " prog-move v{} ({:?}) -> v{} ({:?})",
                                    src.vreg().vreg(),
                                    src_policy,
                                    dst.vreg().vreg(),
                                    dst_policy,
                                ),
                            );
                        }

                        // N.B.: in order to integrate with the move
                        // resolution that joins LRs in general, we
                        // conceptually treat the move as happening
                        // between the move inst's After and the next
                        // inst's Before. Thus the src LR goes up to
                        // (exclusive) next-inst-pre, and the dst LR
                        // starts at next-inst-pre. We have to take
                        // care in our move insertion to handle this
                        // like other inter-inst moves, i.e., at
                        // `Regular` priority, so it properly happens
                        // in parallel with other inter-LR moves.
                        //
                        // Why the progpoint between move and next
                        // inst, and not the progpoint between prev
                        // inst and move? Because a move can be the
                        // first inst in a block, but cannot be the
                        // last; so the following progpoint is always
                        // within the same block, while the previous
                        // one may be an inter-block point (and the
                        // After of the prev inst in a different
                        // block).

                        // Handle the def w.r.t. liveranges: trim the
                        // start of the range and mark it dead at this
                        // point in our backward scan.
                        let pos = ProgPoint::before(inst.next());
                        let mut dst_lr = vreg_ranges[dst.vreg().vreg()];
                        if !live.get(dst.vreg().vreg()) {
                            let from = pos;
                            let to = pos.next();
                            dst_lr = self.add_liverange_to_vreg(
                                VRegIndex::new(dst.vreg().vreg()),
                                CodeRange { from, to },
                            );
                            log::debug!(" -> invalid LR for def; created {:?}", dst_lr);
                        }
                        log::debug!(" -> has existing LR {:?}", dst_lr);
                        // Trim the LR to start here.
                        if self.ranges[dst_lr.index()].range.from
                            == self.cfginfo.block_entry[block.index()]
                        {
                            log::debug!(" -> started at block start; trimming to {:?}", pos);
                            self.ranges[dst_lr.index()].range.from = pos;
                        }
                        self.ranges[dst_lr.index()].set_flag(LiveRangeFlag::StartsAtDef);
                        live.set(dst.vreg().vreg(), false);
                        vreg_ranges[dst.vreg().vreg()] = LiveRangeIndex::invalid();
                        self.vreg_regs[dst.vreg().vreg()] = dst.vreg();

                        // Handle the use w.r.t. liveranges: make it live
                        // and create an initial LR back to the start of
                        // the block.
                        let pos = ProgPoint::after(inst);
                        let range = CodeRange {
                            from: self.cfginfo.block_entry[block.index()],
                            to: pos.next(),
                        };
                        let src_lr =
                            self.add_liverange_to_vreg(VRegIndex::new(src.vreg().vreg()), range);
                        vreg_ranges[src.vreg().vreg()] = src_lr;

                        log::debug!(" -> src LR {:?}", src_lr);

                        // Add to live-set.
                        let src_is_dead_after_move = !live.get(src.vreg().vreg());
                        live.set(src.vreg().vreg(), true);

                        // Add to program-moves lists.
                        self.prog_move_srcs.push((
                            (VRegIndex::new(src.vreg().vreg()), inst),
                            Allocation::none(),
                        ));
                        self.prog_move_dsts.push((
                            (VRegIndex::new(dst.vreg().vreg()), inst.next()),
                            Allocation::none(),
                        ));
                        self.stats.prog_moves += 1;
                        if src_is_dead_after_move {
                            self.stats.prog_moves_dead_src += 1;
                            self.prog_move_merges.push((src_lr, dst_lr));
                        }
                    }

                    continue;
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
                                log::debug!("Def of {} at {:?}", operand.vreg(), pos);

                                // Fill in vreg's actual data.
                                self.vreg_regs[operand.vreg().vreg()] = operand.vreg();

                                // Get or create the LiveRange.
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
                                    );
                                    log::debug!(" -> invalid; created {:?}", lr);
                                }
                                // Create the use in the LiveRange.
                                self.insert_use_into_liverange(lr, Use::new(operand, pos, i as u8));
                                // If def (not mod), this reg is now dead,
                                // scanning backward; make it so.
                                if operand.kind() == OperandKind::Def {
                                    // Trim the range for this vreg to start
                                    // at `pos` if it previously ended at the
                                    // start of this block (i.e. was not
                                    // merged into some larger LiveRange due
                                    // to out-of-order blocks).
                                    if self.ranges[lr.index()].range.from
                                        == self.cfginfo.block_entry[block.index()]
                                    {
                                        log::debug!(
                                            " -> started at block start; trimming to {:?}",
                                            pos
                                        );
                                        self.ranges[lr.index()].range.from = pos;
                                    }

                                    self.ranges[lr.index()].set_flag(LiveRangeFlag::StartsAtDef);

                                    // Remove from live-set.
                                    live.set(operand.vreg().vreg(), false);
                                    vreg_ranges[operand.vreg().vreg()] = LiveRangeIndex::invalid();
                                }
                            }
                            OperandKind::Use => {
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
                                    );
                                    vreg_ranges[operand.vreg().vreg()] = lr;
                                }
                                assert!(lr.is_valid());

                                log::debug!("Use of {:?} at {:?} -> {:?}", operand, pos, lr,);

                                self.insert_use_into_liverange(lr, Use::new(operand, pos, i as u8));

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

        // Sort ranges in each vreg, and uses in each range, so we can
        // iterate over them in order below. The ordering invariant is
        // always maintained for uses and always for ranges in bundles
        // (which are initialized later), but not always for ranges in
        // vregs; those are sorted only when needed, here and then
        // again at the end of allocation when resolving moves.
        for vreg in &mut self.vregs {
            for entry in &mut vreg.ranges {
                // Ranges may have been truncated above at defs. We
                // need to update with the final range here.
                entry.range = self.ranges[entry.index.index()].range;
            }
            vreg.ranges.sort_unstable_by_key(|entry| entry.range.from);
        }

        for range in 0..self.ranges.len() {
            self.ranges[range].uses.sort_unstable_by_key(|u| u.pos);
        }

        // Insert safepoint virtual stack uses, if needed.
        for vreg in self.func.reftype_vregs() {
            let vreg = VRegIndex::new(vreg.vreg());
            let mut inserted = false;
            let mut safepoint_idx = 0;
            for range_idx in 0..self.vregs[vreg.index()].ranges.len() {
                let LiveRangeListEntry { range, index } =
                    self.vregs[vreg.index()].ranges[range_idx];
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

                    log::debug!(
                        "Safepoint-induced stack use of {:?} at {:?} -> {:?}",
                        operand,
                        pos,
                        index,
                    );

                    self.insert_use_into_liverange(index, Use::new(operand, pos, SLOT_NONE));
                    safepoint_idx += 1;

                    inserted = true;
                }

                if inserted {
                    self.ranges[index.index()]
                        .uses
                        .sort_unstable_by_key(|u| u.pos);
                }

                if safepoint_idx >= self.safepoints.len() {
                    break;
                }
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
            for range_idx in 0..self.vregs[vreg].ranges.len() {
                let entry = self.vregs[vreg].ranges[range_idx];
                let range = entry.index;
                log::debug!(
                    "multi-fixed-reg cleanup: vreg {:?} range {:?}",
                    VRegIndex::new(vreg),
                    range,
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

                for u in &mut self.ranges[range.index()].uses {
                    let pos = u.pos;
                    let slot = u.slot as usize;
                    fixup_multi_fixed_vregs(
                        pos,
                        slot,
                        &mut u.operand,
                        &mut self.multi_fixed_reg_fixups,
                    );
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

        Ok(())
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
            ranges: smallvec![],
            spillset: SpillSetIndex::invalid(),
            prio: 0,
            spill_weight_and_props: 0,
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

        // Both bundles must deal with the same RegClass.
        let from_rc = self.spillsets[self.bundles[from.index()].spillset.index()].class;
        let to_rc = self.spillsets[self.bundles[to.index()].spillset.index()].class;
        if from_rc != to_rc {
            log::debug!(" -> mismatching reg classes");
            return false;
        }

        // If either bundle is already assigned (due to a pinned vreg), don't merge.
        if !self.bundles[from.index()].allocation.is_none()
            || !self.bundles[to.index()].allocation.is_none()
        {
            log::debug!("one of the bundles is already assigned (pinned)");
            return false;
        }

        #[cfg(debug)]
        {
            // Sanity check: both bundles should contain only ranges with appropriate VReg classes.
            for entry in &self.bundles[from.index()].ranges {
                let vreg = self.ranges[entry.index.index()].vreg;
                assert_eq!(rc, self.vregs[vreg.index()].reg.class());
            }
            for entry in &self.bundles[to.index()].ranges {
                let vreg = self.ranges[entry.index.index()].vreg;
                assert_eq!(rc, self.vregs[vreg.index()].reg.class());
            }
        }

        // Check for overlap in LiveRanges.
        let ranges_from = &self.bundles[from.index()].ranges[..];
        let ranges_to = &self.bundles[to.index()].ranges[..];
        let mut idx_from = 0;
        let mut idx_to = 0;
        let mut range_count = 0;
        while idx_from < ranges_from.len() && idx_to < ranges_to.len() {
            range_count += 1;
            if range_count > 200 {
                log::debug!(
                    "reached merge complexity (range_count = {}); exiting",
                    range_count
                );
                // Limit merge complexity.
                return false;
            }

            if ranges_from[idx_from].range.from >= ranges_to[idx_to].range.to {
                idx_to += 1;
            } else if ranges_to[idx_to].range.from >= ranges_from[idx_from].range.to {
                idx_from += 1;
            } else {
                // Overlap -- cannot merge.
                log::debug!(
                    " -> overlap between {:?} and {:?}, exiting",
                    ranges_from[idx_from].index,
                    ranges_to[idx_to].index
                );
                return false;
            }
        }

        log::debug!(" -> committing to merge");

        // If we reach here, then the bundles do not overlap -- merge
        // them!  We do this with a merge-sort-like scan over both
        // lists, building a new range list and replacing the list on
        // `to` when we're done.
        let mut idx_from = 0;
        let mut idx_to = 0;
        if ranges_from.is_empty() {
            // `from` bundle is empty -- trivial merge.
            log::debug!(" -> from bundle{} is empty; trivial merge", from.index());
            return true;
        }
        if ranges_to.is_empty() {
            // `to` bundle is empty -- just move the list over from
            // `from` and set `bundle` up-link on all ranges.
            log::debug!(" -> to bundle{} is empty; trivial merge", to.index());
            let list = std::mem::replace(&mut self.bundles[from.index()].ranges, smallvec![]);
            for entry in &list {
                self.ranges[entry.index.index()].bundle = to;

                if log::log_enabled!(log::Level::Debug) {
                    self.annotate(
                        entry.range.from,
                        format!(
                            " MERGE range{} v{} from bundle{} to bundle{}",
                            entry.index.index(),
                            self.ranges[entry.index.index()].vreg.index(),
                            from.index(),
                            to.index(),
                        ),
                    );
                }
            }
            self.bundles[to.index()].ranges = list;

            return true;
        }

        // Two non-empty lists of LiveRanges: traverse both simultaneously and
        // merge ranges into `merged`.
        let mut merged: LiveRangeList = smallvec![];
        while idx_from < ranges_from.len() || idx_to < ranges_to.len() {
            if idx_from < ranges_from.len() && idx_to < ranges_to.len() {
                if ranges_from[idx_from].range.from <= ranges_to[idx_to].range.from {
                    merged.push(ranges_from[idx_from]);
                    idx_from += 1;
                } else {
                    merged.push(ranges_to[idx_to]);
                    idx_to += 1;
                }
            } else if idx_from < ranges_from.len() {
                merged.extend_from_slice(&ranges_from[idx_from..]);
                break;
            } else {
                assert!(idx_to < ranges_to.len());
                merged.extend_from_slice(&ranges_to[idx_to..]);
                break;
            }
        }
        for entry in &merged {
            if self.ranges[entry.index.index()].bundle == from {
                if log::log_enabled!(log::Level::Debug) {
                    self.annotate(
                        entry.range.from,
                        format!(
                            " MERGE range{} v{} from bundle{} to bundle{}",
                            entry.index.index(),
                            self.ranges[entry.index.index()].vreg.index(),
                            from.index(),
                            to.index(),
                        ),
                    );
                }
            }

            log::debug!(
                " -> merged result for bundle{}: range{}",
                to.index(),
                entry.index.index(),
            );
            self.ranges[entry.index.index()].bundle = to;
        }

        self.bundles[to.index()].ranges = merged;
        self.bundles[from.index()].ranges.clear();

        true
    }

    fn merge_vreg_bundles(&mut self) {
        // Create a bundle for every vreg, initially.
        log::debug!("merge_vreg_bundles: creating vreg bundles");
        for vreg in 0..self.vregs.len() {
            let vreg = VRegIndex::new(vreg);
            if self.vregs[vreg.index()].ranges.is_empty() {
                continue;
            }
            let bundle = self.create_bundle();
            self.bundles[bundle.index()].ranges = self.vregs[vreg.index()].ranges.clone();
            log::debug!("vreg v{} gets bundle{}", vreg.index(), bundle.index());
            for entry in &self.bundles[bundle.index()].ranges {
                log::debug!(" -> with LR range{}", entry.index.index());
                self.ranges[entry.index.index()].bundle = bundle;
            }

            // Create a spillslot for this bundle.
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
                        self.ranges[self.vregs[src_vreg.vreg()].ranges[0].index.index()].bundle;
                    assert!(src_bundle.is_valid());
                    let dest_bundle =
                        self.ranges[self.vregs[dst_vreg.vreg()].ranges[0].index.index()].bundle;
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
            let to_bundle = self.ranges[self.vregs[to_vreg.index()].ranges[0].index.index()].bundle;
            assert!(to_bundle.is_valid());
            let from_bundle =
                self.ranges[self.vregs[from_vreg.index()].ranges[0].index.index()].bundle;
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
            self.stats.prog_move_merge_attempt += 1;
            if self.merge_bundles(/* from */ dest_bundle, /* to */ src_bundle) {
                self.stats.prog_move_merge_success += 1;
            }
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
        let mut total = 0;
        for entry in &self.bundles[bundle.index()].ranges {
            total += entry.range.len() as u32;
        }
        total
    }

    fn queue_bundles(&mut self) {
        for bundle in 0..self.bundles.len() {
            log::debug!("enqueueing bundle{}", bundle);
            if self.bundles[bundle].ranges.is_empty() {
                log::debug!(" -> no ranges; skipping");
                continue;
            }
            let bundle = LiveBundleIndex::new(bundle);
            let prio = self.compute_bundle_prio(bundle);
            log::debug!(" -> prio {}", prio);
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
                "bundle{}: spillset={:?} alloc={:?}",
                i,
                b.spillset,
                b.allocation
            );
            for entry in &b.ranges {
                log::debug!(
                    " * range {:?} -- {:?}: range{}",
                    entry.range.from,
                    entry.range.to,
                    entry.index.index()
                );
            }
        }
        log::debug!("VRegs:");
        for (i, v) in self.vregs.iter().enumerate() {
            log::debug!("vreg{}:", i);
            for entry in &v.ranges {
                log::debug!(
                    " * range {:?} -- {:?}: range{}",
                    entry.range.from,
                    entry.range.to,
                    entry.index.index()
                );
            }
        }
        log::debug!("Ranges:");
        for (i, r) in self.ranges.iter().enumerate() {
            log::debug!(
                concat!("range{}: range={:?} vreg={:?} bundle={:?} ", "weight={}"),
                i,
                r.range,
                r.vreg,
                r.bundle,
                r.uses_spill_weight(),
            );
            for u in &r.uses {
                log::debug!(" * use at {:?} (slot {}): {:?}", u.pos, u.slot, u.operand);
            }
        }
    }

    fn compute_requirement(&self, bundle: LiveBundleIndex) -> Option<Requirement> {
        log::debug!("compute_requirement: bundle {:?}", bundle);

        let class = self.spillsets[self.bundles[bundle.index()].spillset.index()].class;
        log::debug!(" -> class = {:?}", class);

        let mut needed = Requirement::Any(class);

        for entry in &self.bundles[bundle.index()].ranges {
            let range = &self.ranges[entry.index.index()];
            log::debug!(
                " -> range LR {} ({:?}): {:?}",
                entry.index.index(),
                entry.index,
                entry.range
            );
            for u in &range.uses {
                let use_req = Requirement::from_operand(u.operand);
                log::debug!(
                    " -> use at {:?} op {:?} req {:?}",
                    u.pos,
                    u.operand,
                    use_req
                );
                needed = needed.merge(use_req)?;
                log::debug!("   -> needed {:?}", needed);
            }
        }

        log::debug!(" -> final needed: {:?}", needed);
        Some(needed)
    }

    fn try_to_allocate_bundle_to_reg(
        &mut self,
        bundle: LiveBundleIndex,
        reg: PRegIndex,
    ) -> AllocRegResult {
        log::debug!("try_to_allocate_bundle_to_reg: {:?} -> {:?}", bundle, reg);
        let mut conflicts = smallvec![];
        // Traverse the BTreeMap in order by requesting the whole
        // range spanned by the bundle and iterating over that
        // concurrently with our ranges. Because our ranges are in
        // order, and the BTreeMap is as well, this allows us to have
        // an overall O(n log n) + O(b) complexity, where the PReg has
        // n current ranges and the bundle has b ranges, rather than
        // O(b * n log n) with the simple probe-for-each-bundle-range
        // approach.
        //
        // Note that the comparator function on a CodeRange tests for *overlap*, so we
        // are checking whether the BTree contains any preg range that
        // *overlaps* with range `range`, not literally the range `range`.
        let bundle_ranges = &self.bundles[bundle.index()].ranges;
        let from_key = LiveRangeKey::from_range(&bundle_ranges.first().unwrap().range);
        let to_key = LiveRangeKey::from_range(&bundle_ranges.last().unwrap().range);
        assert!(from_key <= to_key);
        let mut preg_range_iter = self.pregs[reg.index()]
            .allocations
            .btree
            .range(from_key..=to_key)
            .peekable();
        log::debug!(
            "alloc map for {:?}: {:?}",
            reg,
            self.pregs[reg.index()].allocations.btree
        );
        for entry in bundle_ranges {
            log::debug!(" -> range LR {:?}: {:?}", entry.index, entry.range);
            let key = LiveRangeKey::from_range(&entry.range);

            // Advance our BTree traversal until it is >= this bundle
            // range (i.e., skip PReg allocations in the BTree that
            // are completely before this bundle range).

            while preg_range_iter.peek().is_some() && *preg_range_iter.peek().unwrap().0 < key {
                log::debug!(
                    "Skipping PReg range {:?}",
                    preg_range_iter.peek().unwrap().0
                );
                preg_range_iter.next();
            }

            // If there are no more PReg allocations, we're done!
            if preg_range_iter.peek().is_none() {
                log::debug!(" -> no more PReg allocations; so no conflict possible!");
                break;
            }

            // If the current PReg range is beyond this range, there is no conflict; continue.
            if *preg_range_iter.peek().unwrap().0 > key {
                log::debug!(
                    " -> next PReg allocation is at {:?}; moving to next VReg range",
                    preg_range_iter.peek().unwrap().0
                );
                continue;
            }

            // Otherwise, there is a conflict.
            assert_eq!(*preg_range_iter.peek().unwrap().0, key);
            let preg_range = preg_range_iter.next().unwrap().1;

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
            } else {
                log::debug!("   -> conflict with fixed reservation");
                // range from a direct use of the PReg (due to clobber).
                return AllocRegResult::ConflictWithFixed;
            }
        }

        if conflicts.len() > 0 {
            return AllocRegResult::Conflict(conflicts);
        }

        // We can allocate! Add our ranges to the preg's BTree.
        let preg = self.pregs[reg.index()].reg;
        log::debug!("  -> bundle {:?} assigned to preg {:?}", bundle, preg);
        self.bundles[bundle.index()].allocation = Allocation::reg(preg);
        for entry in &self.bundles[bundle.index()].ranges {
            self.pregs[reg.index()]
                .allocations
                .btree
                .insert(LiveRangeKey::from_range(&entry.range), entry.index);
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
        for entry in &self.bundles[bundle.index()].ranges {
            log::debug!(" -> removing LR {:?} from reg {:?}", entry.index, preg_idx);
            self.pregs[preg_idx.index()]
                .allocations
                .btree
                .remove(&LiveRangeKey::from_range(&entry.range));
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
        log::debug!("recompute bundle properties: bundle {:?}", bundle);

        let minimal;
        let mut fixed = false;
        let bundledata = &self.bundles[bundle.index()];
        let first_range = bundledata.ranges[0].index;
        let first_range_data = &self.ranges[first_range.index()];

        if first_range_data.vreg.is_invalid() {
            log::debug!("  -> no vreg; minimal and fixed");
            minimal = true;
            fixed = true;
        } else {
            for u in &first_range_data.uses {
                if let OperandPolicy::FixedReg(_) = u.operand.policy() {
                    log::debug!("  -> fixed use at {:?}: {:?}", u.pos, u.operand);
                    fixed = true;
                    break;
                }
            }
            // Minimal if this is the only range in the bundle, and if
            // the range covers only one instruction. Note that it
            // could cover just one ProgPoint, i.e. X.Before..X.After,
            // or two ProgPoints, i.e. X.Before..X+1.Before.
            log::debug!("  -> first range has range {:?}", first_range_data.range);
            minimal = self.bundles[bundle.index()].ranges.len() == 1
                && first_range_data.range.from.inst() == first_range_data.range.to.prev().inst();
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
            for entry in &self.bundles[bundle.index()].ranges {
                let range_data = &self.ranges[entry.index.index()];
                log::debug!(
                    "  -> uses spill weight: +{}",
                    range_data.uses_spill_weight()
                );
                total += range_data.uses_spill_weight();
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

        let mut def_splits: SmallVec<[ProgPoint; 4]> = smallvec![];
        let mut seen_defs = 0;
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
        let our_ranges = &self.bundles[bundle.index()].ranges[..];
        let (conflict_from, conflict_to) = if conflicting.is_valid() {
            (
                Some(
                    self.bundles[conflicting.index()]
                        .ranges
                        .first()
                        .unwrap()
                        .range
                        .from,
                ),
                Some(
                    self.bundles[conflicting.index()]
                        .ranges
                        .last()
                        .unwrap()
                        .range
                        .to,
                ),
            )
        } else {
            (None, None)
        };

        let bundle_start = if our_ranges.is_empty() {
            ProgPoint::before(Inst::new(0))
        } else {
            our_ranges.first().unwrap().range.from
        };
        let bundle_end = if our_ranges.is_empty() {
            ProgPoint::before(Inst::new(self.func.insts()))
        } else {
            our_ranges.last().unwrap().range.to
        };

        log::debug!(" -> conflict from {:?} to {:?}", conflict_from, conflict_to);
        let mut clobberidx = 0;
        for entry in our_ranges {
            // Probe the hot-code tree.
            log::debug!(" -> range {:?}", entry.range);
            if let Some(hot_range_idx) = self
                .hot_code
                .btree
                .get(&LiveRangeKey::from_range(&entry.range))
            {
                // `hot_range_idx` is a range that *overlaps* with our range.

                // There may be cold code in our range on either side of the hot
                // range. Record the transition points if so.
                let hot_range = self.ranges[hot_range_idx.index()].range;
                log::debug!("   -> overlaps with hot-code range {:?}", hot_range);
                let start_cold = entry.range.from < hot_range.from;
                let end_cold = entry.range.to > hot_range.to;
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
                if pos >= entry.range.to {
                    break;
                }
                clobberidx += 1;
                if pos < entry.range.from {
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

            for u in &self.ranges[entry.index.index()].uses {
                log::debug!("   -> range has use at {:?}", u.pos);
                update_with_pos(u.pos);
                if u.operand.kind() == OperandKind::Def {
                    if seen_defs > 0 {
                        def_splits.push(u.pos);
                    }
                    seen_defs += 1;
                }
            }
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
        } else if def_splits.len() > 0 {
            log::debug!(" going with non-first def splits: {:?}", def_splits);
            self.stats.splits_defs += 1;
            def_splits
        } else {
            self.stats.splits_all += 1;
            log::debug!(" splitting at all uses");
            self.find_all_use_split_points(bundle)
        }
    }

    fn find_all_use_split_points(&self, bundle: LiveBundleIndex) -> SmallVec<[ProgPoint; 4]> {
        let mut splits = smallvec![];
        let ranges = &self.bundles[bundle.index()].ranges[..];
        log::debug!("finding all use/def splits for {:?}", bundle);
        let bundle_start = if ranges.is_empty() {
            ProgPoint::before(Inst::new(0))
        } else {
            self.ranges[ranges[0].index.index()].range.from
        };
        // N.B.: a minimal bundle must include only ProgPoints in a
        // single instruction, but can include both (can include two
        // ProgPoints). We split here, taking care to never split *in
        // the middle* of an instruction, because we would not be able
        // to insert moves to reify such an assignment.
        for entry in ranges {
            log::debug!(" -> range {:?}: {:?}", entry.index, entry.range);
            for u in &self.ranges[entry.index.index()].uses {
                log::debug!("  -> use: {:?}", u);
                let before_use_inst = if u.operand.kind() == OperandKind::Def {
                    // For a def, split *at* the def -- this may be an
                    // After point, but the value cannot be live into
                    // the def so we don't need to insert a move.
                    u.pos
                } else {
                    // For an use or mod, split before the instruction
                    // -- this allows us to insert a move if
                    // necessary.
                    ProgPoint::before(u.pos.inst())
                };
                let after_use_inst = ProgPoint::before(u.pos.inst().next());
                log::debug!(
                    "  -> splitting before and after use: {:?} and {:?}",
                    before_use_inst,
                    after_use_inst,
                );
                if before_use_inst > bundle_start {
                    splits.push(before_use_inst);
                }
                splits.push(after_use_inst);
            }
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

        let spillset = self.bundles[bundle.index()].spillset;

        let mut split_idx = 0;

        // Fast-forward past any splits that occur before or exactly
        // at the start of the first range in the bundle.
        let bundle_start = if self.bundles[bundle.index()].ranges.is_empty() {
            ProgPoint::before(Inst::new(0))
        } else {
            self.bundles[bundle.index()].ranges[0].range.from
        };
        while split_idx < split_points.len() && split_points[split_idx] <= bundle_start {
            split_idx += 1;
        }

        let mut new_bundles: LiveBundleVec = smallvec![];
        let mut cur_bundle = bundle;
        let ranges = std::mem::replace(&mut self.bundles[bundle.index()].ranges, smallvec![]);
        // - Invariant: current LR `cur_lr` is being built; it has not
        //   yet been added to `cur_bundle`.
        // - Invariant: uses in `cur_uses` have not yet been added to
        //   `cur_lr`.
        for entry in &ranges {
            log::debug!(" -> has range {:?} (LR {:?})", entry.range, entry.index);

            // Until we reach a split point, copy or create the current range.
            let mut cur_range = entry.range;
            let mut cur_lr = entry.index;
            let mut cur_uses =
                std::mem::replace(&mut self.ranges[cur_lr.index()].uses, smallvec![]);
            let mut cur_uses = cur_uses.drain(..).peekable();

            self.ranges[cur_lr.index()].uses_spill_weight_and_flags = 0;

            let update_lr_stats = |lr: &mut LiveRange, u: &Use| {
                if lr.uses.is_empty() && u.operand.kind() == OperandKind::Def {
                    lr.set_flag(LiveRangeFlag::StartsAtDef);
                }
                lr.uses_spill_weight_and_flags += u.weight as u32;
            };

            while cur_range.to > cur_range.from {
                if (split_idx >= split_points.len()) || (split_points[split_idx] >= cur_range.to) {
                    log::debug!(
                        " -> no more split points; placing all remaining uses into cur range{}",
                        cur_lr.index()
                    );
                    // No more split points left, or next split point
                    // is beyond the range: just copy the current
                    // range into the current bundle, and drop all the
                    // remaining uses into it.
                    for u in cur_uses {
                        update_lr_stats(&mut self.ranges[cur_lr.index()], &u);
                        log::debug!("  -> use at {:?}", u.pos);
                        self.ranges[cur_lr.index()].uses.push(u);
                    }
                    self.ranges[cur_lr.index()].bundle = cur_bundle;
                    self.bundles[cur_bundle.index()]
                        .ranges
                        .push(LiveRangeListEntry {
                            range: cur_range,
                            index: cur_lr,
                        });
                    break;
                }

                // If there is a split point prior to or exactly at
                // the start of this LR, then create a new bundle but
                // keep the existing LR, and go around again. Skip all
                // such split-points (lump them into one), while we're
                // at it.
                if split_points[split_idx] <= cur_range.from {
                    log::debug!(
                        " -> split point at {:?} before start of range (range {:?} LR {:?})",
                        split_points[split_idx],
                        cur_range,
                        cur_lr,
                    );
                    cur_bundle = self.create_bundle();
                    log::debug!(" -> new bundle {:?}", cur_bundle);
                    self.ranges[cur_lr.index()].bundle = cur_bundle;
                    new_bundles.push(cur_bundle);
                    self.bundles[cur_bundle.index()].spillset = spillset;
                    while split_idx < split_points.len()
                        && split_points[split_idx] <= cur_range.from
                    {
                        split_idx += 1;
                    }
                    continue;
                }

                // If we reach here, there is at least one split point
                // that lands in the current range, so we need to
                // actually split. Let's create a new LR and bundle
                // for the rest (post-split-point), drop uses up to
                // the split point into current LR and drop current LR
                // into current bundle, then advance current LR and
                // bundle to new LR and bundle.
                let split = split_points[split_idx];
                while split_idx < split_points.len() && split_points[split_idx] == split {
                    // Skip past all duplicate split-points.
                    split_idx += 1;
                }
                log::debug!(" -> split at {:?}", split);

                let existing_range = CodeRange {
                    from: cur_range.from,
                    to: split,
                };
                let new_range = CodeRange {
                    from: split,
                    to: cur_range.to,
                };
                let new_lr = self.create_liverange(new_range);
                let new_bundle = self.create_bundle();
                log::debug!(" -> new LR {:?}, new bundle {:?}", new_lr, new_bundle);
                new_bundles.push(new_bundle);
                self.bundles[new_bundle.index()].spillset = spillset;

                self.ranges[cur_lr.index()].range = existing_range;
                self.ranges[new_lr.index()].vreg = self.ranges[cur_lr.index()].vreg;
                self.ranges[new_lr.index()].bundle = new_bundle;
                self.ranges[cur_lr.index()].bundle = cur_bundle;
                self.bundles[cur_bundle.index()]
                    .ranges
                    .push(LiveRangeListEntry {
                        range: existing_range,
                        index: cur_lr,
                    });
                while let Some(u) = cur_uses.peek() {
                    if u.pos >= split {
                        break;
                    }
                    update_lr_stats(&mut self.ranges[cur_lr.index()], &u);
                    log::debug!(" -> use at {:?} in current LR {:?}", u.pos, cur_lr);
                    self.ranges[cur_lr.index()].uses.push(*u);
                    cur_uses.next();
                }

                self.annotate(
                    existing_range.to,
                    format!(
                        " SPLIT range{} v{} bundle{} to range{} bundle{}",
                        cur_lr.index(),
                        self.ranges[cur_lr.index()].vreg.index(),
                        cur_bundle.index(),
                        new_lr.index(),
                        new_bundle.index(),
                    ),
                );

                cur_range = new_range;
                cur_bundle = new_bundle;
                cur_lr = new_lr;

                // Perform a lazy split in the VReg data. We just
                // append the new LR and its range; we will sort by
                // start of range, and fix up range ends, once when we
                // iterate over the VReg's ranges after allocation
                // completes (this is the only time when order
                // matters).
                self.vregs[self.ranges[new_lr.index()].vreg.index()]
                    .ranges
                    .push(LiveRangeListEntry {
                        range: new_range,
                        index: new_lr,
                    });
            }
        }

        // Recompute weights and priorities of all bundles, and
        // enqueue all split-bundles on the allocation queue.
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
                    let scan_offset = self.ranges
                        [self.bundles[bundle.index()].ranges[0].index.index()]
                    .range
                    .from
                    .inst()
                    .index()
                        + bundle.index();

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
        log::debug!("allocating regs for spilled bundles");
        for i in 0..self.spilled_bundles.len() {
            let bundle = self.spilled_bundles[i]; // don't borrow self
            let any_vreg = self.vreg_regs[self.ranges
                [self.bundles[bundle.index()].ranges[0].index.index()]
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
                log::debug!("trying bundle {:?} to preg {:?}", bundle, preg);
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
            for entry in &self.bundles[bundle.index()].ranges {
                if self.spillslots[spillslot.index()]
                    .ranges
                    .btree
                    .contains_key(&LiveRangeKey::from_range(&entry.range))
                {
                    return false;
                }
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
            for entry in &self.bundles[bundle.index()].ranges {
                log::debug!(
                    "spillslot {:?} getting range {:?} from bundle {:?}: {:?}",
                    spillslot,
                    entry.range,
                    entry.index,
                    bundle,
                );
                self.spillslots[spillslot.index()]
                    .ranges
                    .btree
                    .insert(LiveRangeKey::from_range(&entry.range), entry.index);
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
        to_vreg: Option<VReg>,
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
            to_vreg,
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
        log::debug!("get_alloc_for_range: {:?}", range);
        let bundle = self.ranges[range.index()].bundle;
        log::debug!(" -> bundle: {:?}", bundle);
        let bundledata = &self.bundles[bundle.index()];
        log::debug!(" -> allocation {:?}", bundledata.allocation);
        if bundledata.allocation != Allocation::none() {
            bundledata.allocation
        } else {
            log::debug!(" -> spillset {:?}", bundledata.spillset);
            log::debug!(
                " -> spill slot {:?}",
                self.spillsets[bundledata.spillset.index()].slot
            );
            self.spillslots[self.spillsets[bundledata.spillset.index()].slot.index()].alloc
        }
    }

    fn apply_allocations_and_insert_moves(&mut self) {
        log::debug!("apply_allocations_and_insert_moves");
        log::debug!("blockparam_ins: {:?}", self.blockparam_ins);
        log::debug!("blockparam_outs: {:?}", self.blockparam_outs);

        // Now that all splits are done, we can pay the cost once to
        // sort VReg range lists and update with the final ranges.
        for vreg in &mut self.vregs {
            for entry in &mut vreg.ranges {
                entry.range = self.ranges[entry.index.index()].range;
            }
            vreg.ranges.sort_unstable_by_key(|entry| entry.range.from);
        }

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
            let mut prev = LiveRangeIndex::invalid();
            for range_idx in 0..self.vregs[vreg.index()].ranges.len() {
                let entry = self.vregs[vreg.index()].ranges[range_idx];
                let alloc = self.get_alloc_for_range(entry.index);
                let range = entry.range;
                log::debug!(
                    "apply_allocations: vreg {:?} LR {:?} with range {:?} has alloc {:?}",
                    vreg,
                    entry.index,
                    range,
                    alloc
                );
                debug_assert!(alloc != Allocation::none());

                if log::log_enabled!(log::Level::Debug) {
                    self.annotate(
                        range.from,
                        format!(
                            " <<< start v{} in {} (range{}) (bundle{})",
                            vreg.index(),
                            alloc,
                            entry.index.index(),
                            self.ranges[entry.index.index()].bundle.index(),
                        ),
                    );
                    self.annotate(
                        range.to,
                        format!(
                            "     end   v{} in {} (range{}) (bundle{}) >>>",
                            vreg.index(),
                            alloc,
                            entry.index.index(),
                            self.ranges[entry.index.index()].bundle.index(),
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
                    let prev_range = self.ranges[prev.index()].range;
                    let first_is_def =
                        self.ranges[entry.index.index()].has_flag(LiveRangeFlag::StartsAtDef);
                    debug_assert!(prev_alloc != Allocation::none());
                    if prev_range.to == range.from
                        && !self.is_start_of_block(range.from)
                        && !first_is_def
                    {
                        log::debug!(
                            "prev LR {} abuts LR {} in same block; moving {} -> {} for v{}",
                            prev.index(),
                            entry.index.index(),
                            prev_alloc,
                            alloc,
                            vreg.index()
                        );
                        assert_eq!(range.from.pos(), InstPosition::Before);
                        self.insert_move(
                            range.from,
                            InsertMovePrio::Regular,
                            prev_alloc,
                            alloc,
                            None,
                        );
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
                for use_idx in 0..self.ranges[entry.index.index()].uses.len() {
                    let usedata = self.ranges[entry.index.index()].uses[use_idx];
                    log::debug!("applying to use: {:?}", usedata);
                    debug_assert!(range.contains_point(usedata.pos));
                    let inst = usedata.pos.inst();
                    let slot = usedata.slot;
                    let operand = usedata.operand;
                    // Safepoints add virtual uses with no slots;
                    // avoid these.
                    if slot != SLOT_NONE {
                        self.set_alloc(inst, slot as usize, alloc);
                    }
                    if let OperandPolicy::Reuse(_) = operand.policy() {
                        reuse_input_insts.push(inst);
                    }
                }

                // Scan over program move srcs/dsts to fill in allocations.

                // Move srcs happen at `After` of a given
                // inst. Compute [from, to) semi-inclusive range of
                // inst indices for which we should fill in the source
                // with this LR's allocation.
                //
                // range from inst-Before or inst-After covers cur
                // inst's After; so includes move srcs from inst.
                let move_src_start = (vreg, range.from.inst());
                // range to (exclusive) inst-Before or inst-After
                // covers only prev inst's After; so includes move
                // srcs to (exclusive) inst.
                let move_src_end = (vreg, range.to.inst());
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

                // move dsts happen at Before point.
                //
                // Range from inst-Before includes cur inst, while inst-After includes only next inst.
                let move_dst_start = if range.from.pos() == InstPosition::Before {
                    (vreg, range.from.inst())
                } else {
                    (vreg, range.from.inst().next())
                };
                // Range to (exclusive) inst-Before includes prev
                // inst, so to (exclusive) cur inst; range to
                // (exclusive) inst-After includes cur inst, so to
                // (exclusive) next inst.
                let move_dst_end = if range.to.pos() == InstPosition::Before {
                    (vreg, range.to.inst())
                } else {
                    (vreg, range.to.inst().next())
                };
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

                prev = entry.index;
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
                    // N.B.: though semantically the edge moves happen
                    // after the branch, we must insert them before
                    // the branch because otherwise, of course, they
                    // would never execute. This is correct even in
                    // the presence of branches that read register
                    // inputs (e.g. conditional branches on some RISCs
                    // that branch on reg zero/not-zero, or any
                    // indirect branch), but for a very subtle reason:
                    // all cases of such branches will (or should)
                    // have multiple successors, and thus due to
                    // critical-edge splitting, their successors will
                    // have only the single predecessor, and we prefer
                    // to insert at the head of the successor in that
                    // case (rather than here). We make this a
                    // requirement, in fact: the user of this library
                    // shall not read registers in a branch
                    // instruction of there is only one successor per
                    // the given CFG information.
                    ProgPoint::before(from_last_insn),
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
                self.insert_move(insertion_point, prio, src.alloc, dest.alloc, None);
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
                None,
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
                            None,
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
            .sort_unstable_by_key(|((_, inst), _)| inst.prev());
        let prog_move_srcs = std::mem::replace(&mut self.prog_move_srcs, vec![]);
        let prog_move_dsts = std::mem::replace(&mut self.prog_move_dsts, vec![]);
        assert_eq!(prog_move_srcs.len(), prog_move_dsts.len());
        for (&((_, from_inst), from_alloc), &((to_vreg, to_inst), to_alloc)) in
            prog_move_srcs.iter().zip(prog_move_dsts.iter())
        {
            log::debug!(
                "program move at inst {:?}: alloc {:?} -> {:?} (v{})",
                from_inst,
                from_alloc,
                to_alloc,
                to_vreg.index(),
            );
            assert!(!from_alloc.is_none());
            assert!(!to_alloc.is_none());
            assert_eq!(from_inst, to_inst.prev());
            // N.B.: these moves happen with the *same* priority as
            // LR-to-LR moves, because they work just like them: they
            // connect a use at one progpoint (move-After) with a def
            // at an adjacent progpoint (move+1-Before), so they must
            // happen in parallel with all other LR-to-LR moves.
            self.insert_move(
                ProgPoint::before(to_inst),
                InsertMovePrio::Regular,
                from_alloc,
                to_alloc,
                Some(self.vreg_regs[to_vreg.index()]),
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
                if m.from_alloc.is_reg() && m.to_alloc.is_reg() {
                    assert_eq!(m.from_alloc.class(), m.to_alloc.class());
                }
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
                    if (m.from_alloc != m.to_alloc) || m.to_vreg.is_some() {
                        log::debug!(" {} -> {}", m.from_alloc, m.to_alloc,);
                        parallel_moves.add(m.from_alloc, m.to_alloc, m.to_vreg);
                    }
                }

                let resolved = parallel_moves.resolve();

                for (src, dst, to_vreg) in resolved {
                    log::debug!("  resolved: {} -> {} ({:?})", src, dst, to_vreg);
                    self.add_edit(
                        pos,
                        prio,
                        Edit::Move {
                            from: src,
                            to: dst,
                            to_vreg,
                        },
                    );
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
                    &Edit::Move { from, to, to_vreg } => {
                        self.annotate(
                            ProgPoint::from_index(pos),
                            format!("move {} -> {} ({:?})", from, to, to_vreg),
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
            &Edit::Move { from, to, to_vreg } if from == to && to_vreg.is_none() => return,
            &Edit::Move { from, to, .. } if from.is_reg() && to.is_reg() => {
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
            for entry in &self.vregs[vreg.index()].ranges {
                let range = entry.range;
                let alloc = self.get_alloc_for_range(entry.index);
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
            }
        }

        self.safepoint_slots.sort_unstable();
        log::debug!("final safepoint slots info: {:?}", self.safepoint_slots);
    }

    pub(crate) fn init(&mut self) -> Result<(), RegAllocError> {
        self.create_pregs_and_vregs();
        self.compute_hot_code();
        self.compute_liveness()?;
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
    let cfginfo = CFGInfo::new(func)?;

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
