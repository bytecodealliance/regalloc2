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
 * - "Fixed-stack location": negative spillslot numbers?
 *
 * - Rematerialization
 */

/*
   Performance and code-quality ideas:

   - Reduced spilling when spillslot is still "clean":
     - Track 'dirty' status of reg and elide spill when not dirty.
       - This is slightly tricky: fixpoint problem, across edges.
       - We can simplify by assuming spillslot is dirty if value came
         in on BB edge; only clean if we reload in same block we spill
         in.
       - As a slightly better variation on this, track dirty during
         scan in a single range while resolving moves; in-edge makes
         dirty.

   - Avoid requiring two scratch regs:
     - Require machine impl to be able to (i) push a reg, (ii) pop a
       reg; then generate a balanced pair of push/pop, using the stack
       slot as the scratch.
       - on Cranelift side, take care to generate virtual-SP
         adjustments!
     - For a spillslot->spillslot move, push a fixed reg (say the
       first preferred one), reload into it, spill out of it, and then
       pop old val

   - Avoid rebuilding MachineEnv on every function allocation in
     regalloc.rs shim

   - Profile allocations
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
use fxhash::{FxHashMap, FxHashSet};
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
type UseList = SmallVec<[Use; 2]>;

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
    fn set_cached_spill_weight_and_props(
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
    fn cached_minimal(&self) -> bool {
        self.spill_weight_and_props & (1 << 31) != 0
    }

    #[inline(always)]
    fn cached_fixed(&self) -> bool {
        self.spill_weight_and_props & (1 << 30) != 0
    }

    #[inline(always)]
    fn cached_stack(&self) -> bool {
        self.spill_weight_and_props & (1 << 29) != 0
    }

    #[inline(always)]
    fn cached_spill_weight(&self) -> u32 {
        self.spill_weight_and_props & ((1 << 29) - 1)
    }
}

#[derive(Clone, Debug)]
struct SpillSet {
    vregs: SmallVec<[VRegIndex; 2]>,
    slot: SpillSlotIndex,
    reg_hint: PReg,
    class: RegClass,
    spill_bundle: LiveBundleIndex,
    required: bool,
    size: u8,
}

#[derive(Clone, Debug)]
struct VRegData {
    ranges: LiveRangeList,
    blockparam: Block,
    is_ref: bool,
    is_pinned: bool,
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
    annotations_enabled: bool,
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
    reg_hint: PReg,
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

    #[inline(always)]
    fn insert(&mut self, bundle: LiveBundleIndex, prio: usize, reg_hint: PReg) {
        self.heap.push(PrioQueueEntry {
            prio: prio as u32,
            bundle,
            reg_hint,
        });
    }

    #[inline(always)]
    fn is_empty(self) -> bool {
        self.heap.is_empty()
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<(LiveBundleIndex, PReg)> {
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

#[inline(always)]
fn spill_weight_from_policy(policy: OperandPolicy, loop_depth: usize, is_def: bool) -> u32 {
    // A bonus of 1000 for one loop level, 4000 for two loop levels,
    // 16000 for three loop levels, etc. Avoids exponentiation.
    let hot_bonus = std::cmp::min(16000, 1000 * (1 << (2 * loop_depth)));
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
    Unknown,
    Fixed(PReg),
    Register(RegClass),
    Stack(RegClass),
    Any(RegClass),
    Conflict,
}
impl Requirement {
    #[inline(always)]
    fn class(self) -> RegClass {
        match self {
            Requirement::Unknown => panic!("No class for unknown Requirement"),
            Requirement::Fixed(preg) => preg.class(),
            Requirement::Register(class) | Requirement::Any(class) | Requirement::Stack(class) => {
                class
            }
            Requirement::Conflict => panic!("No class for conflicted Requirement"),
        }
    }
    #[inline(always)]
    fn merge(self, other: Requirement) -> Requirement {
        match (self, other) {
            (Requirement::Unknown, other) | (other, Requirement::Unknown) => other,
            (Requirement::Conflict, _) | (_, Requirement::Conflict) => Requirement::Conflict,
            (other, Requirement::Any(rc)) | (Requirement::Any(rc), other) => {
                if other.class() == rc {
                    other
                } else {
                    Requirement::Conflict
                }
            }
            (Requirement::Stack(rc1), Requirement::Stack(rc2)) => {
                if rc1 == rc2 {
                    self
                } else {
                    Requirement::Conflict
                }
            }
            (Requirement::Register(rc), Requirement::Fixed(preg))
            | (Requirement::Fixed(preg), Requirement::Register(rc)) => {
                if rc == preg.class() {
                    Requirement::Fixed(preg)
                } else {
                    Requirement::Conflict
                }
            }
            (Requirement::Register(rc1), Requirement::Register(rc2)) => {
                if rc1 == rc2 {
                    self
                } else {
                    Requirement::Conflict
                }
            }
            (Requirement::Fixed(a), Requirement::Fixed(b)) if a == b => self,
            _ => Requirement::Conflict,
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
    ConflictWithFixed(u32, ProgPoint),
    ConflictHighCost,
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
    offset_pref: usize,
    offset_non_pref: usize,
    is_fixed: bool,
    fixed: Option<PReg>,
}

impl<'a> RegTraversalIter<'a> {
    pub fn new(
        env: &'a MachineEnv,
        class: RegClass,
        hint_reg: PReg,
        hint2_reg: PReg,
        offset: usize,
        fixed: Option<PReg>,
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
        let class = class as u8 as usize;
        let offset_pref = if env.preferred_regs_by_class[class].len() > 0 {
            offset % env.preferred_regs_by_class[class].len()
        } else {
            0
        };
        let offset_non_pref = if env.non_preferred_regs_by_class[class].len() > 0 {
            offset % env.non_preferred_regs_by_class[class].len()
        } else {
            0
        };
        Self {
            env,
            class,
            hints,
            hint_idx: 0,
            pref_idx: 0,
            non_pref_idx: 0,
            offset_pref,
            offset_non_pref,
            is_fixed: fixed.is_some(),
            fixed,
        }
    }
}

impl<'a> std::iter::Iterator for RegTraversalIter<'a> {
    type Item = PReg;

    fn next(&mut self) -> Option<PReg> {
        if self.is_fixed {
            let ret = self.fixed;
            self.fixed = None;
            return ret;
        }

        fn wrap(idx: usize, limit: usize) -> usize {
            if idx >= limit {
                idx - limit
            } else {
                idx
            }
        }
        if self.hint_idx < 2 && self.hints[self.hint_idx].is_some() {
            let h = self.hints[self.hint_idx];
            self.hint_idx += 1;
            return h;
        }
        while self.pref_idx < self.env.preferred_regs_by_class[self.class].len() {
            let arr = &self.env.preferred_regs_by_class[self.class][..];
            let r = arr[wrap(self.pref_idx + self.offset_pref, arr.len())];
            self.pref_idx += 1;
            if Some(r) == self.hints[0] || Some(r) == self.hints[1] {
                continue;
            }
            return Some(r);
        }
        while self.non_pref_idx < self.env.non_preferred_regs_by_class[self.class].len() {
            let arr = &self.env.non_preferred_regs_by_class[self.class][..];
            let r = arr[wrap(self.non_pref_idx + self.offset_non_pref, arr.len())];
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
    pub(crate) fn new(
        func: &'a F,
        env: &'a MachineEnv,
        cfginfo: CFGInfo,
        annotations_enabled: bool,
    ) -> Self {
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
            annotations_enabled,
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
                    is_pinned: false,
                },
            );
        }
        for v in self.func.reftype_vregs() {
            self.vregs[v.vreg()].is_ref = true;
        }
        for v in self.func.pinned_vregs() {
            self.vregs[v.vreg()].is_pinned = true;
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

        // Invariant: as we are building liveness information, we
        // *always* process instructions bottom-to-top, and as a
        // consequence, new liveranges are always created before any
        // existing liveranges for a given vreg. We assert this here,
        // then use it to avoid an O(n) merge step (which would lead
        // to O(n^2) liveness construction cost overall).
        //
        // We store liveranges in reverse order in the `.ranges`
        // array, then reverse them at the end of
        // `compute_liveness()`.

        assert!(
            self.vregs[vreg.index()].ranges.is_empty()
                || range.to
                    <= self.ranges[self.vregs[vreg.index()]
                        .ranges
                        .last()
                        .unwrap()
                        .index
                        .index()]
                    .range
                    .from
        );

        if self.vregs[vreg.index()].ranges.is_empty()
            || range.to
                < self.ranges[self.vregs[vreg.index()]
                    .ranges
                    .last()
                    .unwrap()
                    .index
                    .index()]
                .range
                .from
        {
            // Is not contiguous with previously-added (immediately
            // following) range; create a new range.
            let lr = self.create_liverange(range);
            self.ranges[lr.index()].vreg = vreg;
            self.vregs[vreg.index()]
                .ranges
                .push(LiveRangeListEntry { range, index: lr });
            lr
        } else {
            // Is contiguous with previously-added range; just extend
            // its range and return it.
            let lr = self.vregs[vreg.index()].ranges.last().unwrap().index;
            assert!(range.to == self.ranges[lr.index()].range.from);
            self.ranges[lr.index()].range.from = range.from;
            lr
        }
    }

    fn insert_use_into_liverange(&mut self, into: LiveRangeIndex, mut u: Use) {
        let operand = u.operand;
        let policy = operand.policy();
        let block = self.cfginfo.insn_block[u.pos.inst().index()];
        let loop_depth = self.cfginfo.approx_loop_depth[block.index()] as usize;
        let weight =
            spill_weight_from_policy(policy, loop_depth, operand.kind() != OperandKind::Use);
        u.weight = u16::try_from(weight).expect("weight too large for u16 field");

        log::debug!(
            "insert use {:?} into lr {:?} with weight {}",
            u,
            into,
            weight,
        );

        // N.B.: we do *not* update `requirement` on the range,
        // because those will be computed during the multi-fixed-reg
        // fixup pass later (after all uses are inserted).

        self.ranges[into.index()].uses.push(u);

        // Update stats.
        self.ranges[into.index()].uses_spill_weight_and_flags += weight;
        log::debug!(
            "  -> now range has weight {}",
            self.ranges[into.index()].uses_spill_weight(),
        );
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
        self.pregs[preg_idx.index()]
            .allocations
            .btree
            .insert(LiveRangeKey::from_range(&range), LiveRangeIndex::invalid());
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
            log::debug!(" -> initial liveout set: {:?}", live);

            for inst in self.func.block_insns(block).rev().iter() {
                if let Some((src, dst)) = self.func.is_move(inst) {
                    live.set(dst.vreg().vreg(), false);
                    live.set(src.vreg().vreg(), true);
                }

                for pos in &[OperandPos::After, OperandPos::Before] {
                    for op in self.func.inst_operands(inst) {
                        if op.pos() == *pos {
                            let was_live = live.get(op.vreg().vreg());
                            log::debug!("op {:?} was_live = {}", op, was_live);
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
            log::debug!(
                "non-empty liveins to entry block: {:?}",
                self.liveins[self.func.entry_block().index()]
            );
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
                    // trivial (vreg to same vreg).
                    if src.vreg() != dst.vreg() {
                        log::debug!(" -> move inst{}: src {} -> dst {}", inst.index(), src, dst);

                        assert_eq!(src.class(), dst.class());
                        assert_eq!(src.kind(), OperandKind::Use);
                        assert_eq!(src.pos(), OperandPos::Before);
                        assert_eq!(dst.kind(), OperandKind::Def);
                        assert_eq!(dst.pos(), OperandPos::After);

                        // If both src and dest are pinned, emit the
                        // move right here, right now.
                        if self.vregs[src.vreg().vreg()].is_pinned
                            && self.vregs[dst.vreg().vreg()].is_pinned
                        {
                            // Update LRs.
                            if !live.get(src.vreg().vreg()) {
                                let lr = self.add_liverange_to_vreg(
                                    VRegIndex::new(src.vreg().vreg()),
                                    CodeRange {
                                        from: self.cfginfo.block_entry[block.index()],
                                        to: ProgPoint::after(inst),
                                    },
                                );
                                live.set(src.vreg().vreg(), true);
                                vreg_ranges[src.vreg().vreg()] = lr;
                            }
                            if live.get(dst.vreg().vreg()) {
                                let lr = vreg_ranges[dst.vreg().vreg()];
                                self.ranges[lr.index()].range.from = ProgPoint::after(inst);
                                live.set(dst.vreg().vreg(), false);
                            } else {
                                self.add_liverange_to_vreg(
                                    VRegIndex::new(dst.vreg().vreg()),
                                    CodeRange {
                                        from: ProgPoint::after(inst),
                                        to: ProgPoint::before(inst.next()),
                                    },
                                );
                            }

                            let src_preg = match src.policy() {
                                OperandPolicy::FixedReg(r) => r,
                                _ => unreachable!(),
                            };
                            let dst_preg = match dst.policy() {
                                OperandPolicy::FixedReg(r) => r,
                                _ => unreachable!(),
                            };
                            self.insert_move(
                                ProgPoint::before(inst),
                                InsertMovePrio::MultiFixedReg,
                                Allocation::reg(src_preg),
                                Allocation::reg(dst_preg),
                                Some(dst.vreg()),
                            );
                        }
                        // If exactly one of source and dest (but not
                        // both) is a pinned-vreg, convert this into a
                        // ghost use on the other vreg with a FixedReg
                        // policy.
                        else if self.vregs[src.vreg().vreg()].is_pinned
                            || self.vregs[dst.vreg().vreg()].is_pinned
                        {
                            log::debug!(
                                " -> exactly one of src/dst is pinned; converting to ghost use"
                            );
                            let (preg, vreg, pinned_vreg, kind, pos, progpoint) =
                                if self.vregs[src.vreg().vreg()].is_pinned {
                                    // Source is pinned: this is a def on the dst with a pinned preg.
                                    (
                                        self.func.is_pinned_vreg(src.vreg()).unwrap(),
                                        dst.vreg(),
                                        src.vreg(),
                                        OperandKind::Def,
                                        OperandPos::After,
                                        ProgPoint::after(inst),
                                    )
                                } else {
                                    // Dest is pinned: this is a use on the src with a pinned preg.
                                    (
                                        self.func.is_pinned_vreg(dst.vreg()).unwrap(),
                                        src.vreg(),
                                        dst.vreg(),
                                        OperandKind::Use,
                                        OperandPos::Before,
                                        ProgPoint::after(inst),
                                    )
                                };
                            let policy = OperandPolicy::FixedReg(preg);
                            let operand = Operand::new(vreg, policy, kind, pos);

                            log::debug!(
                                concat!(
                                    " -> preg {:?} vreg {:?} kind {:?} ",
                                    "pos {:?} progpoint {:?} policy {:?} operand {:?}"
                                ),
                                preg,
                                vreg,
                                kind,
                                pos,
                                progpoint,
                                policy,
                                operand
                            );

                            // Get the LR for the vreg; if none, create one.
                            let mut lr = vreg_ranges[vreg.vreg()];
                            if !live.get(vreg.vreg()) {
                                let from = match kind {
                                    OperandKind::Use => self.cfginfo.block_entry[block.index()],
                                    OperandKind::Def => progpoint,
                                    _ => unreachable!(),
                                };
                                let to = progpoint.next();
                                lr = self.add_liverange_to_vreg(
                                    VRegIndex::new(vreg.vreg()),
                                    CodeRange { from, to },
                                );
                                log::debug!("   -> dead; created LR");
                            }
                            log::debug!("  -> LR {:?}", lr);

                            self.insert_use_into_liverange(
                                lr,
                                Use::new(operand, progpoint, SLOT_NONE),
                            );

                            if kind == OperandKind::Def {
                                live.set(vreg.vreg(), false);
                                if self.ranges[lr.index()].range.from
                                    == self.cfginfo.block_entry[block.index()]
                                {
                                    self.ranges[lr.index()].range.from = progpoint;
                                }
                                self.ranges[lr.index()].set_flag(LiveRangeFlag::StartsAtDef);
                            } else {
                                live.set(vreg.vreg(), true);
                                vreg_ranges[vreg.vreg()] = lr;
                            }

                            // Handle liveness of the other vreg. Note
                            // that this is somewhat special. For the
                            // destination case, we want the pinned
                            // vreg's LR to start just *after* the
                            // operand we inserted above, because
                            // otherwise it would overlap, and
                            // interfere, and prevent allocation. For
                            // the source case, we want to "poke a
                            // hole" in the LR: if it's live going
                            // downward, end it just after the operand
                            // and restart it before; if it isn't
                            // (this is the last use), start it
                            // before.
                            if kind == OperandKind::Def {
                                log::debug!(" -> src on pinned vreg {:?}", pinned_vreg);
                                // The *other* vreg is a def, so the pinned-vreg
                                // mention is a use. If already live,
                                // end the existing LR just *after*
                                // the `progpoint` defined above and
                                // start a new one just *before* the
                                // `progpoint` defined above,
                                // preserving the start. If not, start
                                // a new one live back to the top of
                                // the block, starting just before
                                // `progpoint`.
                                if live.get(pinned_vreg.vreg()) {
                                    let pinned_lr = vreg_ranges[pinned_vreg.vreg()];
                                    let orig_start = self.ranges[pinned_lr.index()].range.from;
                                    log::debug!(
                                        " -> live with LR {:?}; truncating to start at {:?}",
                                        pinned_lr,
                                        progpoint.next()
                                    );
                                    self.ranges[pinned_lr.index()].range.from = progpoint.next();
                                    let new_lr = self.add_liverange_to_vreg(
                                        VRegIndex::new(pinned_vreg.vreg()),
                                        CodeRange {
                                            from: orig_start,
                                            to: progpoint.prev(),
                                        },
                                    );
                                    vreg_ranges[pinned_vreg.vreg()] = new_lr;
                                    log::debug!(" -> created LR {:?} with remaining range from {:?} to {:?}", new_lr, orig_start, progpoint);

                                    // Add an edit right now to indicate that at
                                    // this program point, the given
                                    // preg is now known as that vreg,
                                    // not the preg, but immediately
                                    // after, it is known as the preg
                                    // again. This is used by the
                                    // checker.
                                    self.add_edit(
                                        ProgPoint::before(inst),
                                        InsertMovePrio::MultiFixedReg,
                                        Edit::DefAlloc {
                                            alloc: Allocation::reg(preg),
                                            vreg: dst.vreg(),
                                        },
                                    );
                                    self.add_edit(
                                        ProgPoint::after(inst),
                                        InsertMovePrio::Regular,
                                        Edit::DefAlloc {
                                            alloc: Allocation::reg(preg),
                                            vreg: src.vreg(),
                                        },
                                    );
                                } else {
                                    if inst > self.cfginfo.block_entry[block.index()].inst() {
                                        let new_lr = self.add_liverange_to_vreg(
                                            VRegIndex::new(pinned_vreg.vreg()),
                                            CodeRange {
                                                from: self.cfginfo.block_entry[block.index()],
                                                to: ProgPoint::before(inst),
                                            },
                                        );
                                        vreg_ranges[pinned_vreg.vreg()] = new_lr;
                                        live.set(pinned_vreg.vreg(), true);
                                        log::debug!(
                                            " -> was not live; created new LR {:?}",
                                            new_lr
                                        );
                                    }

                                    // Add an edit right now to indicate that at
                                    // this program point, the given
                                    // preg is now known as that vreg,
                                    // not the preg. This is used by
                                    // the checker.
                                    self.add_edit(
                                        ProgPoint::after(inst),
                                        InsertMovePrio::Regular,
                                        Edit::DefAlloc {
                                            alloc: Allocation::reg(preg),
                                            vreg: dst.vreg(),
                                        },
                                    );
                                }
                            } else {
                                log::debug!(" -> dst on pinned vreg {:?}", pinned_vreg);
                                // The *other* vreg is a use, so the pinned-vreg
                                // mention is a def. Truncate its LR
                                // just *after* the `progpoint`
                                // defined above.
                                if live.get(pinned_vreg.vreg()) {
                                    let pinned_lr = vreg_ranges[pinned_vreg.vreg()];
                                    self.ranges[pinned_lr.index()].range.from = progpoint.next();
                                    log::debug!(
                                        " -> was live with LR {:?}; truncated start to {:?}",
                                        pinned_lr,
                                        progpoint.next()
                                    );
                                    live.set(pinned_vreg.vreg(), false);

                                    // Add a no-op edit right now to indicate that
                                    // at this program point, the
                                    // given preg is now known as that
                                    // preg, not the vreg. This is
                                    // used by the checker.
                                    self.add_edit(
                                        ProgPoint::after(inst),
                                        InsertMovePrio::Regular,
                                        Edit::DefAlloc {
                                            alloc: Allocation::reg(preg),
                                            vreg: dst.vreg(),
                                        },
                                    );
                                }
                                // Otherwise, if dead, no need to create
                                // a dummy LR -- there is no
                                // reservation to make (the other vreg
                                // will land in the reg with the
                                // fixed-reg operand constraint, but
                                // it's a dead move anyway).
                            }
                        } else {
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

                            if self.annotations_enabled {
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
                            let src_lr = if !live.get(src.vreg().vreg()) {
                                let range = CodeRange {
                                    from: self.cfginfo.block_entry[block.index()],
                                    to: pos.next(),
                                };
                                let src_lr = self.add_liverange_to_vreg(
                                    VRegIndex::new(src.vreg().vreg()),
                                    range,
                                );
                                vreg_ranges[src.vreg().vreg()] = src_lr;
                                src_lr
                            } else {
                                vreg_ranges[src.vreg().vreg()]
                            };

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
                                    vreg_ranges[operand.vreg().vreg()] = lr;
                                    live.set(operand.vreg().vreg(), true);
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

        // Make ranges in each vreg and uses in each range appear in
        // sorted order. We built them in reverse order above, so this
        // is a simple reversal, *not* a full sort.
        //
        // The ordering invariant is always maintained for uses and
        // always for ranges in bundles (which are initialized later),
        // but not always for ranges in vregs; those are sorted only
        // when needed, here and then again at the end of allocation
        // when resolving moves.

        for vreg in &mut self.vregs {
            vreg.ranges.reverse();
            let mut last = None;
            for entry in &mut vreg.ranges {
                // Ranges may have been truncated above at defs. We
                // need to update with the final range here.
                entry.range = self.ranges[entry.index.index()].range;
                // Assert in-order and non-overlapping.
                assert!(last.is_none() || last.unwrap() <= entry.range.from);
                last = Some(entry.range.to);
            }
        }

        for range in 0..self.ranges.len() {
            self.ranges[range].uses.reverse();
            debug_assert!(self.ranges[range]
                .uses
                .windows(2)
                .all(|win| win[0].pos <= win[1].pos));
        }

        // Insert safepoint virtual stack uses, if needed.
        for vreg in self.func.reftype_vregs() {
            if self.vregs[vreg.vreg()].is_pinned {
                continue;
            }
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

        // Check for overlap in LiveRanges and for conflicting
        // requirements.
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

        // Check for a requirements conflict.
        let req = self
            .compute_requirement(from)
            .merge(self.compute_requirement(to));
        if req == Requirement::Conflict {
            log::debug!(" -> conflicting requirements; aborting merge");
            return false;
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

                if self.annotations_enabled {
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
        log::debug!(
            "merging: ranges_from = {:?} ranges_to = {:?}",
            ranges_from,
            ranges_to
        );
        while idx_from < ranges_from.len() || idx_to < ranges_to.len() {
            if idx_from < ranges_from.len() && idx_to < ranges_to.len() {
                if ranges_from[idx_from].range.from <= ranges_to[idx_to].range.from {
                    self.ranges[ranges_from[idx_from].index.index()].bundle = to;
                    merged.push(ranges_from[idx_from]);
                    idx_from += 1;
                } else {
                    merged.push(ranges_to[idx_to]);
                    idx_to += 1;
                }
            } else if idx_from < ranges_from.len() {
                for entry in &ranges_from[idx_from..] {
                    self.ranges[entry.index.index()].bundle = to;
                }
                merged.extend_from_slice(&ranges_from[idx_from..]);
                break;
            } else {
                assert!(idx_to < ranges_to.len());
                merged.extend_from_slice(&ranges_to[idx_to..]);
                break;
            }
        }

        #[cfg(debug_assertions)]
        {
            log::debug!("merging: merged = {:?}", merged);
            let mut last_range = None;
            for entry in &merged {
                if last_range.is_some() {
                    assert!(last_range.unwrap() < entry.range);
                }
                last_range = Some(entry.range);

                if self.ranges[entry.index.index()].bundle == from {
                    if self.annotations_enabled {
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
            }
        }

        self.bundles[to.index()].ranges = merged;
        self.bundles[from.index()].ranges.clear();

        if self.bundles[from.index()].spillset != self.bundles[to.index()].spillset {
            let from_vregs = std::mem::replace(
                &mut self.spillsets[self.bundles[from.index()].spillset.index()].vregs,
                smallvec![],
            );
            let to_vregs = &mut self.spillsets[self.bundles[to.index()].spillset.index()].vregs;
            for vreg in from_vregs {
                if !to_vregs.contains(&vreg) {
                    to_vregs.push(vreg);
                }
            }
        }

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

            // If this is a pinned vreg, go ahead and add it to the
            // commitment map, and avoid creating a bundle entirely.
            if self.vregs[vreg.index()].is_pinned {
                for entry in &self.vregs[vreg.index()].ranges {
                    let preg = self
                        .func
                        .is_pinned_vreg(self.vreg_regs[vreg.index()])
                        .unwrap();
                    let key = LiveRangeKey::from_range(&entry.range);
                    self.pregs[preg.index()]
                        .allocations
                        .btree
                        .insert(key, LiveRangeIndex::invalid());
                }
                continue;
            }

            let bundle = self.create_bundle();
            self.bundles[bundle.index()].ranges = self.vregs[vreg.index()].ranges.clone();
            log::debug!("vreg v{} gets bundle{}", vreg.index(), bundle.index());
            for entry in &self.bundles[bundle.index()].ranges {
                log::debug!(
                    " -> with LR range{}: {:?}",
                    entry.index.index(),
                    entry.range
                );
                self.ranges[entry.index.index()].bundle = bundle;
            }

            // Create a spillslot for this bundle.
            let ssidx = SpillSetIndex::new(self.spillsets.len());
            let reg = self.vreg_regs[vreg.index()];
            let size = self.func.spillslot_size(reg.class(), reg) as u8;
            self.spillsets.push(SpillSet {
                vregs: smallvec![vreg],
                slot: SpillSlotIndex::invalid(),
                size,
                required: false,
                class: reg.class(),
                reg_hint: PReg::invalid(),
                spill_bundle: LiveBundleIndex::invalid(),
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
                    if self.vregs[src_vreg.vreg()].is_pinned
                        || self.vregs[dst_vreg.vreg()].is_pinned
                    {
                        continue;
                    }

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

            let dst_vreg = self.vreg_regs[self.ranges[dst.index()].vreg.index()];
            let src_vreg = self.vreg_regs[self.ranges[src.index()].vreg.index()];
            if self.vregs[src_vreg.vreg()].is_pinned && self.vregs[dst_vreg.vreg()].is_pinned {
                continue;
            }
            if self.vregs[src_vreg.vreg()].is_pinned {
                let dest_bundle = self.ranges[dst.index()].bundle;
                let spillset = self.bundles[dest_bundle.index()].spillset;
                self.spillsets[spillset.index()].reg_hint =
                    self.func.is_pinned_vreg(src_vreg).unwrap();
                continue;
            }
            if self.vregs[dst_vreg.vreg()].is_pinned {
                let src_bundle = self.ranges[src.index()].bundle;
                let spillset = self.bundles[src_bundle.index()].spillset;
                self.spillsets[spillset.index()].reg_hint =
                    self.func.is_pinned_vreg(dst_vreg).unwrap();
                continue;
            }

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
            self.allocation_queue
                .insert(bundle, prio as usize, PReg::invalid());
        }
        self.stats.merged_bundle_count = self.allocation_queue.heap.len();
    }

    fn process_bundles(&mut self) -> Result<(), RegAllocError> {
        let mut count = 0;
        while let Some((bundle, reg_hint)) = self.allocation_queue.pop() {
            self.stats.process_bundle_count += 1;
            self.process_bundle(bundle, reg_hint)?;
            count += 1;
            if count > self.func.insts() * 50 {
                self.dump_state();
                panic!("Infinite loop!");
            }
        }
        self.stats.final_liverange_count = self.ranges.len();
        self.stats.final_bundle_count = self.bundles.len();
        self.stats.spill_bundle_count = self.spilled_bundles.len();

        Ok(())
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
                "range{}: range={:?} vreg={:?} bundle={:?} weight={}",
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

    fn try_to_allocate_bundle_to_reg(
        &mut self,
        bundle: LiveBundleIndex,
        reg: PRegIndex,
        // if the max bundle weight in the conflict set exceeds this
        // cost (if provided), just return
        // `AllocRegResult::ConflictHighCost`.
        max_allowable_cost: Option<u32>,
    ) -> AllocRegResult {
        log::debug!("try_to_allocate_bundle_to_reg: {:?} -> {:?}", bundle, reg);
        let mut conflicts = smallvec![];
        let mut max_conflict_weight = 0;
        // Traverse the BTreeMap in order by requesting the whole
        // range spanned by the bundle and iterating over that
        // concurrently with our ranges. Because our ranges are in
        // order, and the BTreeMap is as well, this allows us to have
        // an overall O(n log n) + O(b) complexity, where the PReg has
        // n current ranges and the bundle has b ranges, rather than
        // O(b * n log n) with the simple probe-for-each-bundle-range
        // approach.
        //
        // Note that the comparator function on a CodeRange tests for
        // *overlap*, so we are checking whether the BTree contains
        // any preg range that *overlaps* with range `range`, not
        // literally the range `range`.
        let bundle_ranges = &self.bundles[bundle.index()].ranges;
        let from_key = LiveRangeKey::from_range(&CodeRange {
            from: bundle_ranges.first().unwrap().range.from,
            to: bundle_ranges.first().unwrap().range.from,
        });
        let mut preg_range_iter = self.pregs[reg.index()]
            .allocations
            .btree
            .range(from_key..)
            .peekable();
        log::debug!(
            "alloc map for {:?} in range {:?}..: {:?}",
            reg,
            from_key,
            self.pregs[reg.index()].allocations.btree
        );
        'ranges: for entry in bundle_ranges {
            log::debug!(" -> range LR {:?}: {:?}", entry.index, entry.range);
            let key = LiveRangeKey::from_range(&entry.range);

            'alloc: loop {
                log::debug!("  -> PReg range {:?}", preg_range_iter.peek());

                // Advance our BTree traversal until it is >= this bundle
                // range (i.e., skip PReg allocations in the BTree that
                // are completely before this bundle range).

                if preg_range_iter.peek().is_some() && *preg_range_iter.peek().unwrap().0 < key {
                    log::debug!(
                        "Skipping PReg range {:?}",
                        preg_range_iter.peek().unwrap().0
                    );
                    preg_range_iter.next();
                    continue 'alloc;
                }

                // If there are no more PReg allocations, we're done!
                if preg_range_iter.peek().is_none() {
                    log::debug!(" -> no more PReg allocations; so no conflict possible!");
                    break 'ranges;
                }

                // If the current PReg range is beyond this range, there is no conflict; continue.
                if *preg_range_iter.peek().unwrap().0 > key {
                    log::debug!(
                        " -> next PReg allocation is at {:?}; moving to next VReg range",
                        preg_range_iter.peek().unwrap().0
                    );
                    break 'alloc;
                }

                // Otherwise, there is a conflict.
                let preg_key = *preg_range_iter.peek().unwrap().0;
                assert_eq!(preg_key, key); // Assert that this range overlaps.
                let preg_range = preg_range_iter.next().unwrap().1;

                log::debug!(" -> btree contains range {:?} that overlaps", preg_range);
                if preg_range.is_valid() {
                    log::debug!("   -> from vreg {:?}", self.ranges[preg_range.index()].vreg);
                    // range from an allocated bundle: find the bundle and add to
                    // conflicts list.
                    let conflict_bundle = self.ranges[preg_range.index()].bundle;
                    log::debug!("   -> conflict bundle {:?}", conflict_bundle);
                    if !conflicts.iter().any(|b| *b == conflict_bundle) {
                        conflicts.push(conflict_bundle);
                        max_conflict_weight = std::cmp::max(
                            max_conflict_weight,
                            self.bundles[conflict_bundle.index()].cached_spill_weight(),
                        );
                        if max_allowable_cost.is_some()
                            && max_conflict_weight > max_allowable_cost.unwrap()
                        {
                            log::debug!("   -> reached high cost, retrying early");
                            return AllocRegResult::ConflictHighCost;
                        }
                    }
                } else {
                    log::debug!("   -> conflict with fixed reservation");
                    // range from a direct use of the PReg (due to clobber).
                    return AllocRegResult::ConflictWithFixed(
                        max_conflict_weight,
                        ProgPoint::from_index(preg_key.from),
                    );
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
        self.allocation_queue
            .insert(bundle, prio as usize, PReg::invalid());
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
        let mut stack = false;
        let bundledata = &self.bundles[bundle.index()];
        let first_range = bundledata.ranges[0].index;
        let first_range_data = &self.ranges[first_range.index()];

        if first_range_data.vreg.is_invalid() {
            log::debug!("  -> no vreg; minimal and fixed");
            minimal = true;
            fixed = true;
        } else {
            for u in &first_range_data.uses {
                log::debug!("  -> use: {:?}", u);
                if let OperandPolicy::FixedReg(_) = u.operand.policy() {
                    log::debug!("  -> fixed use at {:?}: {:?}", u.pos, u.operand);
                    fixed = true;
                }
                if let OperandPolicy::Stack = u.operand.policy() {
                    log::debug!("  -> stack use at {:?}: {:?}", u.pos, u.operand);
                    stack = true;
                }
                if stack && fixed {
                    break;
                }
            }
            // Minimal if the range covers only one instruction. Note
            // that it could cover just one ProgPoint,
            // i.e. X.Before..X.After, or two ProgPoints,
            // i.e. X.Before..X+1.Before.
            log::debug!("  -> first range has range {:?}", first_range_data.range);
            let bundle_start = self.bundles[bundle.index()]
                .ranges
                .first()
                .unwrap()
                .range
                .from;
            let bundle_end = self.bundles[bundle.index()].ranges.last().unwrap().range.to;
            minimal = bundle_start.inst() == bundle_end.prev().inst();
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
            stack,
        );
    }

    fn minimal_bundle(&mut self, bundle: LiveBundleIndex) -> bool {
        self.bundles[bundle.index()].cached_minimal()
    }

    fn recompute_range_properties(&mut self, range: LiveRangeIndex) {
        let rangedata = &mut self.ranges[range.index()];
        let mut w = 0;
        for u in &rangedata.uses {
            w += u.weight as u32;
            log::debug!("range{}: use {:?}", range.index(), u);
        }
        rangedata.set_uses_spill_weight(w);
        if rangedata.uses.len() > 0 && rangedata.uses[0].operand.kind() == OperandKind::Def {
            // Note that we *set* the flag here, but we never *clear*
            // it: it may be set by a progmove as well (which does not
            // create an explicit use or def), and we want to preserve
            // that. We will never split or trim ranges in a way that
            // removes a def at the front and requires the flag to be
            // cleared.
            rangedata.set_flag(LiveRangeFlag::StartsAtDef);
        }
    }

    fn split_and_requeue_bundle(
        &mut self,
        bundle: LiveBundleIndex,
        mut split_at: ProgPoint,
        reg_hint: PReg,
    ) {
        self.stats.splits += 1;
        log::debug!(
            "split bundle {:?} at {:?} and requeue with reg hint (for first part) {:?}",
            bundle,
            split_at,
            reg_hint,
        );

        // Split `bundle` at `split_at`, creating new LiveRanges and
        // bundles (and updating vregs' linked lists appropriately),
        // and enqueue the new bundles.

        let spillset = self.bundles[bundle.index()].spillset;

        assert!(!self.bundles[bundle.index()].ranges.is_empty());
        // Split point *at* start is OK; this means we peel off
        // exactly one use to create a minimal bundle.
        let bundle_start = self.bundles[bundle.index()]
            .ranges
            .first()
            .unwrap()
            .range
            .from;
        assert!(split_at >= bundle_start);
        let bundle_end = self.bundles[bundle.index()].ranges.last().unwrap().range.to;
        assert!(split_at < bundle_end);

        // Is the split point *at* the start? If so, peel off the
        // first use: set the split point just after it, or just
        // before it if it comes after the start of the bundle.
        if split_at == bundle_start {
            // Find any uses; if none, just chop off one instruction.
            let mut first_use = None;
            'outer: for entry in &self.bundles[bundle.index()].ranges {
                for u in &self.ranges[entry.index.index()].uses {
                    first_use = Some(u.pos);
                    break 'outer;
                }
            }
            log::debug!(" -> first use loc is {:?}", first_use);
            split_at = match first_use {
                Some(pos) => {
                    if pos.inst() == bundle_start.inst() {
                        ProgPoint::before(pos.inst().next())
                    } else {
                        ProgPoint::before(pos.inst())
                    }
                }
                None => ProgPoint::before(
                    self.bundles[bundle.index()]
                        .ranges
                        .first()
                        .unwrap()
                        .range
                        .from
                        .inst()
                        .next(),
                ),
            };
            log::debug!(
                "split point is at bundle start; advancing to {:?}",
                split_at
            );
        } else {
            // Don't split in the middle of an instruction -- this could
            // create impossible moves (we cannot insert a move between an
            // instruction's uses and defs).
            if split_at.pos() == InstPosition::After {
                split_at = split_at.next();
            }
            if split_at >= bundle_end {
                split_at = split_at.prev().prev();
            }
        }

        assert!(split_at > bundle_start && split_at < bundle_end);

        // We need to find which LRs fall on each side of the split,
        // which LR we need to split down the middle, then update the
        // current bundle, create a new one, and (re)-queue both.

        log::debug!(" -> LRs: {:?}", self.bundles[bundle.index()].ranges);

        let mut last_lr_in_old_bundle_idx = 0; // last LR-list index in old bundle
        let mut first_lr_in_new_bundle_idx = 0; // first LR-list index in new bundle
        for (i, entry) in self.bundles[bundle.index()].ranges.iter().enumerate() {
            if split_at > entry.range.from {
                last_lr_in_old_bundle_idx = i;
                first_lr_in_new_bundle_idx = i;
            }
            if split_at < entry.range.to {
                first_lr_in_new_bundle_idx = i;
                break;
            }
        }

        log::debug!(
            " -> last LR in old bundle: LR {:?}",
            self.bundles[bundle.index()].ranges[last_lr_in_old_bundle_idx]
        );
        log::debug!(
            " -> first LR in new bundle: LR {:?}",
            self.bundles[bundle.index()].ranges[first_lr_in_new_bundle_idx]
        );

        // Take the sublist of LRs that will go in the new bundle.
        let mut new_lr_list: LiveRangeList = self.bundles[bundle.index()]
            .ranges
            .iter()
            .cloned()
            .skip(first_lr_in_new_bundle_idx)
            .collect();
        self.bundles[bundle.index()]
            .ranges
            .truncate(last_lr_in_old_bundle_idx + 1);

        // If the first entry in `new_lr_list` is a LR that is split
        // down the middle, replace it with a new LR and chop off the
        // end of the same LR in the original list.
        if split_at > new_lr_list[0].range.from {
            assert_eq!(last_lr_in_old_bundle_idx, first_lr_in_new_bundle_idx);
            let orig_lr = new_lr_list[0].index;
            let new_lr = self.create_liverange(CodeRange {
                from: split_at,
                to: new_lr_list[0].range.to,
            });
            self.ranges[new_lr.index()].vreg = self.ranges[orig_lr.index()].vreg;
            log::debug!(" -> splitting LR {:?} into {:?}", orig_lr, new_lr);
            let first_use = self.ranges[orig_lr.index()]
                .uses
                .iter()
                .position(|u| u.pos >= split_at)
                .unwrap_or(self.ranges[orig_lr.index()].uses.len());
            let rest_uses: UseList = self.ranges[orig_lr.index()]
                .uses
                .iter()
                .cloned()
                .skip(first_use)
                .collect();
            self.ranges[new_lr.index()].uses = rest_uses;
            self.ranges[orig_lr.index()].uses.truncate(first_use);
            self.recompute_range_properties(orig_lr);
            self.recompute_range_properties(new_lr);
            new_lr_list[0].index = new_lr;
            new_lr_list[0].range = self.ranges[new_lr.index()].range;
            self.ranges[orig_lr.index()].range.to = split_at;
            self.bundles[bundle.index()].ranges[last_lr_in_old_bundle_idx].range =
                self.ranges[orig_lr.index()].range;

            // Perform a lazy split in the VReg data. We just
            // append the new LR and its range; we will sort by
            // start of range, and fix up range ends, once when we
            // iterate over the VReg's ranges after allocation
            // completes (this is the only time when order
            // matters).
            self.vregs[self.ranges[new_lr.index()].vreg.index()]
                .ranges
                .push(LiveRangeListEntry {
                    range: self.ranges[new_lr.index()].range,
                    index: new_lr,
                });
        }

        let new_bundle = self.create_bundle();
        log::debug!(" -> creating new bundle {:?}", new_bundle);
        self.bundles[new_bundle.index()].spillset = spillset;
        for entry in &new_lr_list {
            self.ranges[entry.index.index()].bundle = new_bundle;
        }
        self.bundles[new_bundle.index()].ranges = new_lr_list;

        self.recompute_bundle_properties(bundle);
        self.recompute_bundle_properties(new_bundle);
        let prio = self.compute_bundle_prio(bundle);
        let new_prio = self.compute_bundle_prio(new_bundle);
        self.bundles[bundle.index()].prio = prio;
        self.bundles[new_bundle.index()].prio = new_prio;
        self.allocation_queue
            .insert(bundle, prio as usize, reg_hint);
        self.allocation_queue
            .insert(new_bundle, new_prio as usize, reg_hint);
    }

    fn compute_requirement(&self, bundle: LiveBundleIndex) -> Requirement {
        let mut req = Requirement::Unknown;
        log::debug!("compute_requirement: {:?}", bundle);
        for entry in &self.bundles[bundle.index()].ranges {
            log::debug!(" -> LR {:?}", entry.index);
            for u in &self.ranges[entry.index.index()].uses {
                log::debug!("  -> use {:?}", u);
                let r = Requirement::from_operand(u.operand);
                req = req.merge(r);
                log::debug!("     -> req {:?}", req);
            }
        }
        log::debug!(" -> final: {:?}", req);
        req
    }

    fn process_bundle(
        &mut self,
        bundle: LiveBundleIndex,
        reg_hint: PReg,
    ) -> Result<(), RegAllocError> {
        let req = self.compute_requirement(bundle);
        // Grab a hint from either the queue or our spillset, if any.
        let hint_reg = if reg_hint != PReg::invalid() {
            reg_hint
        } else {
            self.spillsets[self.bundles[bundle.index()].spillset.index()].reg_hint
        };
        log::debug!("process_bundle: bundle {:?} hint {:?}", bundle, hint_reg,);

        if let Requirement::Conflict = req {
            // We have to split right away.
            assert!(
                !self.minimal_bundle(bundle),
                "Minimal bundle with conflict!"
            );
            let bundle_start = self.bundles[bundle.index()].ranges[0].range.from;
            self.split_and_requeue_bundle(
                bundle,
                /* split_at_point = */ bundle_start,
                reg_hint,
            );
            return Ok(());
        }

        // Try to allocate!
        let mut attempts = 0;
        loop {
            attempts += 1;
            log::debug!("attempt {}, req {:?}", attempts, req);
            debug_assert!(attempts < 100 * self.func.insts());

            let (class, fixed_preg) = match req {
                Requirement::Fixed(preg) => (preg.class(), Some(preg)),
                Requirement::Register(class) => (class, None),
                Requirement::Stack(_) => {
                    // If we must be on the stack, mark our spillset
                    // as required immediately.
                    self.spillsets[self.bundles[bundle.index()].spillset.index()].required = true;
                    return Ok(());
                }

                Requirement::Any(_) | Requirement::Unknown => {
                    // If a register is not *required*, spill now (we'll retry
                    // allocation on spilled bundles later).
                    log::debug!("spilling bundle {:?} to spilled_bundles list", bundle);
                    self.spilled_bundles.push(bundle);
                    return Ok(());
                }

                Requirement::Conflict => unreachable!(),
            };
            // Scan all pregs, or the one fixed preg, and attempt to allocate.

            let mut lowest_cost_evict_conflict_set: Option<LiveBundleVec> = None;
            let mut lowest_cost_evict_conflict_cost: Option<u32> = None;

            let mut lowest_cost_split_conflict_cost: Option<u32> = None;
            let mut lowest_cost_split_conflict_point = ProgPoint::before(Inst::new(0));
            let mut lowest_cost_split_conflict_reg = PReg::invalid();

            // Heuristic: start the scan for an available
            // register at an offset influenced both by our
            // location in the code and by the bundle we're
            // considering. This has the effect of spreading
            // demand more evenly across registers.
            let scan_offset = self.ranges[self.bundles[bundle.index()].ranges[0].index.index()]
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
                fixed_preg,
            ) {
                self.stats.process_bundle_reg_probes_any += 1;
                let preg_idx = PRegIndex::new(preg.index());
                log::debug!("trying preg {:?}", preg_idx);

                let scan_limit_cost = match (
                    lowest_cost_evict_conflict_cost,
                    lowest_cost_split_conflict_cost,
                ) {
                    (Some(a), Some(b)) => Some(std::cmp::max(a, b)),
                    _ => None,
                };
                match self.try_to_allocate_bundle_to_reg(bundle, preg_idx, scan_limit_cost) {
                    AllocRegResult::Allocated(alloc) => {
                        self.stats.process_bundle_reg_success_any += 1;
                        log::debug!(" -> allocated to any {:?}", preg_idx);
                        self.spillsets[self.bundles[bundle.index()].spillset.index()].reg_hint =
                            alloc.as_reg().unwrap();
                        return Ok(());
                    }
                    AllocRegResult::Conflict(bundles) => {
                        log::debug!(" -> conflict with bundles {:?}", bundles);

                        let first_conflict_point =
                            self.bundles[bundles[0].index()].ranges[0].range.from;

                        let conflict_cost = self.maximum_spill_weight_in_bundle_set(&bundles);

                        if lowest_cost_evict_conflict_cost.is_none()
                            || conflict_cost < lowest_cost_evict_conflict_cost.unwrap()
                        {
                            lowest_cost_evict_conflict_cost = Some(conflict_cost);
                            lowest_cost_evict_conflict_set = Some(bundles);
                        }

                        let loop_depth = self.cfginfo.approx_loop_depth
                            [self.cfginfo.insn_block[first_conflict_point.inst().index()].index()];
                        let move_cost = spill_weight_from_policy(
                            OperandPolicy::Reg,
                            loop_depth as usize,
                            /* is_def = */ true,
                        );
                        if lowest_cost_split_conflict_cost.is_none()
                            || (conflict_cost + move_cost)
                                < lowest_cost_split_conflict_cost.unwrap()
                        {
                            lowest_cost_split_conflict_cost = Some(conflict_cost + move_cost);
                            lowest_cost_split_conflict_point = first_conflict_point;
                            lowest_cost_split_conflict_reg = preg;
                        }
                    }
                    AllocRegResult::ConflictWithFixed(max_cost, point) => {
                        log::debug!(" -> conflict with fixed alloc; cost of other bundles up to point is {}, conflict at {:?}", max_cost, point);

                        let loop_depth = self.cfginfo.approx_loop_depth
                            [self.cfginfo.insn_block[point.inst().index()].index()];
                        let move_cost = spill_weight_from_policy(
                            OperandPolicy::Reg,
                            loop_depth as usize,
                            /* is_def = */ true,
                        );

                        if lowest_cost_split_conflict_cost.is_none()
                            || (max_cost + move_cost) < lowest_cost_split_conflict_cost.unwrap()
                        {
                            lowest_cost_split_conflict_cost = Some(max_cost + move_cost);
                            lowest_cost_split_conflict_point = point;
                            lowest_cost_split_conflict_reg = preg;
                        }
                    }
                    AllocRegResult::ConflictHighCost => {
                        // Simply don't consider -- we already have
                        // a lower-cost conflict bundle option
                        // to evict.
                        continue;
                    }
                }
            }

            // Otherwise, we *require* a register, but didn't fit into
            // any with current bundle assignments. Hence, we will need
            // to either split or attempt to evict some bundles.

            log::debug!(
                " -> lowest cost evict: set {:?}, cost {:?}",
                lowest_cost_evict_conflict_set,
                lowest_cost_evict_conflict_cost,
            );
            log::debug!(
                " -> lowest cost split: cost {:?}, point {:?}, reg {:?}",
                lowest_cost_split_conflict_cost,
                lowest_cost_split_conflict_point,
                lowest_cost_split_conflict_reg
            );

            // If we reach here, we *must* have an option either to split or evict.
            assert!(
                lowest_cost_split_conflict_cost.is_some()
                    || lowest_cost_evict_conflict_cost.is_some()
            );

            let our_spill_weight = self.bundle_spill_weight(bundle);
            log::debug!(" -> our spill weight: {}", our_spill_weight);

            // We detect the "too-many-live-registers" case here and
            // return an error cleanly, rather than panicking, because
            // the regalloc.rs fuzzer depends on the register
            // allocator to correctly reject impossible-to-allocate
            // programs in order to discard invalid test cases.
            if self.minimal_bundle(bundle)
                && (attempts >= 2
                    || lowest_cost_evict_conflict_cost.is_none()
                    || lowest_cost_evict_conflict_cost.unwrap() >= our_spill_weight)
            {
                if let Requirement::Register(class) = req {
                    // Check if this is a too-many-live-registers situation.
                    let range = self.bundles[bundle.index()].ranges[0].range;
                    let mut min_bundles_assigned = 0;
                    let mut fixed_assigned = 0;
                    let mut total_regs = 0;
                    for preg in self.env.preferred_regs_by_class[class as u8 as usize]
                        .iter()
                        .chain(self.env.non_preferred_regs_by_class[class as u8 as usize].iter())
                    {
                        if let Some(&lr) = self.pregs[preg.index()]
                            .allocations
                            .btree
                            .get(&LiveRangeKey::from_range(&range))
                        {
                            if lr.is_valid() {
                                if self.minimal_bundle(self.ranges[lr.index()].bundle) {
                                    min_bundles_assigned += 1;
                                }
                            } else {
                                fixed_assigned += 1;
                            }
                        }
                        total_regs += 1;
                    }
                    if min_bundles_assigned + fixed_assigned == total_regs {
                        return Err(RegAllocError::TooManyLiveRegs);
                    }
                }

                panic!("Could not allocate minimal bundle, but the allocation problem should be possible to solve");
            }

            // If our bundle's weight is less than or equal to(*) the
            // evict cost, choose to split.  Also pick splitting if
            // we're on our second or more attempt and we didn't
            // allocate.  Also pick splitting if the conflict set is
            // empty, meaning a fixed conflict that can't be evicted.
            //
            // (*) the "equal to" part is very important: it prevents
            // an infinite loop where two bundles with equal spill
            // cost continually evict each other in an infinite
            // allocation loop. In such a case, the first bundle in
            // wins, and the other splits.
            //
            // Note that we don't split if the bundle is minimal.
            if !self.minimal_bundle(bundle)
                && (attempts >= 2
                    || lowest_cost_evict_conflict_cost.is_none()
                    || our_spill_weight <= lowest_cost_evict_conflict_cost.unwrap())
            {
                log::debug!(
                    " -> deciding to split: our spill weight is {}",
                    self.bundle_spill_weight(bundle)
                );
                let bundle_start = self.bundles[bundle.index()].ranges[0].range.from;
                let mut split_at_point =
                    std::cmp::max(lowest_cost_split_conflict_point, bundle_start);
                let requeue_with_reg = lowest_cost_split_conflict_reg;

                // Adjust `split_at_point` if it is within a deeper loop
                // than the bundle start -- hoist it to just before the
                // first loop header it encounters.
                let bundle_start_depth = self.cfginfo.approx_loop_depth
                    [self.cfginfo.insn_block[bundle_start.inst().index()].index()];
                let split_at_depth = self.cfginfo.approx_loop_depth
                    [self.cfginfo.insn_block[split_at_point.inst().index()].index()];
                if split_at_depth > bundle_start_depth {
                    for block in (self.cfginfo.insn_block[bundle_start.inst().index()].index() + 1)
                        ..=self.cfginfo.insn_block[split_at_point.inst().index()].index()
                    {
                        if self.cfginfo.approx_loop_depth[block] > bundle_start_depth {
                            split_at_point = self.cfginfo.block_entry[block];
                            break;
                        }
                    }
                }

                self.split_and_requeue_bundle(bundle, split_at_point, requeue_with_reg);
                return Ok(());
            } else {
                // Evict all bundles in `conflicting bundles` and try again.
                self.stats.evict_bundle_event += 1;
                for &bundle in &lowest_cost_evict_conflict_set.unwrap() {
                    log::debug!(" -> evicting {:?}", bundle);
                    self.evict_bundle(bundle);
                    self.stats.evict_bundle_count += 1;
                }
            }
        }
    }

    fn try_allocating_regs_for_spilled_bundles(&mut self) {
        log::debug!("allocating regs for spilled bundles");
        for i in 0..self.spilled_bundles.len() {
            let bundle = self.spilled_bundles[i]; // don't borrow self

            let class = self.spillsets[self.bundles[bundle.index()].spillset.index()].class;

            // This may be an empty-range bundle whose ranges are not
            // sorted; sort all range-lists again here.
            self.bundles[bundle.index()]
                .ranges
                .sort_unstable_by_key(|entry| entry.range.from);

            let mut success = false;
            self.stats.spill_bundle_reg_probes += 1;
            for preg in RegTraversalIter::new(
                self.env,
                class,
                PReg::invalid(),
                PReg::invalid(),
                bundle.index(),
                None,
            ) {
                log::debug!("trying bundle {:?} to preg {:?}", bundle, preg);
                let preg_idx = PRegIndex::new(preg.index());
                if let AllocRegResult::Allocated(_) =
                    self.try_to_allocate_bundle_to_reg(bundle, preg_idx, None)
                {
                    self.stats.spill_bundle_reg_success += 1;
                    success = true;
                    break;
                }
            }
            if !success {
                log::debug!(
                    "spilling bundle {:?}: marking spillset {:?} as required",
                    bundle,
                    self.bundles[bundle.index()].spillset
                );
                self.spillsets[self.bundles[bundle.index()].spillset.index()].required = true;
            }
        }
    }

    fn spillslot_can_fit_spillset(
        &mut self,
        spillslot: SpillSlotIndex,
        spillset: SpillSetIndex,
    ) -> bool {
        for &vreg in &self.spillsets[spillset.index()].vregs {
            for entry in &self.vregs[vreg.index()].ranges {
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
        for i in 0..self.spillsets[spillset.index()].vregs.len() {
            // don't borrow self
            let vreg = self.spillsets[spillset.index()].vregs[i];
            log::debug!(
                "spillslot {:?} alloc'ed to spillset {:?}: vreg {:?}",
                spillslot,
                spillset,
                vreg,
            );
            for entry in &self.vregs[vreg.index()].ranges {
                log::debug!(
                    "spillslot {:?} getting range {:?} from LR {:?} from vreg {:?}",
                    spillslot,
                    entry.range,
                    entry.index,
                    vreg,
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
            if !self.spillsets[spillset.index()].required {
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

            let pinned_alloc = if self.vregs[vreg.index()].is_pinned {
                self.func.is_pinned_vreg(self.vreg_regs[vreg.index()])
            } else {
                None
            };

            let mut clean_spillslot: Option<SpillSlot> = None;

            // For each range in each vreg, insert moves or
            // half-moves.  We also scan over `blockparam_ins` and
            // `blockparam_outs`, which are sorted by (block, vreg),
            // and over program-move srcs/dsts to fill in allocations.
            let mut prev = LiveRangeIndex::invalid();
            for range_idx in 0..self.vregs[vreg.index()].ranges.len() {
                let entry = self.vregs[vreg.index()].ranges[range_idx];
                let alloc = pinned_alloc
                    .map(|preg| Allocation::reg(preg))
                    .unwrap_or_else(|| self.get_alloc_for_range(entry.index));
                let range = entry.range;
                log::debug!(
                    "apply_allocations: vreg {:?} LR {:?} with range {:?} has alloc {:?} (pinned {:?})",
                    vreg,
                    entry.index,
                    range,
                    alloc,
                    pinned_alloc,
                );
                debug_assert!(alloc != Allocation::none());

                if self.annotations_enabled {
                    self.annotate(
                        range.from,
                        format!(
                            " <<< start v{} in {} (range{}) (bundle{})",
                            vreg.index(),
                            alloc,
                            entry.index.index(),
                            self.ranges[entry.index.index()].bundle.raw_u32(),
                        ),
                    );
                    self.annotate(
                        range.to,
                        format!(
                            "     end   v{} in {} (range{}) (bundle{}) >>>",
                            vreg.index(),
                            alloc,
                            entry.index.index(),
                            self.ranges[entry.index.index()].bundle.raw_u32(),
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
                //
                // Also note that this case is not applicable to
                // pinned vregs (because they are always in one PReg).
                if pinned_alloc.is_none() && prev.is_valid() {
                    let prev_alloc = self.get_alloc_for_range(prev);
                    let prev_range = self.ranges[prev.index()].range;
                    let first_is_def =
                        self.ranges[entry.index.index()].has_flag(LiveRangeFlag::StartsAtDef);
                    debug_assert!(prev_alloc != Allocation::none());

                    // If this is a stack-to-reg move, track that the reg is a clean copy of a spillslot.
                    if prev_alloc.is_stack() && alloc.is_reg() {
                        clean_spillslot = Some(prev_alloc.as_stack().unwrap());
                    }
                    // If this is a reg-to-stack move, elide it if the spillslot is still clean.
                    let skip_spill = prev_alloc.is_reg()
                        && alloc.is_stack()
                        && clean_spillslot == alloc.as_stack();

                    if prev_range.to == range.from
                        && !self.is_start_of_block(range.from)
                        && !first_is_def
                        && !skip_spill
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
                            Some(self.vreg_regs[vreg.index()]),
                        );
                    }
                }

                // If this range either spans any block boundary, or
                // has any mods/defs, then the spillslot (if any) that
                // its value came from is no longer 'clean'.
                if clean_spillslot.is_some() {
                    if self.cfginfo.insn_block[range.from.inst().index()]
                        != self.cfginfo.insn_block[range.to.prev().inst().index()]
                        || range.from
                            == self.cfginfo.block_entry
                                [self.cfginfo.insn_block[range.from.inst().index()].index()]
                    {
                        clean_spillslot = None;
                    } else if self.ranges[entry.index.index()].has_flag(LiveRangeFlag::StartsAtDef)
                    {
                        clean_spillslot = None;
                    } else {
                        for u in &self.ranges[entry.index.index()].uses {
                            match u.operand.kind() {
                                OperandKind::Def | OperandKind::Mod => {
                                    clean_spillslot = None;
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                }

                // The block-to-block edge-move logic is not
                // applicable to pinned vregs, which are always in one
                // PReg (so never need moves within their own vreg
                // ranges).
                if pinned_alloc.is_none() {
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

                                if self.annotations_enabled {
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
                                #[cfg(debug)]
                                {
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
                        self.blockparam_allocs.push((
                            blockparam_block,
                            blockparam_idx,
                            vreg,
                            alloc,
                        ));
                    }
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
                        #[cfg(debug)]
                        {
                            if log::log_enabled!(log::Level::Debug) {
                                self.annotate(
                                    ProgPoint::before(inst),
                                    format!(
                                        " reuse-input-copy: {} -> {}",
                                        input_alloc, output_alloc
                                    ),
                                );
                            }
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
            let mut self_moves: SmallVec<[InsertedMove; 8]> = smallvec![];

            for m in moves {
                if m.from_alloc.is_reg() && m.to_alloc.is_reg() {
                    assert_eq!(m.from_alloc.class(), m.to_alloc.class());
                }
                if m.from_alloc == m.to_alloc {
                    if m.to_vreg.is_some() {
                        self_moves.push(m.clone());
                    }
                    continue;
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

            for m in &self_moves {
                self.add_edit(
                    pos,
                    prio,
                    Edit::Move {
                        from: m.from_alloc,
                        to: m.to_alloc,
                        to_vreg: m.to_vreg,
                    },
                );
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
        if self.annotations_enabled {
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
                    &Edit::DefAlloc { alloc, vreg } => {
                        let s = format!("defalloc {:?} := {:?}", alloc, vreg);
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
        self.compute_liveness()?;
        self.merge_vreg_bundles();
        self.queue_bundles();
        if log::log_enabled!(log::Level::Debug) {
            self.dump_state();
        }
        Ok(())
    }

    pub(crate) fn run(&mut self) -> Result<(), RegAllocError> {
        self.process_bundles()?;
        self.try_allocating_regs_for_spilled_bundles();
        self.allocate_spillslots();
        self.apply_allocations_and_insert_moves();
        self.resolve_inserted_moves();
        self.compute_stackmaps();
        Ok(())
    }

    fn annotate(&mut self, progpoint: ProgPoint, s: String) {
        if self.annotations_enabled {
            self.debug_annotations
                .entry(progpoint)
                .or_insert_with(|| vec![])
                .push(s);
        }
    }

    fn dump_results(&self) {
        log::info!("=== REGALLOC RESULTS ===");
        for block in 0..self.func.blocks() {
            let block = Block::new(block);
            log::info!(
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
                    log::info!("  inst{}-pre: {}", inst.index(), annotation);
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
                log::info!(
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
                    log::info!("  inst{}-post: {}", inst.index(), annotation);
                }
            }
        }
    }
}

pub fn run<F: Function>(
    func: &F,
    mach_env: &MachineEnv,
    enable_annotations: bool,
) -> Result<Output, RegAllocError> {
    let cfginfo = CFGInfo::new(func)?;

    let mut env = Env::new(func, mach_env, cfginfo, enable_annotations);
    env.init()?;

    env.run()?;

    if enable_annotations {
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
