/*
 * The following code is derived from `lib/src/checker.rs` in the
 * regalloc.rs project
 * (https://github.com/bytecodealliance/regalloc.rs). regalloc.rs is
 * also licensed under Apache-2.0 with the LLVM exception, as the rest
 * of regalloc2 is.
 */

//! Checker: verifies that spills/reloads/moves retain equivalent
//! dataflow to original, VReg-based code.
//!
//! The basic idea is that we track symbolic values as they flow
//! through spills and reloads.  The symbolic values represent
//! particular virtual registers in the original function body
//! presented to the register allocator. Any instruction in the
//! original function body (i.e., not added by the allocator)
//! conceptually generates a symbolic value "Vn" when storing to (or
//! modifying) a virtual register.
//!
//! A symbolic value is logically a *set of virtual registers*,
//! representing all virtual registers equal to the value in the given
//! storage slot at a given program point. This representation (as
//! opposed to tracking just one virtual register) is necessary
//! because the regalloc may implement moves in the source program
//! (via move instructions or blockparam assignments on edges) in
//! "intelligent" ways, taking advantage of values that are already in
//! the right place, so we need to know *all* names for a value.
//!
//! These symbolic values are precise but partial: in other words, if
//! a physical register is described as containing a virtual register
//! at a program point, it must actually contain the value of this
//! register (modulo any analysis bugs); but it may describe fewer
//! virtual registers even in cases where one *could* statically prove
//! that it contains a certain register, because the analysis is not
//! perfectly path-sensitive or value-sensitive. However, all
//! assignments *produced by our register allocator* should be
//! analyzed fully precisely. (This last point is important and bears
//! repeating: we only need to verify the programs that we produce,
//! not arbitrary programs.)
//!
//! Operand constraints (fixed register, register, any) are also checked
//! at each operand.
//!
//! ## Formal Definition
//!
//! The analysis lattice is:
//!
//!                      Top (V)
//!                         |
//!                        𝒫(V)   // the Powerset of the set of virtual regs
//!                         |
//!                 Bottom ( ∅ )     // the empty set
//!
//! and the lattice ordering relation is the subset relation: S ≤ U
//! iff S ⊆ U. The lattice meet-function is intersection.
//!
//! The dataflow analysis state at each program point (each point
//! before or after an instruction) is:
//!
//!   - map of: Allocation -> lattice value
//!
//! And the transfer functions for instructions are (where `A` is the
//! above map from allocated physical registers to lattice values):
//!
//!   - `Edit::Move` inserted by RA:       [ alloc_d := alloc_s ]
//!
//!       A' = A[alloc_d → A[alloc_s]]
//!
//!   - statement in pre-regalloc function [ V_i := op V_j, V_k, ... ]
//!     with allocated form                [ A_i := op A_j, A_k, ... ]
//!
//!       A' = { A_k → A[A_k] \ { V_i } for k ≠ i } ∪
//!            { A_i -> { V_i } }
//!
//!     In other words, a statement, even after allocation, generates
//!     a symbol that corresponds to its original virtual-register
//!     def. Simultaneously, that same virtual register symbol is removed
//!     from all other allocs: they no longer carry the current value.
//!
//!   - Parallel moves or blockparam-assignments in original program
//!                                       [ V_d1 := V_s1, V_d2 := V_s2, ... ]
//!
//!       A' = { A_k → subst(A[A_k]) for all k }
//!            where subst(S) removes symbols for overwritten virtual
//!            registers (V_d1 .. V_dn) and then adds V_di whenever
//!            V_si appeared prior to the removals.
//!
//! To check correctness, we first find the dataflow fixpoint with the
//! above lattice and transfer/meet functions. Then, at each op, we
//! examine the dataflow solution at the preceding program point, and
//! check that the allocation for each op arg (input/use) contains the
//! symbol corresponding to the original virtual register specified
//! for this arg.

#![allow(dead_code)]

use crate::{
    Allocation, AllocationKind, Block, Edit, Function, Inst, InstOrEdit, InstPosition, Operand,
    OperandConstraint, OperandKind, OperandPos, Output, PReg, RegClass, VReg,
};
use fxhash::{FxHashMap, FxHashSet};
use smallvec::{smallvec, SmallVec};
use std::collections::VecDeque;
use std::default::Default;
use std::hash::Hash;
use std::result::Result;

/// A set of errors detected by the regalloc checker.
#[derive(Clone, Debug)]
pub struct CheckerErrors {
    errors: Vec<CheckerError>,
}

/// A single error detected by the regalloc checker.
#[derive(Clone, Debug)]
pub enum CheckerError {
    MissingAllocation {
        inst: Inst,
        op: Operand,
    },
    UnknownValueInAllocation {
        inst: Inst,
        op: Operand,
        alloc: Allocation,
    },
    ConflictedValueInAllocation {
        inst: Inst,
        op: Operand,
        alloc: Allocation,
    },
    IncorrectValuesInAllocation {
        inst: Inst,
        op: Operand,
        alloc: Allocation,
        actual: FxHashSet<VReg>,
    },
    ConstraintViolated {
        inst: Inst,
        op: Operand,
        alloc: Allocation,
    },
    AllocationIsNotReg {
        inst: Inst,
        op: Operand,
        alloc: Allocation,
    },
    AllocationIsNotFixedReg {
        inst: Inst,
        op: Operand,
        alloc: Allocation,
    },
    AllocationIsNotReuse {
        inst: Inst,
        op: Operand,
        alloc: Allocation,
        expected_alloc: Allocation,
    },
    AllocationIsNotStack {
        inst: Inst,
        op: Operand,
        alloc: Allocation,
    },
    ConflictedValueInStackmap {
        inst: Inst,
        alloc: Allocation,
    },
    NonRefValuesInStackmap {
        inst: Inst,
        alloc: Allocation,
        vregs: FxHashSet<VReg>,
    },
}

/// Abstract state for an allocation.
///
/// Equivalent to a set of virtual register names, with the
/// universe-set as top and empty set as bottom lattice element. The
/// meet-function is thus set intersection.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CheckerValue {
    /// This value is the "universe set".
    universe: bool,
    /// The VRegs that this value is equal to.
    vregs: FxHashSet<VReg>,
}

impl Default for CheckerValue {
    fn default() -> CheckerValue {
        CheckerValue {
            universe: true,
            vregs: FxHashSet::default(),
        }
    }
}

impl CheckerValue {
    /// Meet function of the abstract-interpretation value
    /// lattice. Returns a boolean value indicating whether `self` was
    /// changed.
    fn meet_with(&mut self, other: &CheckerValue) -> bool {
        if self.universe {
            *self = other.clone();
            true
        } else if other.universe {
            false
        } else {
            let mut remove_keys: SmallVec<[VReg; 4]> = smallvec![];
            for vreg in &self.vregs {
                if !other.vregs.contains(vreg) {
                    // Not present in other; this is intersection, so
                    // remove.
                    remove_keys.push(vreg.clone());
                }
            }

            for key in &remove_keys {
                self.vregs.remove(key);
            }

            !remove_keys.is_empty()
        }
    }

    fn from_reg(reg: VReg) -> CheckerValue {
        CheckerValue {
            universe: false,
            vregs: std::iter::once(reg).collect(),
        }
    }

    fn remove_vreg(&mut self, reg: VReg) {
        self.vregs.remove(&reg);
    }

    fn empty() -> CheckerValue {
        CheckerValue {
            universe: false,
            vregs: FxHashSet::default(),
        }
    }
}

/// State that steps through program points as we scan over the instruction stream.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CheckerState {
    top: bool,
    allocations: FxHashMap<Allocation, CheckerValue>,
}

impl Default for CheckerState {
    fn default() -> CheckerState {
        CheckerState {
            top: true,
            allocations: FxHashMap::default(),
        }
    }
}

impl std::fmt::Display for CheckerValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.universe {
            write!(f, "top")
        } else {
            write!(f, "{{ ")?;
            for vreg in &self.vregs {
                write!(f, "{} ", vreg)?;
            }
            write!(f, "}}")?;
            Ok(())
        }
    }
}

/// Meet function for analysis value: meet individual values at
/// matching allocations, and intersect keys (remove key-value pairs
/// only on one side). Returns boolean flag indicating whether `into`
/// changed.
fn merge_map<K: Copy + Clone + PartialEq + Eq + Hash>(
    into: &mut FxHashMap<K, CheckerValue>,
    from: &FxHashMap<K, CheckerValue>,
) -> bool {
    let mut changed = false;
    let mut remove_keys: SmallVec<[K; 4]> = smallvec![];
    for (k, into_v) in into.iter_mut() {
        if let Some(from_v) = from.get(k) {
            changed |= into_v.meet_with(from_v);
        } else {
            remove_keys.push(k.clone());
        }
    }

    for remove_key in &remove_keys {
        into.remove(remove_key);
    }
    changed |= !remove_keys.is_empty();

    changed
}

impl CheckerState {
    /// Create a new checker state.
    fn new() -> CheckerState {
        Default::default()
    }

    /// Merge this checker state with another at a CFG join-point.
    fn meet_with(&mut self, other: &CheckerState) -> bool {
        if self.top {
            *self = other.clone();
            !self.top
        } else if other.top {
            false
        } else {
            self.top = false;
            merge_map(&mut self.allocations, &other.allocations)
        }
    }

    fn check_val(
        &self,
        inst: Inst,
        op: Operand,
        alloc: Allocation,
        val: &CheckerValue,
        allocs: &[Allocation],
    ) -> Result<(), CheckerError> {
        if alloc == Allocation::none() {
            return Err(CheckerError::MissingAllocation { inst, op });
        }

        if val.universe {
            return Err(CheckerError::UnknownValueInAllocation { inst, op, alloc });
        }

        if !val.vregs.contains(&op.vreg()) {
            return Err(CheckerError::IncorrectValuesInAllocation {
                inst,
                op,
                alloc,
                actual: val.vregs.clone(),
            });
        }

        self.check_constraint(inst, op, alloc, allocs)?;

        Ok(())
    }

    /// Check an instruction against this state. This must be called
    /// twice: once with `InstPosition::Before`, and once with
    /// `InstPosition::After` (after updating state with defs).
    fn check<'a, F: Function>(
        &self,
        pos: InstPosition,
        checkinst: &CheckerInst,
        checker: &Checker<'a, F>,
    ) -> Result<(), CheckerError> {
        let default_val = Default::default();
        match checkinst {
            &CheckerInst::Op {
                inst,
                ref operands,
                ref allocs,
                ..
            } => {
                // Skip Use-checks at the After point if there are any
                // reused inputs: the Def which reuses the input
                // happens early.
                let has_reused_input = operands
                    .iter()
                    .any(|op| matches!(op.constraint(), OperandConstraint::Reuse(_)));
                if has_reused_input && pos == InstPosition::After {
                    return Ok(());
                }

                // For each operand, check (i) that the allocation
                // contains the expected vreg, and (ii) that it meets
                // the requirements of the OperandConstraint.
                for (op, alloc) in operands.iter().zip(allocs.iter()) {
                    let is_here = match (op.pos(), pos) {
                        (OperandPos::Early, InstPosition::Before) => true,
                        (OperandPos::Late, InstPosition::After) => true,
                        _ => false,
                    };
                    if !is_here {
                        continue;
                    }
                    if op.kind() == OperandKind::Def {
                        continue;
                    }

                    let val = self.allocations.get(alloc).unwrap_or(&default_val);
                    trace!(
                        "checker: checkinst {:?}: op {:?}, alloc {:?}, checker value {:?}",
                        checkinst,
                        op,
                        alloc,
                        val
                    );
                    self.check_val(inst, *op, *alloc, val, allocs)?;
                }
            }
            &CheckerInst::Safepoint { inst, ref allocs } => {
                for &alloc in allocs {
                    let val = self.allocations.get(&alloc).unwrap_or(&default_val);
                    trace!(
                        "checker: checkinst {:?}: safepoint slot {}, checker value {:?}",
                        checkinst,
                        alloc,
                        val
                    );

                    let reffy = val
                        .vregs
                        .iter()
                        .any(|vreg| checker.reftyped_vregs.contains(vreg));
                    if !reffy {
                        return Err(CheckerError::NonRefValuesInStackmap {
                            inst,
                            alloc,
                            vregs: val.vregs.clone(),
                        });
                    }
                }
            }
            &CheckerInst::ParallelMove { .. } | &CheckerInst::Move { .. } => {
                // This doesn't need verification; we just update
                // according to the move semantics in the step
                // function below.
            }
        }
        Ok(())
    }

    /// Update according to instruction.
    fn update<'a, F: Function>(&mut self, checkinst: &CheckerInst, checker: &Checker<'a, F>) {
        self.top = false;
        let default_val = Default::default();
        match checkinst {
            &CheckerInst::Move { into, from } => {
                let val = self.allocations.get(&from).unwrap_or(&default_val).clone();
                trace!(
                    "checker: checkinst {:?} updating: move {:?} -> {:?} val {:?}",
                    checkinst,
                    from,
                    into,
                    val
                );
                self.allocations.insert(into, val);
            }
            &CheckerInst::ParallelMove { ref into, ref from } => {
                // First, build map of actions for each vreg in an
                // alloc. If an alloc has a reg V_i before a parallel
                // move, then for each use of V_i as a source (V_i ->
                // V_j), we might add a new V_j wherever V_i appears;
                // and if V_i is used as a dest (at most once), then
                // it must be removed first from allocs' vreg sets.
                let mut additions: FxHashMap<VReg, SmallVec<[VReg; 2]>> = FxHashMap::default();
                let mut deletions: FxHashSet<VReg> = FxHashSet::default();

                for (&dest, &src) in into.iter().zip(from.iter()) {
                    deletions.insert(dest);
                    additions
                        .entry(src)
                        .or_insert_with(|| smallvec![])
                        .push(dest);
                }

                // Now process each allocation's set of vreg labels,
                // first deleting those labels that were updated by
                // this parallel move, then adding back labels
                // redefined by the move.
                for value in self.allocations.values_mut() {
                    if value.universe {
                        continue;
                    }
                    let mut insertions: SmallVec<[VReg; 2]> = smallvec![];
                    for &vreg in &value.vregs {
                        if let Some(additions) = additions.get(&vreg) {
                            insertions.extend(additions.iter().cloned());
                        }
                    }
                    for &d in &deletions {
                        value.vregs.remove(&d);
                    }
                    value.vregs.extend(insertions);
                }
            }
            &CheckerInst::Op {
                ref operands,
                ref allocs,
                ref clobbers,
                ..
            } => {
                // For each def, (i) update alloc to reflect defined
                // vreg (and only that vreg), and (ii) update all
                // other allocs in the checker state by removing this
                // vreg, if defined (other defs are now stale).
                for (op, alloc) in operands.iter().zip(allocs.iter()) {
                    if op.kind() != OperandKind::Def {
                        continue;
                    }
                    self.allocations
                        .insert(*alloc, CheckerValue::from_reg(op.vreg()));
                    for (other_alloc, other_value) in &mut self.allocations {
                        if *alloc != *other_alloc {
                            other_value.remove_vreg(op.vreg());
                        }
                    }
                }
                for clobber in clobbers {
                    self.allocations.remove(&Allocation::reg(*clobber));
                }
            }
            &CheckerInst::Safepoint { ref allocs, .. } => {
                for (alloc, value) in &mut self.allocations {
                    if alloc.is_reg() {
                        continue;
                    }
                    if !allocs.contains(&alloc) {
                        // Remove all reftyped vregs as labels.
                        let new_vregs = value
                            .vregs
                            .difference(&checker.reftyped_vregs)
                            .cloned()
                            .collect();
                        value.vregs = new_vregs;
                    }
                }
            }
        }
    }

    fn check_constraint(
        &self,
        inst: Inst,
        op: Operand,
        alloc: Allocation,
        allocs: &[Allocation],
    ) -> Result<(), CheckerError> {
        match op.constraint() {
            OperandConstraint::Any => {}
            OperandConstraint::Reg => {
                // Reject pregs that represent a fixed stack slot.
                if let Some(preg) = alloc.as_reg() {
                    if preg.class() == RegClass::Int && (0..32).contains(&preg.hw_enc()) {
                        return Ok(());
                    }
                }
                return Err(CheckerError::AllocationIsNotReg { inst, op, alloc });
            }
            OperandConstraint::Stack => {
                if alloc.kind() != AllocationKind::Stack {
                    // Accept pregs that represent a fixed stack slot.
                    if let Some(preg) = alloc.as_reg() {
                        if preg.class() == RegClass::Int && (32..63).contains(&preg.hw_enc()) {
                            return Ok(());
                        }
                    }
                    return Err(CheckerError::AllocationIsNotStack { inst, op, alloc });
                }
            }
            OperandConstraint::FixedReg(preg) => {
                if alloc != Allocation::reg(preg) {
                    return Err(CheckerError::AllocationIsNotFixedReg { inst, op, alloc });
                }
            }
            OperandConstraint::Reuse(idx) => {
                if alloc.kind() != AllocationKind::Reg {
                    return Err(CheckerError::AllocationIsNotReg { inst, op, alloc });
                }
                if alloc != allocs[idx] {
                    return Err(CheckerError::AllocationIsNotReuse {
                        inst,
                        op,
                        alloc,
                        expected_alloc: allocs[idx],
                    });
                }
            }
        }
        Ok(())
    }
}

/// An instruction representation in the checker's BB summary.
#[derive(Clone, Debug)]
pub(crate) enum CheckerInst {
    /// A move between allocations (these could be registers or
    /// spillslots).
    Move { into: Allocation, from: Allocation },

    /// A parallel move in the original program. Simultaneously moves
    /// from all source vregs to all corresponding dest vregs,
    /// permitting overlap in the src and dest sets and doing all
    /// reads before any writes.
    ParallelMove { into: Vec<VReg>, from: Vec<VReg> },

    /// A regular instruction with fixed use and def slots. Contains
    /// both the original operands (as given to the regalloc) and the
    /// allocation results.
    Op {
        inst: Inst,
        operands: Vec<Operand>,
        allocs: Vec<Allocation>,
        clobbers: Vec<PReg>,
    },

    /// A safepoint, with the given Allocations specified as containing
    /// reftyped values. All other reftyped values become invalid.
    Safepoint { inst: Inst, allocs: Vec<Allocation> },
}

#[derive(Debug)]
pub struct Checker<'a, F: Function> {
    f: &'a F,
    bb_in: FxHashMap<Block, CheckerState>,
    bb_insts: FxHashMap<Block, Vec<CheckerInst>>,
    edge_insts: FxHashMap<(Block, Block), Vec<CheckerInst>>,
    reftyped_vregs: FxHashSet<VReg>,
}

impl<'a, F: Function> Checker<'a, F> {
    /// Create a new checker for the given function, initializing CFG
    /// info immediately.  The client should call the `add_*()`
    /// methods to add abstract instructions to each BB before
    /// invoking `run()` to check for errors.
    pub fn new(f: &'a F) -> Checker<'a, F> {
        let mut bb_in = FxHashMap::default();
        let mut bb_insts = FxHashMap::default();
        let mut edge_insts = FxHashMap::default();
        let mut reftyped_vregs = FxHashSet::default();

        for block in 0..f.num_blocks() {
            let block = Block::new(block);
            bb_in.insert(block, Default::default());
            bb_insts.insert(block, vec![]);
            for &succ in f.block_succs(block) {
                edge_insts.insert((block, succ), vec![]);
            }
        }

        for &vreg in f.reftype_vregs() {
            reftyped_vregs.insert(vreg);
        }

        Checker {
            f,
            bb_in,
            bb_insts,
            edge_insts,
            reftyped_vregs,
        }
    }

    /// Build the list of checker instructions based on the given func
    /// and allocation results.
    pub fn prepare(&mut self, out: &Output) {
        trace!("checker: out = {:?}", out);
        // Preprocess safepoint stack-maps into per-inst vecs.
        let mut safepoint_slots: FxHashMap<Inst, Vec<Allocation>> = FxHashMap::default();
        for &(progpoint, slot) in &out.safepoint_slots {
            safepoint_slots
                .entry(progpoint.inst())
                .or_insert_with(|| vec![])
                .push(slot);
        }

        let mut last_inst = None;
        for block in 0..self.f.num_blocks() {
            let block = Block::new(block);
            for inst_or_edit in out.block_insts_and_edits(self.f, block) {
                match inst_or_edit {
                    InstOrEdit::Inst(inst) => {
                        debug_assert!(last_inst.is_none() || inst > last_inst.unwrap());
                        last_inst = Some(inst);
                        self.handle_inst(block, inst, &mut safepoint_slots, out);
                    }
                    InstOrEdit::Edit(edit) => self.handle_edit(block, edit),
                }
            }
        }
    }

    /// For each original instruction, create an `Op`.
    fn handle_inst(
        &mut self,
        block: Block,
        inst: Inst,
        safepoint_slots: &mut FxHashMap<Inst, Vec<Allocation>>,
        out: &Output,
    ) {
        // If this is a safepoint, then check the spillslots at this point.
        if self.f.requires_refs_on_stack(inst) {
            let allocs = safepoint_slots.remove(&inst).unwrap_or_else(|| vec![]);

            let checkinst = CheckerInst::Safepoint { inst, allocs };
            self.bb_insts.get_mut(&block).unwrap().push(checkinst);
        }

        // Skip normal checks if this is a branch: the blockparams do
        // not exist in post-regalloc code, and the edge-moves have to
        // be inserted before the branch rather than after.
        if !self.f.is_branch(inst) {
            // Instruction itself.
            let operands: Vec<_> = self.f.inst_operands(inst).iter().cloned().collect();
            let allocs: Vec<_> = out.inst_allocs(inst).iter().cloned().collect();
            let clobbers: Vec<_> = self.f.inst_clobbers(inst).iter().cloned().collect();
            let checkinst = CheckerInst::Op {
                inst,
                operands,
                allocs,
                clobbers,
            };
            trace!("checker: adding inst {:?}", checkinst);
            self.bb_insts.get_mut(&block).unwrap().push(checkinst);
        }
        // Instead, if this is a branch, emit a ParallelMove on each
        // outgoing edge as necessary to handle blockparams.
        else {
            for (i, &succ) in self.f.block_succs(block).iter().enumerate() {
                let args = self.f.branch_blockparams(block, inst, i);
                let params = self.f.block_params(succ);
                assert_eq!(args.len(), params.len());
                if args.len() > 0 {
                    self.edge_insts.get_mut(&(block, succ)).unwrap().push(
                        CheckerInst::ParallelMove {
                            into: params.iter().cloned().collect(),
                            from: args.iter().cloned().collect(),
                        },
                    );
                }
            }
        }
    }

    fn handle_edit(&mut self, block: Block, edit: &Edit) {
        trace!("checker: adding edit {:?}", edit);
        match edit {
            &Edit::Move { from, to } => {
                self.bb_insts
                    .get_mut(&block)
                    .unwrap()
                    .push(CheckerInst::Move { into: to, from });
            }
            _ => {}
        }
    }

    /// Perform the dataflow analysis to compute checker state at each BB entry.
    fn analyze(&mut self) {
        let mut queue = VecDeque::new();
        let mut queue_set = FxHashSet::default();
        for block in 0..self.f.num_blocks() {
            let block = Block::new(block);
            queue.push_back(block);
            queue_set.insert(block);
        }

        while !queue.is_empty() {
            let block = queue.pop_front().unwrap();
            queue_set.remove(&block);
            let mut state = self.bb_in.get(&block).cloned().unwrap();
            trace!("analyze: block {} has state {:?}", block.index(), state);
            for inst in self.bb_insts.get(&block).unwrap() {
                state.update(inst, self);
                trace!("analyze: inst {:?} -> state {:?}", inst, state);
            }

            for &succ in self.f.block_succs(block) {
                let mut new_state = state.clone();
                for edge_inst in self.edge_insts.get(&(block, succ)).unwrap() {
                    new_state.update(edge_inst, self);
                    trace!(
                        "analyze: succ {:?}: inst {:?} -> state {:?}",
                        succ,
                        edge_inst,
                        new_state
                    );
                }

                let cur_succ_in = self.bb_in.get(&succ).unwrap();
                trace!(
                    "meeting state {:?} for block {} with state {:?} for block {}",
                    new_state,
                    block.index(),
                    cur_succ_in,
                    succ.index()
                );
                new_state.meet_with(cur_succ_in);
                let changed = &new_state != cur_succ_in;
                trace!(" -> {:?}, changed {}", new_state, changed);

                if changed {
                    trace!(
                        "analyze: block {} state changed from {:?} to {:?}; pushing onto queue",
                        succ.index(),
                        cur_succ_in,
                        new_state
                    );
                    self.bb_in.insert(succ, new_state);
                    if !queue_set.contains(&succ) {
                        queue.push_back(succ);
                        queue_set.insert(succ);
                    }
                }
            }
        }
    }

    /// Using BB-start state computed by `analyze()`, step the checker state
    /// through each BB and check each instruction's register allocations
    /// for errors.
    fn find_errors(&self) -> Result<(), CheckerErrors> {
        let mut errors = vec![];
        for (block, input) in &self.bb_in {
            let mut state = input.clone();
            for inst in self.bb_insts.get(block).unwrap() {
                if let Err(e) = state.check(InstPosition::Before, inst, self) {
                    trace!("Checker error: {:?}", e);
                    errors.push(e);
                }
                state.update(inst, self);
                if let Err(e) = state.check(InstPosition::After, inst, self) {
                    trace!("Checker error: {:?}", e);
                    errors.push(e);
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CheckerErrors { errors })
        }
    }

    /// Find any errors, returning `Err(CheckerErrors)` with all errors found
    /// or `Ok(())` otherwise.
    pub fn run(mut self) -> Result<(), CheckerErrors> {
        self.analyze();
        let result = self.find_errors();

        trace!("=== CHECKER RESULT ===");
        fn print_state(state: &CheckerState) {
            let mut s = vec![];
            for (alloc, state) in &state.allocations {
                s.push(format!("{} := {}", alloc, state));
            }
            trace!("    {{ {} }}", s.join(", "))
        }
        for vreg in self.f.reftype_vregs() {
            trace!("  REF: {}", vreg);
        }
        for bb in 0..self.f.num_blocks() {
            let bb = Block::new(bb);
            trace!("block{}:", bb.index());
            let insts = self.bb_insts.get(&bb).unwrap();
            let mut state = self.bb_in.get(&bb).unwrap().clone();
            print_state(&state);
            for inst in insts {
                match inst {
                    &CheckerInst::Op {
                        inst,
                        ref operands,
                        ref allocs,
                        ref clobbers,
                    } => {
                        trace!(
                            "  inst{}: {:?} ({:?}) clobbers:{:?}",
                            inst.index(),
                            operands,
                            allocs,
                            clobbers
                        );
                    }
                    &CheckerInst::Move { from, into } => {
                        trace!("    {} -> {}", from, into);
                    }
                    &CheckerInst::Safepoint { ref allocs, .. } => {
                        let mut slotargs = vec![];
                        for &slot in allocs {
                            slotargs.push(format!("{}", slot));
                        }
                        trace!("    safepoint: {}", slotargs.join(", "));
                    }
                    &CheckerInst::ParallelMove { .. } => {
                        panic!("unexpected parallel_move in body (non-edge)")
                    }
                }
                state.update(inst, &self);
                print_state(&state);
            }

            for &succ in self.f.block_succs(bb) {
                trace!("  succ {:?}:", succ);
                let mut state = state.clone();
                for edge_inst in self.edge_insts.get(&(bb, succ)).unwrap() {
                    match edge_inst {
                        &CheckerInst::ParallelMove { ref from, ref into } => {
                            trace!("    parallel_move {:?} -> {:?}", from, into);
                        }
                        _ => panic!("unexpected edge_inst: not a parallel move"),
                    }
                    state.update(edge_inst, &self);
                    print_state(&state);
                }
            }
        }

        result
    }
}
