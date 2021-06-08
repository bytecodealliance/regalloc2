/*
 * The following code is derived from `lib/src/checker.rs` in the
 * regalloc.rs project
 * (https://github.com/bytecodealliance/regalloc.rs). regalloc.rs is
 * also licensed under Apache-2.0 with the LLVM exception, as the rest
 * of regalloc2's non-Ion-derived code is.
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
//! Operand policies (fixed register, register, any) are also checked
//! at each operand.
//!
//! The dataflow analysis state at each program point is:
//!
//!   - map of: Allocation -> lattice value  (top > Vn symbols (unordered) > bottom)
//!
//! And the transfer functions for instructions are:
//!
//!   - `Edit::Move` inserted by RA:       [ alloc_d := alloc_s ]
//!
//!       A[alloc_d] := A[alloc_s]
//!
//!   - phi-node          [ V_i := phi block_j:V_j, block_k:V_k, ... ]
//!     with allocations  [ A_i := phi block_j:A_j, block_k:A_k, ... ]
//!     (N.B.: phi-nodes are not semantically present in the final
//!      machine code, but we include their allocations so that this
//!      checker can work)
//!
//!       A[A_i] := meet(A_j, A_k, ...)
//!
//!   - statement in pre-regalloc function [ V_i := op V_j, V_k, ... ]
//!     with allocated form                [ A_i := op A_j, A_k, ... ]
//!
//!       A[A_i] := `V_i`
//!
//!     In other words, a statement, even after allocation, generates
//!     a symbol that corresponds to its original virtual-register
//!     def.
//!
//!     (N.B.: moves in pre-regalloc function fall into this last case
//!     -- they are "just another operation" and generate a new
//!     symbol)
//!
//! At control-flow join points, the symbols meet using a very simple
//! lattice meet-function: two different symbols in the same
//! allocation meet to "conflicted"; otherwise, the symbol meets with
//! itself to produce itself (reflexivity).
//!
//! To check correctness, we first find the dataflow fixpoint with the
//! above lattice and transfer/meet functions. Then, at each op, we
//! examine the dataflow solution at the preceding program point, and
//! check that the allocation for each op arg (input/use) contains the
//! symbol corresponding to the original virtual register specified
//! for this arg.

#![allow(dead_code)]

use crate::{
    Allocation, AllocationKind, Block, Edit, Function, Inst, InstPosition, Operand, OperandKind,
    OperandPolicy, OperandPos, Output, PReg, ProgPoint, SpillSlot, VReg,
};

use std::collections::{HashMap, HashSet, VecDeque};
use std::default::Default;
use std::hash::Hash;
use std::result::Result;

use log::debug;

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
    IncorrectValueInAllocation {
        inst: Inst,
        op: Operand,
        alloc: Allocation,
        actual: VReg,
    },
    PolicyViolated {
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
        slot: SpillSlot,
    },
    NonRefValueInStackmap {
        inst: Inst,
        slot: SpillSlot,
        vreg: VReg,
    },
}

/// Abstract state for an allocation.
///
/// Forms a lattice with \top (`Unknown`), \bot (`Conflicted`), and a
/// number of mutually unordered value-points in between, one per real
/// or virtual register. Any two different registers meet to \bot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CheckerValue {
    /// "top" value: this storage slot has no known value.
    Unknown,
    /// "bottom" value: this storage slot has a conflicted value.
    Conflicted,
    /// Reg: this storage slot has a value that originated as a def
    /// into the given virtual register.
    ///
    /// The boolean flag indicates whether the value is
    /// reference-typed.
    Reg(VReg, bool),
}

impl Default for CheckerValue {
    fn default() -> CheckerValue {
        CheckerValue::Unknown
    }
}

impl CheckerValue {
    /// Meet function of the abstract-interpretation value lattice.
    fn meet(&self, other: &CheckerValue) -> CheckerValue {
        match (self, other) {
            (&CheckerValue::Unknown, _) => *other,
            (_, &CheckerValue::Unknown) => *self,
            (&CheckerValue::Conflicted, _) => *self,
            (_, &CheckerValue::Conflicted) => *other,
            (&CheckerValue::Reg(r1, ref1), &CheckerValue::Reg(r2, ref2))
                if r1 == r2 && ref1 == ref2 =>
            {
                CheckerValue::Reg(r1, ref1)
            }
            _ => {
                log::debug!("{:?} and {:?} meet to Conflicted", self, other);
                CheckerValue::Conflicted
            }
        }
    }
}

/// State that steps through program points as we scan over the instruction stream.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CheckerState {
    allocations: HashMap<Allocation, CheckerValue>,
}

impl Default for CheckerState {
    fn default() -> CheckerState {
        CheckerState {
            allocations: HashMap::new(),
        }
    }
}

impl std::fmt::Display for CheckerValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CheckerValue::Unknown => write!(f, "?"),
            CheckerValue::Conflicted => write!(f, "!"),
            CheckerValue::Reg(r, false) => write!(f, "{}", r),
            CheckerValue::Reg(r, true) => write!(f, "{}/ref", r),
        }
    }
}

fn merge_map<K: Copy + Clone + PartialEq + Eq + Hash>(
    into: &mut HashMap<K, CheckerValue>,
    from: &HashMap<K, CheckerValue>,
) {
    for (k, v) in from {
        let into_v = into.entry(*k).or_insert(Default::default());
        let merged = into_v.meet(v);
        *into_v = merged;
    }
}

impl CheckerState {
    /// Create a new checker state.
    fn new() -> CheckerState {
        Default::default()
    }

    /// Merge this checker state with another at a CFG join-point.
    fn meet_with(&mut self, other: &CheckerState) {
        merge_map(&mut self.allocations, &other.allocations);
    }

    fn check_val(
        &self,
        inst: Inst,
        op: Operand,
        alloc: Allocation,
        val: CheckerValue,
        allocs: &[Allocation],
    ) -> Result<(), CheckerError> {
        if alloc == Allocation::none() {
            return Err(CheckerError::MissingAllocation { inst, op });
        }

        match val {
            CheckerValue::Unknown => {
                return Err(CheckerError::UnknownValueInAllocation { inst, op, alloc });
            }
            CheckerValue::Conflicted => {
                return Err(CheckerError::ConflictedValueInAllocation { inst, op, alloc });
            }
            CheckerValue::Reg(r, _) if r != op.vreg() => {
                return Err(CheckerError::IncorrectValueInAllocation {
                    inst,
                    op,
                    alloc,
                    actual: r,
                });
            }
            _ => {}
        }

        self.check_policy(inst, op, alloc, allocs)?;

        Ok(())
    }

    /// Check an instruction against this state. This must be called
    /// twice: once with `InstPosition::Before`, and once with
    /// `InstPosition::After` (after updating state with defs).
    fn check(&self, pos: InstPosition, checkinst: &CheckerInst) -> Result<(), CheckerError> {
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
                    .any(|op| matches!(op.policy(), OperandPolicy::Reuse(_)));
                if has_reused_input && pos == InstPosition::After {
                    return Ok(());
                }

                // For each operand, check (i) that the allocation
                // contains the expected vreg, and (ii) that it meets
                // the requirements of the OperandPolicy.
                for (op, alloc) in operands.iter().zip(allocs.iter()) {
                    let is_here = match (op.pos(), pos) {
                        (OperandPos::Before, InstPosition::Before) => true,
                        (OperandPos::After, InstPosition::After) => true,
                        _ => false,
                    };
                    if !is_here {
                        continue;
                    }
                    if op.kind() == OperandKind::Def {
                        continue;
                    }

                    let val = self
                        .allocations
                        .get(alloc)
                        .cloned()
                        .unwrap_or(Default::default());
                    debug!(
                        "checker: checkinst {:?}: op {:?}, alloc {:?}, checker value {:?}",
                        checkinst, op, alloc, val
                    );
                    self.check_val(inst, *op, *alloc, val, allocs)?;
                }
            }
            &CheckerInst::Safepoint { inst, ref slots } => {
                for &slot in slots {
                    let alloc = Allocation::stack(slot);
                    let val = self
                        .allocations
                        .get(&alloc)
                        .cloned()
                        .unwrap_or(Default::default());
                    debug!(
                        "checker: checkinst {:?}: safepoint slot {}, checker value {:?}",
                        checkinst, slot, val
                    );

                    match val {
                        CheckerValue::Unknown => {}
                        CheckerValue::Conflicted => {
                            return Err(CheckerError::ConflictedValueInStackmap { inst, slot });
                        }
                        CheckerValue::Reg(vreg, false) => {
                            return Err(CheckerError::NonRefValueInStackmap { inst, slot, vreg });
                        }
                        CheckerValue::Reg(_, true) => {}
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Update according to instruction.
    fn update<'a, F: Function>(&mut self, checkinst: &CheckerInst, checker: &Checker<'a, F>) {
        match checkinst {
            &CheckerInst::Move { into, from } => {
                let val = self
                    .allocations
                    .get(&from)
                    .cloned()
                    .unwrap_or(Default::default());
                debug!(
                    "checker: checkinst {:?} updating: move {:?} -> {:?} val {:?}",
                    checkinst, from, into, val
                );
                self.allocations.insert(into, val);
            }
            &CheckerInst::Op {
                ref operands,
                ref allocs,
                ref clobbers,
                ..
            } => {
                for (op, alloc) in operands.iter().zip(allocs.iter()) {
                    if op.kind() != OperandKind::Def {
                        continue;
                    }
                    let reftyped = checker.reftyped_vregs.contains(&op.vreg());
                    self.allocations
                        .insert(*alloc, CheckerValue::Reg(op.vreg(), reftyped));
                }
                for clobber in clobbers {
                    self.allocations.remove(&Allocation::reg(*clobber));
                }
            }
            &CheckerInst::BlockParams {
                ref vregs,
                ref allocs,
                ..
            } => {
                for (vreg, alloc) in vregs.iter().zip(allocs.iter()) {
                    let reftyped = checker.reftyped_vregs.contains(vreg);
                    self.allocations
                        .insert(*alloc, CheckerValue::Reg(*vreg, reftyped));
                }
            }
            &CheckerInst::DefAlloc { alloc, vreg } => {
                let reftyped = checker.reftyped_vregs.contains(&vreg);
                self.allocations
                    .insert(alloc, CheckerValue::Reg(vreg, reftyped));
            }
            &CheckerInst::Safepoint { ref slots, .. } => {
                for (alloc, value) in &mut self.allocations {
                    if let CheckerValue::Reg(_, true) = *value {
                        if alloc.is_reg() {
                            *value = CheckerValue::Conflicted;
                        } else if alloc.is_stack() && !slots.contains(&alloc.as_stack().unwrap()) {
                            *value = CheckerValue::Conflicted;
                        }
                    }
                }
            }
        }
    }

    fn check_policy(
        &self,
        inst: Inst,
        op: Operand,
        alloc: Allocation,
        allocs: &[Allocation],
    ) -> Result<(), CheckerError> {
        match op.policy() {
            OperandPolicy::Any => {}
            OperandPolicy::Reg => {
                if alloc.kind() != AllocationKind::Reg {
                    return Err(CheckerError::AllocationIsNotReg { inst, op, alloc });
                }
            }
            OperandPolicy::Stack => {
                if alloc.kind() != AllocationKind::Stack {
                    return Err(CheckerError::AllocationIsNotStack { inst, op, alloc });
                }
            }
            OperandPolicy::FixedReg(preg) => {
                if alloc != Allocation::reg(preg) {
                    return Err(CheckerError::AllocationIsNotFixedReg { inst, op, alloc });
                }
            }
            OperandPolicy::Reuse(idx) => {
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

    /// A regular instruction with fixed use and def slots. Contains
    /// both the original operands (as given to the regalloc) and the
    /// allocation results.
    Op {
        inst: Inst,
        operands: Vec<Operand>,
        allocs: Vec<Allocation>,
        clobbers: Vec<PReg>,
    },

    /// The top of a block with blockparams. We define the given vregs
    /// into the given allocations.
    BlockParams {
        block: Block,
        vregs: Vec<VReg>,
        allocs: Vec<Allocation>,
    },

    /// Define an allocation's contents. Like BlockParams but for one
    /// allocation. Used sometimes when moves are elided but ownership
    /// of a value is logically transferred to a new vreg.
    DefAlloc { alloc: Allocation, vreg: VReg },

    /// A safepoint, with the given SpillSlots specified as containing
    /// reftyped values. All other reftyped values become invalid.
    Safepoint { inst: Inst, slots: Vec<SpillSlot> },
}

#[derive(Debug)]
pub struct Checker<'a, F: Function> {
    f: &'a F,
    bb_in: HashMap<Block, CheckerState>,
    bb_insts: HashMap<Block, Vec<CheckerInst>>,
    reftyped_vregs: HashSet<VReg>,
}

impl<'a, F: Function> Checker<'a, F> {
    /// Create a new checker for the given function, initializing CFG
    /// info immediately.  The client should call the `add_*()`
    /// methods to add abstract instructions to each BB before
    /// invoking `run()` to check for errors.
    pub fn new(f: &'a F) -> Checker<'a, F> {
        let mut bb_in = HashMap::new();
        let mut bb_insts = HashMap::new();
        let mut reftyped_vregs = HashSet::new();

        for block in 0..f.blocks() {
            let block = Block::new(block);
            bb_in.insert(block, Default::default());
            bb_insts.insert(block, vec![]);
        }

        for &vreg in f.reftype_vregs() {
            reftyped_vregs.insert(vreg);
        }

        Checker {
            f,
            bb_in,
            bb_insts,
            reftyped_vregs,
        }
    }

    /// Build the list of checker instructions based on the given func
    /// and allocation results.
    pub fn prepare(&mut self, out: &Output) {
        debug!("checker: out = {:?}", out);
        // Preprocess safepoint stack-maps into per-inst vecs.
        let mut safepoint_slots: HashMap<Inst, Vec<SpillSlot>> = HashMap::new();
        for &(progpoint, slot) in &out.safepoint_slots {
            safepoint_slots
                .entry(progpoint.inst())
                .or_insert_with(|| vec![])
                .push(slot);
        }

        // For each original instruction, create an `Op`.
        let mut last_inst = None;
        let mut insert_idx = 0;
        for block in 0..self.f.blocks() {
            let block = Block::new(block);
            for inst in self.f.block_insns(block).iter() {
                assert!(last_inst.is_none() || inst > last_inst.unwrap());
                last_inst = Some(inst);

                // Any inserted edits before instruction.
                self.handle_edits(block, out, &mut insert_idx, ProgPoint::before(inst));

                // If this is a safepoint, then check the spillslots at this point.
                if self.f.is_safepoint(inst) {
                    let slots = safepoint_slots.remove(&inst).unwrap_or_else(|| vec![]);

                    let checkinst = CheckerInst::Safepoint { inst, slots };
                    self.bb_insts.get_mut(&block).unwrap().push(checkinst);
                }

                // Skip if this is a branch: the blockparams do not
                // exist in post-regalloc code, and the edge-moves
                // have to be inserted before the branch rather than
                // after.
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
                    debug!("checker: adding inst {:?}", checkinst);
                    self.bb_insts.get_mut(&block).unwrap().push(checkinst);
                }

                // Any inserted edits after instruction.
                self.handle_edits(block, out, &mut insert_idx, ProgPoint::after(inst));
            }
        }
    }

    fn handle_edits(&mut self, block: Block, out: &Output, idx: &mut usize, pos: ProgPoint) {
        while *idx < out.edits.len() && out.edits[*idx].0 <= pos {
            let &(edit_pos, ref edit) = &out.edits[*idx];
            *idx += 1;
            if edit_pos < pos {
                continue;
            }
            debug!("checker: adding edit {:?} at pos {:?}", edit, pos);
            match edit {
                &Edit::Move { from, to, to_vreg } => {
                    self.bb_insts
                        .get_mut(&block)
                        .unwrap()
                        .push(CheckerInst::Move { into: to, from });
                    if let Some(vreg) = to_vreg {
                        self.bb_insts
                            .get_mut(&block)
                            .unwrap()
                            .push(CheckerInst::DefAlloc { alloc: to, vreg });
                    }
                }
                &Edit::DefAlloc { alloc, vreg } => {
                    self.bb_insts
                        .get_mut(&block)
                        .unwrap()
                        .push(CheckerInst::DefAlloc { alloc, vreg });
                }
                &Edit::BlockParams {
                    ref vregs,
                    ref allocs,
                } => {
                    let inst = CheckerInst::BlockParams {
                        block,
                        vregs: vregs.clone(),
                        allocs: allocs.clone(),
                    };
                    self.bb_insts.get_mut(&block).unwrap().push(inst);
                }
            }
        }
    }

    /// Perform the dataflow analysis to compute checker state at each BB entry.
    fn analyze(&mut self) {
        let mut queue = VecDeque::new();
        let mut queue_set = HashSet::new();
        for block in 0..self.f.blocks() {
            let block = Block::new(block);
            queue.push_back(block);
            queue_set.insert(block);
        }

        while !queue.is_empty() {
            let block = queue.pop_front().unwrap();
            queue_set.remove(&block);
            let mut state = self.bb_in.get(&block).cloned().unwrap();
            debug!("analyze: block {} has state {:?}", block.index(), state);
            for inst in self.bb_insts.get(&block).unwrap() {
                state.update(inst, self);
                debug!("analyze: inst {:?} -> state {:?}", inst, state);
            }

            for &succ in self.f.block_succs(block) {
                let cur_succ_in = self.bb_in.get(&succ).unwrap();
                let mut new_state = state.clone();
                new_state.meet_with(cur_succ_in);
                let changed = &new_state != cur_succ_in;
                if changed {
                    debug!(
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
                if let Err(e) = state.check(InstPosition::Before, inst) {
                    debug!("Checker error: {:?}", e);
                    errors.push(e);
                }
                state.update(inst, self);
                if let Err(e) = state.check(InstPosition::After, inst) {
                    debug!("Checker error: {:?}", e);
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

        debug!("=== CHECKER RESULT ===");
        fn print_state(state: &CheckerState) {
            let mut s = vec![];
            for (alloc, state) in &state.allocations {
                s.push(format!("{} := {}", alloc, state));
            }
            debug!("    {{ {} }}", s.join(", "))
        }
        for vreg in self.f.reftype_vregs() {
            debug!("  REF: {}", vreg);
        }
        for bb in 0..self.f.blocks() {
            let bb = Block::new(bb);
            debug!("block{}:", bb.index());
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
                        debug!(
                            "  inst{}: {:?} ({:?}) clobbers:{:?}",
                            inst.index(),
                            operands,
                            allocs,
                            clobbers
                        );
                    }
                    &CheckerInst::Move { from, into } => {
                        debug!("    {} -> {}", from, into);
                    }
                    &CheckerInst::BlockParams {
                        ref vregs,
                        ref allocs,
                        ..
                    } => {
                        let mut args = vec![];
                        for (vreg, alloc) in vregs.iter().zip(allocs.iter()) {
                            args.push(format!("{}:{}", vreg, alloc));
                        }
                        debug!("    blockparams: {}", args.join(", "));
                    }
                    &CheckerInst::DefAlloc { alloc, vreg } => {
                        debug!("    defalloc: {}:{}", vreg, alloc);
                    }
                    &CheckerInst::Safepoint { ref slots, .. } => {
                        let mut slotargs = vec![];
                        for &slot in slots {
                            slotargs.push(format!("{}", slot));
                        }
                        debug!("    safepoint: {}", slotargs.join(", "));
                    }
                }
                state.update(inst, &self);
                print_state(&state);
            }
        }

        result
    }
}
