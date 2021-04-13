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
    OperandPolicy, OperandPos, Output, ProgPoint, VReg,
};

use std::collections::{HashMap, VecDeque};
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
            (&CheckerValue::Reg(r1, ref1), &CheckerValue::Reg(r2, ref2)) if r1 == r2 => {
                CheckerValue::Reg(r1, ref1 || ref2)
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
            CheckerValue::Reg(r, _) => write!(f, "{}", r),
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
                        (OperandPos::Before, InstPosition::Before)
                        | (OperandPos::Both, InstPosition::Before) => true,
                        (OperandPos::After, InstPosition::After)
                        | (OperandPos::Both, InstPosition::After) => true,
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
            _ => {}
        }
        Ok(())
    }

    /// Update according to instruction.
    fn update(&mut self, checkinst: &CheckerInst) {
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
                ..
            } => {
                for (op, alloc) in operands.iter().zip(allocs.iter()) {
                    if op.kind() != OperandKind::Def {
                        continue;
                    }
                    self.allocations
                        .insert(*alloc, CheckerValue::Reg(op.vreg(), false));
                }
            }
            &CheckerInst::BlockParams {
                ref vregs,
                ref allocs,
                ..
            } => {
                for (vreg, alloc) in vregs.iter().zip(allocs.iter()) {
                    self.allocations
                        .insert(*alloc, CheckerValue::Reg(*vreg, false));
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
    },

    /// The top of a block with blockparams. We define the given vregs
    /// into the given allocations.
    BlockParams {
        block: Block,
        vregs: Vec<VReg>,
        allocs: Vec<Allocation>,
    },
}

#[derive(Debug)]
pub struct Checker<'a, F: Function> {
    f: &'a F,
    bb_in: HashMap<Block, CheckerState>,
    bb_insts: HashMap<Block, Vec<CheckerInst>>,
}

impl<'a, F: Function> Checker<'a, F> {
    /// Create a new checker for the given function, initializing CFG
    /// info immediately.  The client should call the `add_*()`
    /// methods to add abstract instructions to each BB before
    /// invoking `run()` to check for errors.
    pub fn new(f: &'a F) -> Checker<'a, F> {
        let mut bb_in = HashMap::new();
        let mut bb_insts = HashMap::new();

        for block in 0..f.blocks() {
            let block = Block::new(block);
            bb_in.insert(block, Default::default());
            bb_insts.insert(block, vec![]);
        }

        Checker { f, bb_in, bb_insts }
    }

    /// Build the list of checker instructions based on the given func
    /// and allocation results.
    pub fn prepare(&mut self, out: &Output) {
        debug!("checker: out = {:?}", out);
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

                // Instruction itself.
                let operands: Vec<_> = self.f.inst_operands(inst).iter().cloned().collect();
                let allocs: Vec<_> = out.inst_allocs(inst).iter().cloned().collect();
                let checkinst = CheckerInst::Op {
                    inst,
                    operands,
                    allocs,
                };
                debug!("checker: adding inst {:?}", checkinst);
                self.bb_insts.get_mut(&block).unwrap().push(checkinst);

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
                &Edit::Move { from, to, .. } => {
                    self.bb_insts
                        .get_mut(&block)
                        .unwrap()
                        .push(CheckerInst::Move { into: to, from });
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
        queue.push_back(self.f.entry_block());

        while !queue.is_empty() {
            let block = queue.pop_front().unwrap();
            let mut state = self.bb_in.get(&block).cloned().unwrap();
            debug!("analyze: block {} has state {:?}", block.index(), state);
            for inst in self.bb_insts.get(&block).unwrap() {
                state.update(inst);
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
                    queue.push_back(succ);
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
                state.update(inst);
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
                    } => {
                        debug!("  inst{}: {:?} ({:?})", inst.index(), operands, allocs);
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
                }
                state.update(inst);
                print_state(&state);
            }
        }

        result
    }
}
