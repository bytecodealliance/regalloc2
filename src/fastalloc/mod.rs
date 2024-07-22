use core::convert::TryInto;
use core::iter::FromIterator;
use core::ops::{Index, IndexMut};
use crate::{Block, Inst, OperandKind, Operand, PReg, RegClass, VReg, SpillSlot, AllocationKind, OperandConstraint, InstPosition};
use crate::{Function, MachineEnv, ssa::validate_ssa, ProgPoint, Edit, Output};
use crate::{cfg::CFGInfo, RegAllocError, Allocation, ion::Stats};
use alloc::collections::{BTreeSet, VecDeque};
use alloc::vec::Vec;
use hashbrown::HashSet;

mod lru;
mod iter;
use lru::*;
use iter::*;

#[derive(Debug)]
struct Allocs {
    allocs: Vec<Allocation>,
    /// `inst_alloc_offsets[i]` is the offset into `allocs` for the allocations of
    /// instruction `i`'s operands.
    inst_alloc_offsets: Vec<u32>,
}

impl Allocs {
    fn new<F: Function>(func: &F, env: &MachineEnv) -> Self {
        // The number of operands is <= number of virtual registers
        // It can be lesser in the case where virtual registers are used multiple
        // times in a single instruction.
        let mut allocs = Vec::with_capacity(func.num_vregs());
        let mut inst_alloc_offsets = Vec::with_capacity(func.num_vregs());
        for inst in 0..func.num_insts() {
            let operands_len = func.inst_operands(Inst::new(inst)).len() as u32;
            inst_alloc_offsets.push(allocs.len() as u32);
            for _ in 0..operands_len {
                allocs.push(Allocation::none());
            }
        }
        Self {
            allocs,
            inst_alloc_offsets,
        }
    }
}

impl Index<(usize, usize)> for Allocs {
    type Output = Allocation;

    /// Retrieve the allocation for operand `idx.1` at instruction `idx.0`
    fn index(&self, idx: (usize, usize)) -> &Allocation {
        &self.allocs[self.inst_alloc_offsets[idx.0] as usize + idx.1]
    }
}

impl IndexMut<(usize, usize)> for Allocs {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Allocation {
        &mut self.allocs[self.inst_alloc_offsets[idx.0] as usize + idx.1]
    }
}

#[derive(Debug)]
pub struct Env<'a, F: Function> {
    func: &'a F,

    /// The current allocations for all virtual registers.
    vreg_allocs: Vec<Allocation>,
    /// Spillslots for all virtual registers.
    /// `vreg_spillslots[i]` is the spillslot for virtual register `i`.
    vreg_spillslots: Vec<SpillSlot>,
    /// The virtual registers that are currently live.
    live_vregs: HashSet<VReg>,
    /// Allocatable free physical registers for classes Int, Float, and Vector, respectively.
    freepregs: PartedByRegClass<BTreeSet<PReg>>,
    /// Least-recently-used caches for register classes Int, Float, and Vector, respectively.
    lrus: Lrus,
    /// `vreg_in_preg[i]` is the virtual register currently in the physical register
    /// with index `i`.
    vreg_in_preg: Vec<VReg>,
    /// For parallel moves from branch args to block param spillslots.
    temp_spillslots: PartedByRegClass<Vec<SpillSlot>>,
    /// The edits to be inserted before the currently processed instruction.
    inst_pre_edits: VecDeque<(ProgPoint, Edit, RegClass)>,
    /// The edits to be inserted after the currently processed instruction.
    inst_post_edits: VecDeque<(ProgPoint, Edit, RegClass)>,
    /// All the allocatables registers that were used for one thing or the other
    /// but need to be freed after the current instruction has completed processing,
    /// not immediately, like allocatable registers used as scratch registers.
    /// 
    /// This is used to keep track of them so that they can be marked as free for reallocation
    /// after the instruction has completed processing.
    free_after_curr_inst: HashSet<PReg>,
    /// The virtual registers of operands that have been allocated in the current instruction.
    /// This needs to be kept track of to generate the correct moves in the case where a
    /// single virtual register is used multiple times in a single instruction with
    /// different constraints.
    vregs_allocd_in_curr_inst: HashSet<VReg>,
    /// Physical registers that were used for late def operands and now free to be
    /// reused for early operands in the current instruction.
    ///
    /// After late defs have been allocated, rather than returning their registers to
    /// the free register list, it is added here to avoid the registers being used as
    /// scratch registers.
    /// 
    /// For example, consider the following:
    /// def v0, use v1
    /// If the processing of v1 requires a stack-to-stack move, then a scratch register is
    /// used and the instruction becomes:
    /// def v0, use v1
    /// move from stack0 to p0
    /// move from p0 to stack1
    /// 
    /// Since scratch registers may be drawn from the free register list and v0 will be allocated and
    /// deallocated before v1, then it's possible for the scratch register p0 to be v0's allocation,
    /// which is incorrect because p0 will end up holding whatever is in stack0, not v0.
    /// `freed_def_regs` avoids this by allowing the late def registers to be reused without making it
    /// possible for this scratch register scenario to happen.
    freed_def_pregs: PartedByRegClass<BTreeSet<PReg>>,
    /// Used to keep track of which used vregs are being used for the first time
    /// in the instruction. This is used to determine whether or not reused operands
    /// for reuse-input constraints should be restored after an instruction.
    first_use: HashSet<VReg>,
    /// Used to keep track of which allocations have been used by use operands in the
    /// current instruction. This is to determine whether or not an allocation
    /// for a reuse operand was reused by a use operand, and make decisions on
    /// whether or not to free the allocation.
    allocs_used_by_use_ops: HashSet<Allocation>,
    fixed_stack_slots: Vec<PReg>,

    // Output.
    allocs: Allocs,
    edits: VecDeque<(ProgPoint, Edit)>,
    num_spillslots: u32,
    stats: Stats,
}

impl<'a, F: Function> Env<'a, F> {
    fn new(func: &'a F, env: &'a MachineEnv) -> Self {
        trace!("multispillslots_named_by_last_slot: {:?}", func.multi_spillslot_named_by_last_slot());
        let mut regs = [
            env.preferred_regs_by_class[RegClass::Int as usize].clone(),
            env.preferred_regs_by_class[RegClass::Float as usize].clone(),
            env.preferred_regs_by_class[RegClass::Vector as usize].clone(),
        ];
        regs[0].extend(env.non_preferred_regs_by_class[RegClass::Int as usize].iter().cloned());
        regs[1].extend(env.non_preferred_regs_by_class[RegClass::Float as usize].iter().cloned());
        regs[2].extend(env.non_preferred_regs_by_class[RegClass::Vector as usize].iter().cloned());
        use alloc::vec;
        trace!("{:?}", env);
        Self {
            func,
            vreg_allocs: vec![Allocation::none(); func.num_vregs()],
            vreg_spillslots: vec![SpillSlot::invalid(); func.num_vregs()],
            live_vregs: HashSet::with_capacity(func.num_vregs()),
            freepregs: PartedByRegClass {
                items: [
                    BTreeSet::from_iter(regs[0].clone()),
                    BTreeSet::from_iter(regs[1].clone()),
                    BTreeSet::from_iter(regs[2].clone()),
                ]
            },
            lrus: Lrus::new(
                regs[0].len(),
                regs[1].len(),
                regs[2].len()
            ),
            vreg_in_preg: vec![VReg::invalid(); PReg::NUM_INDEX],
            fixed_stack_slots: env.fixed_stack_slots.clone(),
            temp_spillslots: PartedByRegClass { items: [
                Vec::with_capacity(func.num_vregs()),
                Vec::with_capacity(func.num_vregs()),
                Vec::with_capacity(func.num_vregs()),
            ] },
            inst_pre_edits: VecDeque::new(),
            inst_post_edits: VecDeque::new(),
            free_after_curr_inst: HashSet::new(),
            vregs_allocd_in_curr_inst: HashSet::new(),
            freed_def_pregs: PartedByRegClass { items: [BTreeSet::new(), BTreeSet::new(), BTreeSet::new()] },
            first_use: HashSet::new(),
            allocs_used_by_use_ops: HashSet::new(),
            allocs: Allocs::new(func, env),
            edits: VecDeque::new(),
            num_spillslots: 0,
            stats: Stats::default(),
        }
    }

    fn is_stack(&self, alloc: Allocation) -> bool {
        if alloc.is_stack() {
            return true;
        }
        if alloc.is_reg() {
            return self.fixed_stack_slots.contains(&alloc.as_reg().unwrap());
        }
        false
    }

    fn process_edit(&mut self, point: ProgPoint, edit: Edit, class: RegClass) {
        trace!("Processing edit: {:?}", edit);
        let Edit::Move { from, to } = edit;
        if self.is_stack(from) && self.is_stack(to) {
            let scratch_reg = *self.freepregs[class].last().expect("Allocation impossible?");
            trace!("Edit is stack-to-stack, generating two moves with a scratch register {:?}", scratch_reg);
            let scratch_alloc = Allocation::reg(scratch_reg);
            trace!("Processed Edit: {:?}", (point, Edit::Move {
                from: scratch_alloc,
                to,
            }));
            self.edits.push_front((point, Edit::Move {
                from: scratch_alloc,
                to,
            }));
            trace!("Processed Edit: {:?}", (point, Edit::Move {
                from,
                to: scratch_alloc,
            }));
            self.edits.push_front((point, Edit::Move {
                from,
                to: scratch_alloc,
            }));
        } else {
            trace!("Edit is not stack-to-stack. Adding it directly:");
            trace!("Processed Edit: {:?}", (point, Edit::Move {
                from,
                to,
            }));
            self.edits.push_front((point, Edit::Move {
                from,
                to,
            }));
        }
    }

    fn add_move_later(&mut self, inst: Inst, from: Allocation, to: Allocation, class: RegClass, pos: InstPosition, prepend: bool) {
        let target_edits = match pos {
            InstPosition::After => &mut self.inst_post_edits,
            InstPosition::Before => &mut self.inst_pre_edits
        };
        trace!("Recording edit to add later: {:?}", (ProgPoint::new(inst, pos), Edit::Move {
            from,
            to
        }, class));
        // The sorting out of stack-to-stack moves will be done when the instruction's
        // edits are processed after all operands have been allocated.
        if prepend {
            target_edits.push_front((ProgPoint::new(inst, pos), Edit::Move {
                from,
                to,
            }, class));
        } else {
            target_edits.push_back((ProgPoint::new(inst, pos), Edit::Move {
                from,
                to,
            }, class));
        }
    }

    /*
    fn add_move(&mut self, inst: Inst, from: Allocation, to: Allocation, class: RegClass, pos: InstPosition) {
        if self.is_stack(from) && self.is_stack(to) {
            let scratch_reg = if self.freepregs[class].is_empty() {
                self.evict_any_reg(inst, class)
            } else {
                // The physical register used as a scratch register here has to be explicitly
                // removed from the free registers list. This is to avoid scenarios like the
                // following:
                // 1. def v0, use v1, use v2
                // If v2 needs to be moved from a stack slot to some other specific stack slot
                // for the constraints to be satisified, then some scratch register p0 has to be
                // used to avoid stack-to-stack moves. If p0 is not a dedicated scratch register,
                // then the following sequence is possible:
                // First, p0 is used as a scratch register to move from the stack to whatever stackslot
                // v2 is supposed to be in and the moves are inserted before the instruction.
                // Second, v1 is allocated to p0 (thinking p0 is free). But before the instruction,
                // p0 is overwritten by v2, so v2 will be used instead of v1.
                //
                // So, rather than leave the register in the free list, it's removed from this list
                // and added back after the complete processing of all the operands in the instruction.
                self.freepregs[class].pop().unwrap()
            };
            self.free_after_curr_inst.push(scratch_reg);
            // Remove the scratch register from the LRU to stop it from
            // being used as an allocatable register for the current instruction.
            // It should be added back after processing the current instruction.
            self.lrus[class].remove(scratch_reg.hw_enc());
            let scratch_alloc = Allocation::reg(scratch_reg);
            let mut target_edits = &mut self.edits;
            if pos == InstPosition::Before {
                target_edits = &mut self.pre_edits;
            }
            // Edits are added in reverse order because the edits
            // will be reversed when all allocation is completed.
            trace!("Edit: {:?}", (ProgPoint::new(inst, pos), Edit::Move {
                from: scratch_alloc,
                to,
            }));
            target_edits.push((ProgPoint::new(inst, pos), Edit::Move {
                from: scratch_alloc,
                to,
            }));
            trace!("Edit: {:?}", (ProgPoint::new(inst, pos), Edit::Move {
                from,
                to: scratch_alloc,
            }));
            target_edits.push((ProgPoint::new(inst, pos), Edit::Move {
                from,
                to: scratch_alloc,
            }))
            
        } else {
            let mut target_edits = &mut self.edits;
            if pos == InstPosition::Before {
                target_edits = &mut self.pre_edits;
            }
            trace!("Edit: {:?}", (ProgPoint::new(inst, pos), Edit::Move {
                from,
                to,
            }));
            target_edits.push((ProgPoint::new(inst, pos), Edit::Move {
                from,
                to,
            }));
        }
    }*/

    fn move_after_inst(&mut self, inst: Inst, vreg: VReg, to: Allocation) {
        self.add_move_later(inst, self.vreg_allocs[vreg.vreg()], to, vreg.class(), InstPosition::After, false);
    }

    fn move_before_inst(&mut self, inst: Inst, vreg: VReg, to: Allocation) {
        self.add_move_later(inst, self.vreg_allocs[vreg.vreg()], to, vreg.class(), InstPosition::Before, false);
    }

    fn allocd_within_constraint(&self, op: Operand) -> bool {
        let curr_alloc = self.vreg_allocs[op.vreg().vreg()];
        self.alloc_meets_op_constraint(curr_alloc, op.class(), op.constraint())
    }
    
    fn alloc_meets_op_constraint(&self, alloc: Allocation, class: RegClass, constraint: OperandConstraint) -> bool {
        match constraint {
            OperandConstraint::Any => alloc.is_some(),
            OperandConstraint::Reg => {
                alloc.is_reg() && alloc.as_reg().unwrap().class() == class
                    && !self.is_stack(alloc)
            },
            OperandConstraint::Stack => self.is_stack(alloc),
            OperandConstraint::FixedReg(preg) => alloc.is_reg() &&
                alloc.as_reg().unwrap() == preg,
            OperandConstraint::Reuse(_) => {
                unreachable!()
            }
        }
    }

    fn evict_vreg_in_preg(&mut self, inst: Inst, preg: PReg) {
        let evicted_vreg = self.vreg_in_preg[preg.index()];
        debug_assert_ne!(evicted_vreg, VReg::invalid());
        if self.vreg_spillslots[evicted_vreg.vreg()].is_invalid() {
            self.vreg_spillslots[evicted_vreg.vreg()] = self.allocstack(&evicted_vreg);
        }
        let slot = self.vreg_spillslots[evicted_vreg.vreg()];
        self.vreg_allocs[evicted_vreg.vreg()] = Allocation::stack(slot);
        trace!("Move reason: eviction");
        self.move_after_inst(inst, evicted_vreg, Allocation::reg(preg));
    }

    fn evict_any_reg(&mut self, inst: Inst, regclass: RegClass) -> PReg {
        let preg = self.lrus[regclass].pop();
        // TODO: Check if the preg has already been allocated for this
        // instruction. If it has, then there are too many stuff to
        // allocate, making allocation impossible.
        // Remember that for this to be true, the fixed registers must have
        // be allocated already. Why? Because if some register p0 has been allocated
        // and some fixed constraint register is encountered that needs p0, then
        // allocation will fail regardless of whether or not there are other free registers
        self.evict_vreg_in_preg(inst, preg);
        preg
    }

    fn freealloc(&mut self, vreg: VReg, add_to_freelist: bool) {
        trace!("Freeing vreg {:?} (add_to_freelist: {:?})", vreg, add_to_freelist);
        let alloc = self.vreg_allocs[vreg.vreg()];
        match alloc.kind() {
            AllocationKind::Reg => {
                let preg = alloc.as_reg().unwrap();
                self.vreg_in_preg[preg.index()] = VReg::invalid();
                // If it's a fixed stack slot, then it's not allocatable.
                if !self.is_stack(alloc) {
                    if add_to_freelist {
                        // Added to the freed def pregs list, not the free pregs
                        // list to avoid a def's allocated register being used
                        // as a scratch register.
                        self.freed_def_pregs[vreg.class()].insert(preg);
                    }
                }
            }
            AllocationKind::Stack => {
                // Do nothing.
                // I think it the allocation will be cheaper this way.
            }
            AllocationKind::None => panic!("Attempting to free an unallocated operand!")
        }
        self.vreg_allocs[vreg.vreg()] = Allocation::none();
        self.live_vregs.remove(&vreg);
    }
    
    /// Allocates a spill slot on the stack for `vreg`
    fn allocstack(&mut self, vreg: &VReg) -> SpillSlot {
        let size: u32 = self.func.spillslot_size(vreg.class()).try_into().unwrap();
        // Rest of this function was copied verbatim 
        // from `Env::allocate_spillslot` in src/ion/spill.rs.
        let mut offset = self.num_spillslots;
        // Align up to `size`.
        debug_assert!(size.is_power_of_two());
        offset = (offset + size - 1) & !(size - 1);
        let slot = if self.func.multi_spillslot_named_by_last_slot() {
            offset + size - 1
        } else {
            offset
        };
        offset += size;
        self.num_spillslots = offset;
        SpillSlot::new(slot as usize)
    }

    /// Allocates a physical register for the operand `op`.
    fn alloc_reg_for_operand(&mut self, inst: Inst, op: Operand) {
        trace!("freepregs: {:?}", self.freepregs[RegClass::Int]);
        trace!("freed_def_pregs: {:?}", self.freed_def_pregs[RegClass::Int]);
        trace!("");
        if let Some(freed_def_preg) = self.freed_def_pregs[op.class()].pop_last() {
            // Don't poke the LRU because the freed def register is no
            // longer in the LRU. It's not there so as to avoid getting it
            // used as a scratch register.
            trace!("Reusing the freed def preg: {:?}", freed_def_preg);
            self.vreg_allocs[op.vreg().vreg()] = Allocation::reg(freed_def_preg);
            self.vreg_in_preg[freed_def_preg.index()] = op.vreg();
        } else {
            let preg = if self.freepregs[op.class()].is_empty() {
                trace!("Evicting a register");
                self.evict_any_reg(inst, op.class())
            } else {
                trace!("Getting a register from freepregs");
                self.freepregs[op.class()].pop_last().unwrap()
            };
            trace!("The allocated register for vreg {:?}: {:?}", preg, op.vreg());
            self.lrus[op.class()].poke(preg);
            self.vreg_allocs[op.vreg().vreg()] = Allocation::reg(preg);
            self.vreg_in_preg[preg.index()] = op.vreg();
        }
    }

    fn alloc_fixed_reg_for_operand(&mut self, inst: Inst, op: Operand, preg: PReg) {
        trace!("The fixed preg: {:?}", preg);
        let mut preg_is_allocatable = false;
        if self.vreg_in_preg[preg.index()] != VReg::invalid() {
            // Something is already in that register. Evict it.
            self.evict_vreg_in_preg(inst, preg);
        } else if self.freed_def_pregs[preg.class()].contains(&preg) {
            // Consider the scenario:
            // def v0 (fixed: p0), use v1 (fixed: p0)
            // In the above, p0 has already been used for v0, and since it's a
            // def operand, the register has been freed and kept in `freed_def_pregs`,
            // so it can be added back to the free pregs list after the instruction 
            // has finished processing.
            // To avoid the preg being added back to the free list, it must be removed
            // from `freed_def_pregs` here.
            preg_is_allocatable = true;
            self.freed_def_pregs[preg.class()].remove(&preg);
        } else {
            // Find the register in the list of free registers (if it's there).
            // If it's not there, then it must be be a fixed stack slot.
            preg_is_allocatable = self.freepregs[op.class()].remove(&preg);
        }
        if preg_is_allocatable {
            self.lrus[op.class()].poke(preg);
        }
        self.vreg_allocs[op.vreg().vreg()] = Allocation::reg(preg);
        self.vreg_in_preg[preg.index()] = op.vreg();
    }

    /// Allocates for the operand `op` with index `op_idx` into the
    /// vector of instruction `inst`'s operands.
    /// Only non reuse-input operands.
    fn alloc_operand(&mut self, inst: Inst, op: Operand, op_idx: usize) {
        match op.constraint() {
            OperandConstraint::Any => {
                self.alloc_reg_for_operand(inst, op);
            }
            OperandConstraint::Reg => {
                self.alloc_reg_for_operand(inst, op);
            }
            OperandConstraint::Stack => {
                panic!("Stack only allocations aren't supported yet");
            }
            OperandConstraint::FixedReg(preg) => {
                self.alloc_fixed_reg_for_operand(inst, op, preg);
            }
            OperandConstraint::Reuse(_) => {
                // This is handled elsewhere
                unreachable!();
            }
        }
        self.allocs[(inst.index(), op_idx)] = self.vreg_allocs[op.vreg().vreg()];
    }

    /// Only processes non reuse-input operands
    fn process_operand_allocation(&mut self, inst: Inst, op: Operand, op_idx: usize) {
        debug_assert!(!matches!(op.constraint(), OperandConstraint::Reuse(_)));
        self.live_vregs.insert(op.vreg());
        if !self.allocd_within_constraint(op) {
            let prev_alloc = self.vreg_allocs[op.vreg().vreg()];
            self.alloc_operand(inst, op, op_idx);
            // Need to insert a move to propagate flow from the current
            // allocation to the subsequent places where the value was
            // used (in `prev_alloc`, that is).
            if prev_alloc.is_some() {
                trace!("Move reason: Prev allocation doesn't meet constraints");
                if op.kind() == OperandKind::Def {
                    // In the case where `op` is a def,
                    // the allocation of `op` will not be holding the value
                    // of `op` before the instruction. Since it's a def,
                    // it will only hold the value after. So, the move
                    // has to be done after.
                    //
                    // The move also has to be prepended. Consider the scenario:
                    // 
                    // 1. def v0 (any reg), use v1 (fixed: p0)
                    // 2. use v0 (fixed: p0)
                    //
                    // During the processing of the first instruction, v0 is already in
                    // p0. Since v1 has a fixed register constraint, it's processed
                    // first and evicts v0 from p0. Edits are inserted to flow v0 from
                    // its spillslot to p0 after the instruction:
                    //
                    // 1. def v0 (any reg), use v1 (fixed: p0)
                    // move from stack_v0 to p0
                    // 2. use v0 (fixed: p0)
                    // 
                    // When it's time to process v0, it has to be moved again: this time
                    // because it needs to be in a register, not on the stack.
                    // Edits are inserted to flow v0 from its spillslot to the newly allocated
                    // register, say p1.
                    //
                    // 1. def v0 (any reg), use v1 (fixed: p0)
                    // move from stack_v0 to p0
                    // move from p1 to stack_v0
                    // 2. use v0 (fixed: p0)
                    //
                    // The problem here is that the edits are out of order. p1, the
                    // allocation used for v0 in inst 1., is never moved into p0,
                    // the location v0 is expected to be in after inst 1.
                    // This messes up the dataflow.
                    // To avoid this, the moves are prepended.
                    //self.move_after_inst(inst, op.vreg(), prev_alloc);
                    self.add_move_later(
                        inst,
                        self.vreg_allocs[op.vreg().vreg()],
                        prev_alloc,
                        op.class(),
                        InstPosition::After,
                        true
                    );
                } else {
                    // In the case where `op` is a use, the defined value could
                    // have the same allocation as the `op` allocation. This
                    // is due to the fact that def operands are allocated and freed before
                    // use operands. Because of this, `op`'s allocation could be
                    // overwritten by the defined value's. And after the instruction,
                    // the defined value could be in `op`'s allocation, resulting in
                    // an incorrect value being moved into `prev_alloc`.
                    // Since, it's a use, the correct `op` value will already be in
                    // the `op` allocation before the instruction.
                    // Because of this, the move is done before, not after, `inst`.
                    //
                    // This was handled by a simple move from the operand to its previous
                    // allocation before the instruction, but this is incorrect.
                    // Consider the scenario:
                    // 1. use v0 (fixed: p0), use v1 (fixed: p1)
                    // 2. use v0 (fixed: p1)
                    // By the time inst 1 is to be processed, v0 will be in p1.
                    // But v1 should be in p1, not v0. If v0 is moved to p1 before inst 1,
                    // then it will overwrite v1 and v0 will be used instead of v1.
                    // It's also possible that the register used by v0 could be reused
                    // with a def operand.
                    // To resolve this, v0 is moved into its spillslot before inst 1.
                    // Then it's moved from its spillslot into p1 after inst 1, which is the place
                    // where it's expected to be after the instruction.
                    // This is to avoid two problems:
                    // 1. Overwriting a vreg that uses p1 in the current instruction.
                    // 2. Avoiding a situation where a def reuses the register used by v0
                    // and overwrites v0.
                    //
                    // It is possible for a virtual register to be used twice in the
                    // same instruction with different constraints.
                    // For example:
                    // 1. use v0 (fixed: stack0), use v0 (fixed: p0)
                    // 2. use v0 (fixed: p1)
                    // By the time inst 1 is to be processed, v0 will be in p1.
                    // But it should be in p0 and stack0. If stack0 is processed
                    // first, moves will be inserted to move from stack0 to v0's
                    // spillslot before inst 1 and to move from spillslot
                    // to p1 after the instruction:
                    //
                    // move from stack0 to stack_v0
                    // 1. use v0 (fixed: stack0), use v0 (fixed: p0)
                    // move from stack_v0 to p1
                    // 2. use v0 (fixed: p1)
                    //
                    // But when the second use is encountered, moves will be inserted again
                    // and mess up the dataflow:
                    //
                    // move from p0 to stack_v0
                    // move from stack0 to stack_v0
                    // 1. use v0 (fixed: stack0), use v0 (fixed: p0)
                    // move from stack_v0 to p1
                    // move from stack_v0 to p1
                    // 2. use v0 (fixed: p1)
                    //
                    // Assuming that after instruction 1 is processed, v0's 
                    // location is p0, then stack0 will always overwrite it,
                    // and v0 is not in stack0 (it's in p0, now).
                    // To avoid this scenario, these moves are only inserted
                    // for the first encountered constraint in an instruction.
                    // After this, any other operands with the same virtual register
                    // but different constraint will simply generate a move from the
                    // new location to the prev_alloc. This new move is inserted before
                    // the original one because the new location is now where v0 is
                    // expected to be before the instruction.
                    // For example:
                    //
                    // move from stack0 to stack_v0
                    // 1. use v0 (fixed: stack0), use v0 (fixed: p0)
                    // move from stack_v0 to p1
                    // 2. use v0 (fixed: p1)
                    //
                    // When the second use is encountered, the current location for v0 becomes
                    // p0 and a move from p0 to stack0 is prepended to the edits:
                    //
                    // move from p0 to stack0
                    // move from stack0 to stack_v0
                    // 1. use v0 (fixed: stack0), use v0 (fixed: p0)
                    // move from stack_v0 to p1
                    // 2. use v0 (fixed: p1)
                    
                    if !self.vregs_allocd_in_curr_inst.contains(&op.vreg()) {
                        if self.vreg_spillslots[op.vreg().vreg()].is_invalid() {
                            self.vreg_spillslots[op.vreg().vreg()] = self.allocstack(&op.vreg());
                        }
                        let op_spillslot = Allocation::stack(self.vreg_spillslots[op.vreg().vreg()]);
                        self.add_move_later(
                            inst,
                            self.vreg_allocs[op.vreg().vreg()],
                            op_spillslot,
                            op.class(),
                            InstPosition::Before,
                            false,
                        );
                        self.add_move_later(
                            inst,
                            op_spillslot,
                            prev_alloc,
                            op.class(),
                            InstPosition::After,
                            false,
                        );
                    } else {
                        self.add_move_later(
                            inst,
                            self.vreg_allocs[op.vreg().vreg()],
                            prev_alloc,
                            op.class(),
                            InstPosition::Before,
                            true,
                        );
                    }
                }
                if prev_alloc.is_reg() {
                    // Free the previous allocation so that it can be
                    // reused.
                    let preg = prev_alloc.as_reg().unwrap();
                    self.vreg_in_preg[preg.index()] = VReg::invalid();
                    // If it's a fixed stack slot, then it's not allocatable.
                    if !self.is_stack(prev_alloc) {
                        trace!("{:?} is no longer using preg {:?}, so freeing it after instruction", op.vreg(), preg);
                        self.free_after_curr_inst.insert(preg);
                        self.lrus[preg.class()].remove(preg.hw_enc());
                    }
                }
            } else if op.kind() == OperandKind::Use {
                trace!("{:?}'s first use", op.vreg());
                self.first_use.insert(op.vreg());
            }
            trace!("Allocation for instruction {:?} and operand {:?}: {:?}", inst, op, self.allocs[(inst.index(), op_idx)]);
        } else {
            self.allocs[(inst.index(), op_idx)] = self.vreg_allocs[op.vreg().vreg()];
            trace!("Allocation for instruction {:?} and operand {:?}: {:?}", inst, op, self.allocs[(inst.index(), op_idx)]);
        }
        let new_alloc = self.vreg_allocs[op.vreg().vreg()];
        if new_alloc.is_reg() {
            // If the new allocation was once a freed prev_alloc, remove it
            // from the free after current inst list.
            // For example:
            //
            // 1. use v0 (fixed: p0), use v0 (fixed: p1)
            // 2. use v0 (fixed: p1)
            //
            // In the processing of the above, v0 is allocated to p1 at inst 2.
            // During the processing of inst 1, v0's allocation is changed to p0
            // and p1 is put on the free after current inst list to make it
            // available for later allocation.
            // But then, it's reallocated for the second operand.
            // To prevent reallocating a register while a live one is still in it,
            // this register has to be removed from the list.
            let preg = new_alloc.as_reg().unwrap();
            if self.free_after_curr_inst.contains(&preg) {
                trace!("{:?} is now using preg {:?}. Removing it from the free after instruction list", op.vreg(), preg);
                self.free_after_curr_inst.remove(&preg);
            }
            // The LRU doesn't need to be modified here because it has already
            // been handled during the new allocation.
        }
        if op.kind() == OperandKind::Use {
            // Need to remember that this allocation is used in this instruction
            // by a use operand, to make decisions on whether to free a reuse operand's
            // allocation during the processing of reuse operands.
            self.allocs_used_by_use_ops.insert(new_alloc);
        }
        self.vregs_allocd_in_curr_inst.insert(op.vreg());
    }

    fn process_reuse_operand_allocation(
        &mut self,
        inst: Inst,
        op: Operand,
        op_idx: usize,
        reused_op: Operand,
        reused_idx: usize
    ) {
        debug_assert!(matches!(op.constraint(), OperandConstraint::Reuse(_)));
        // TODO: Check if reuse operand is not a def and if it's not, return an error.

        // We first need to check if the reuse operand has already been allocated,
        // in a previous alloc_inst call. There are 2 cases that need to be considered here:
        //
        // Case 1: The reuse operand has already been allocated.
        // An example:
        // inst 1: reuse def v0 (1), use v1
        // inst 2: use v0
        // In the above example, v0 will have already been allocated by the time inst 1
        // is about to be processed.
        // After this inst 1, v0 is expected to be in some location l0.
        // But because of the constraints, it should be in v1's location.
        // To account for this, the reused input (v1) is moved into its spillslot before the instruction
        // and its allocation is used for both the reuse operand (v0) and the reused input
        // (the reused input's allocation is used for both of them, just in case the
        // reused input has a fixed register constraint).
        // After the instruction, v0 is first moved from v1's allocation to l0, the location it's expected to be
        // after the instruction and v1 is moved from its spillslot into its current allocation.
        //
        // Case 2: The reuse operand has not yet been allocated.
        // This could happen in a scenario such as:
        // inst 1: reuse def v0 (1), use v1
        // inst 2: use v1
        // Since v1 and v0 have to be the same allocation, then one of the following could be done:
        // 1. A new location is allocated for v0, v1 is moved into that new location before the
        // instruction, and the allocs for both v0 and v1 are set to that location (Not good if
        // v1 has a fixed register constraint).
        // 2. v1 is moved into its spillslot before the instruction, used as the allocation for
        // v0, then v1 is moved from its spillslot into its allocation after the instruction.
        //
        // No 1. is better with respect to moves: only 1 move is generated rather than 2.
        // No 2. is better with respect to allocations: no extra allocation is required. Especially
        // considering the fact that, since reuse operands are always defs, the allocation will be
        // deallocated immediately.
        // No 1. may lead to better runtime performance, because less stack movements are required
        // (assuming no eviction takes place) while no 2. may lead to better compile time performance
        // because less bookkeeping has to be done to effect it.
        // We're going with no 2. here.
        //
        // There is also a problem that could arise when the reused input is the first encountered
        // use of a vreg.
        // Consider a scenario:
        // 
        // 1. def v12 (reuse: 1), use v7 (fixed: p31)
        // 2. def v13, use v12 (fixed: p31)
        // v12 is in p31 afterwards
        //
        // Inst 2 is processed first and after its processing
        // v12 is in p31, right before inst 2.
        // During the processing of inst 1, because of the way reuse
        // operands are handled, v7's allocation is first saved before inst 1,
        // then it is restored afterwards. The resulting modifications are:
        //
        // move from p31 to stack_v7  // v7 is in p31 from this point upwards
        // 1. def v12 (reuse: 1), use v7 (fixed: p31)  // Both are allocated to p31
        // move from p31 to p31       // to flow v12 to the location it's expected to be afterwards
        // move from stack_v7 to p31  // to restore v7
        // 2. def v13, use v12 (fixed: p31)
        //
        // The problem with the above is that the reuse operand handling assumed that vregs
        // used in an instruction will be live after, but it isn't in this case. v12 uses p31
        // after inst 1 because it is supposed to be free. Since v7's first use is in inst 1,
        // it should not be moved into its allocation afterwards.
        // Hence, moves to flow the reused input into its allocation after the instruction
        // are inserted only if the input lives past the instruction, that is, its first use
        // is not in this instruction.

        // TODO: Ensure that the reused operand is a use.
        // TODO: Ensure that the reuse operand and its reused input have the
        // same register class.
        trace!("Move Reason: Reuse constraints");

        let reused_op_first_use = self.first_use.contains(&reused_op.vreg());
        if self.vreg_allocs[op.vreg().vreg()].is_some() {
            let op_prev_alloc = self.vreg_allocs[op.vreg().vreg()];
            let reused_op_vreg = reused_op.vreg();
            if self.vreg_spillslots[reused_op_vreg.vreg()].is_invalid() {
                self.vreg_spillslots[reused_op_vreg.vreg()] = self.allocstack(&reused_op_vreg);
            }
            let reused_op_spillslot = self.vreg_spillslots[reused_op.vreg().vreg()];
            
            // If this is the reused operand's first use, then don't
            // restore it afterwards, because it doesn't live past this instruction.
            if !reused_op_first_use {
                // Move the reused input into its spillslot.
                self.add_move_later(
                    inst,
                    self.vreg_allocs[reused_op_vreg.vreg()],
                    Allocation::stack(reused_op_spillslot),
                    op.class(),
                    InstPosition::Before,
                    false,
                );
            }

            // Move the reuse operand from the reused input's allocation into the location it's
            // expected to be in after the current instruction.
            self.add_move_later(
                inst,
                self.vreg_allocs[reused_op_vreg.vreg()],
                op_prev_alloc,
                op.class(),
                InstPosition::After,
                false,
            );

            // If this is the reused operand's first use, then don't
            // restore it afterwards, because it doesn't live past this instruction.
            if !reused_op_first_use {
                // Move the reused input from its spillslot into its current allocation
                self.add_move_later(
                    inst,
                    Allocation::stack(reused_op_spillslot),
                    self.vreg_allocs[reused_op_vreg.vreg()],
                    op.class(),
                    InstPosition::After,
                    false,
                );
            }

            self.allocs[(inst.index(), op_idx)] = self.vreg_allocs[reused_op_vreg.vreg()];
            self.allocs[(inst.index(), reused_idx)] = self.vreg_allocs[reused_op_vreg.vreg()];

            // Deallocate the reuse operand.
            // We can't just deallocate the reuse operand.
            // The reason for this is that, since reuse operands are defs
            // it is possible for its allocation to be reused by a use operand.
            // If it is freed here, then the allocation could be reallocated to another
            // vreg while the use it was allocated to is still live.
            // For example:
            //
            // 1. def v0
            // 2. def v1, use v2
            // 3. def v3 (reuse: 1), use v0
            //
            // If v0 is allocated to p0, then v3 will also be allocated to p0.
            // Since reuse operands are processed last, then if v3 is just freed normally,
            // then p0 will be free for allocation to v1 and v2, overwriting whatever
            // value was defd in v0 in inst 1.
            // To avoid this allocation of a place that has already been allocated to a live vreg,
            // the `add_to_freelist` parameter is set to true
            // only if the reuse operand's allocation was not reused by any use operands
            // in the instruction.
            let op_alloc_is_in_use = self.allocs_used_by_use_ops.contains(&op_prev_alloc);
            self.freealloc(op.vreg(), !op_alloc_is_in_use);
            trace!("Allocation for instruction {:?} and operand {:?}: {:?}", inst, op, self.allocs[(inst.index(), op_idx)]);
        } else {
            let reused_op_vreg = reused_op.vreg();
            // If this is the reused operand's first use, then don't
            // restore it afterwards, because it doesn't live past this instruction.
            if !reused_op_first_use {
                if self.vreg_spillslots[reused_op_vreg.vreg()].is_invalid() {
                    self.vreg_spillslots[reused_op_vreg.vreg()] = self.allocstack(&reused_op_vreg);
                }
                let reused_op_spillslot = self.vreg_spillslots[reused_op.vreg().vreg()];
                // Move the reused input into its spillslot before the instruction.
                self.add_move_later(
                    inst,
                    self.vreg_allocs[reused_op_vreg.vreg()],
                    Allocation::stack(reused_op_spillslot),
                    op.class(),
                    InstPosition::Before,
                    false,
                );
                // Move back into its allocation.
                self.add_move_later(
                    inst,
                    Allocation::stack(reused_op_spillslot),
                    self.vreg_allocs[reused_op_vreg.vreg()],
                    op.class(),
                    InstPosition::After,
                    false,
                );
            }
            self.allocs[(inst.index(), op_idx)] = self.vreg_allocs[reused_op_vreg.vreg()];
            trace!("Allocation for instruction {:?} and operand {:?}: {:?}", inst, op, self.allocs[(inst.index(), op_idx)]);
        }
    }

    fn alloc_slots_for_block_params(&mut self, succ: Block) {
        for vreg in self.func.block_params(succ) {
            if self.vreg_spillslots[vreg.vreg()].is_invalid() {
                self.vreg_spillslots[vreg.vreg()] = self.allocstack(vreg);
                trace!("Block param {:?} is in {:?}", vreg, Allocation::stack(self.vreg_spillslots[vreg.vreg()]));
            }
        }
    }

    /// If instruction `inst` is a branch in `block`,
    /// this function places branch arguments in the spillslots
    /// expected by the destination blocks.
    /// 
    /// The process used to do this is as follows:
    /// 
    /// 1. Move all branch arguments into corresponding temporary spillslots.
    /// 2. Move values from the temporary spillslots to corresponding block param spillslots.
    /// 3. Move values from the temporary spillslots to post-block locatioks, if any, for
    /// non-block-param arguments.
    /// 
    /// These temporaries are used because the moves have to be parallel in the case where
    /// a block parameter of the successor block is a branch argument.
    fn process_branch(&mut self, block: Block, inst: Inst) {
        
        // Used to know which temporary spillslot should be used next.
        let mut next_temp_idx = PartedByRegClass { items: [0, 0, 0] };

        fn reset_temp_idx(next_temp_idx: &mut PartedByRegClass<usize>) {
            next_temp_idx[RegClass::Int] = 0;
            next_temp_idx[RegClass::Float] = 0;
            next_temp_idx[RegClass::Vector] = 0;
        }

        // In the case where the block param of a successor is also a branch arg,
        // the reading of all the block params must be done before the writing.
        // This is necessary to prevent overwriting the branch arg's value before
        // placing it in the corresponding branch param spillslot.
        
        // NO LONGER BEING INSERTED IN REVERSE.
        // And because edits are inserted in reverse, the algorithm has to process
        // the branch args which are not branch params first. This will result in the
        // output code processing branch args which are params before the others.

        for succ in self.func.block_succs(block).iter() {
            self.alloc_slots_for_block_params(*succ);
        }

        for (succ_idx, _) in self.func.block_succs(block).iter().enumerate() {
            // Move from branch args spillslots to temporaries.
            //
            // Consider a scenario: block X branches to block Y and block Y branches to block X.
            // Block Y has block param vp and block X uses virtual register va as the branch arg for vp.
            // Block X has an instruction that uses vp.
            // In the case where branch arg va is defined in a predecessor, there is a possibility
            // that, at the beginning of the block, during the reload, that va will always overwrite vp.
            // This could happen because at the end of the block, va is allocated to be in vp's
            // spillslot. If va isn't used throughout the block (or if all its use constraints allow it to be
            // in vp's spillslot), then during reload, it will still be allocated to vp's spillslot.
            // This will mean that at the beginning of the block, both va and vp will be expected to be
            // in vp's spillslot. An edit will be inserted to move from va's spillslot to vp's.
            // And depending on the constraints of vp's use, an edit may or may not be inserted to move 
            // from vp's spillslot to somewhere else.
            // Either way, the correctness of the dataflow will depend on the order of edits.
            // If vp is required in be on the stack, then no edit will be inserted for it (it's already on
            // the stack, in its spillslot). But an edit will be inserted to move from va's spillslot
            // to vp's.
            // If block Y has other predecessors that define vp to be other values, then this dataflow
            // is clearly wrong.
            //
            // To avoid this scenario, branch args are placed into their own spillslots here
            // so that if they aren't moved at all throughout the block, they will not be expected to 
            // be in another vreg's spillslot at the block beginning.
            for vreg in self.func.branch_blockparams(block, inst, succ_idx).iter() {
                self.live_vregs.insert(*vreg);
                if self.vreg_spillslots[vreg.vreg()].is_invalid() {
                    self.vreg_spillslots[vreg.vreg()] = self.allocstack(vreg);
                    trace!("Block arg {:?} is going to be in {:?}", vreg, Allocation::stack(self.vreg_spillslots[vreg.vreg()]));
                }
                if self.temp_spillslots[vreg.class()].len() == next_temp_idx[vreg.class()] {
                    let newslot = self.allocstack(vreg);
                    self.temp_spillslots[vreg.class()].push(newslot);
                }
                let temp_slot = self.temp_spillslots[vreg.class()][next_temp_idx[vreg.class()]];
                let temp = Allocation::stack(temp_slot);
                next_temp_idx[vreg.class()] += 1;
                let vreg_spill = Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
                trace!("{:?} which is going to be in {:?} inserting move to {:?}", vreg, vreg_spill, temp);
                self.add_move_later(inst, vreg_spill, temp, vreg.class(), InstPosition::Before, false);
            }
        }
    
        reset_temp_idx(&mut next_temp_idx);

        for (succ_idx, succ) in self.func.block_succs(block).iter().enumerate() {
            let succ_params = self.func.block_params(*succ);

            // Move from temporaries to block param spillslots.
            for (pos, vreg) in self.func.branch_blockparams(block, inst, succ_idx).iter().enumerate() {
                let succ_param_vreg = succ_params[pos];
                let param_alloc = Allocation::stack(self.vreg_spillslots[succ_param_vreg.vreg()]);
                let temp_slot = self.temp_spillslots[vreg.class()][next_temp_idx[vreg.class()]];
                let temp = Allocation::stack(temp_slot);
                self.vreg_allocs[vreg.vreg()] = temp;
                next_temp_idx[vreg.class()] += 1;
                trace!(" --- Placing branch arg {:?} in {:?}", vreg, temp);
                trace!("{:?} which is now in {:?} inserting move to {:?}", vreg, temp, param_alloc);
                self.add_move_later(inst, temp, param_alloc, vreg.class(), InstPosition::Before, false);
            }
        }

        reset_temp_idx(&mut next_temp_idx);

        /*for (succ_idx, succ) in self.func.block_succs(block).iter().enumerate() {
            let succ_params = self.func.block_params(*succ);

            // Move from temporaries to post block locations.
            for vreg in self.func.branch_blockparams(block, inst, succ_idx).iter() {
                let temp_slot = self.temp_spillslots[vreg.class()][next_temp_idx[vreg.class()]];
                let temp = Allocation::stack(temp_slot);
                next_temp_idx[vreg.class()] += 1;
                if succ_params.contains(vreg) {
                    // Skip to avoid overwriting the new value for the block param,
                    // which will be moved into its spillslot from its temporary.
                    continue;
                }
                let prev_alloc = self.vreg_allocs[vreg.vreg()];
                if prev_alloc.is_some() {
                    trace!("{:?} which is going to be in {:?} inserting move to {:?}", vreg, temp, prev_alloc);
                    self.add_move_later(inst, temp, prev_alloc, vreg.class(), InstPosition::Before);
                } else {
                    trace!("{:?} prev alloc is none, so no moving here", vreg);
                }
            }
        }*/

        for (succ_idx, _) in self.func.block_succs(block).iter().enumerate() {
            for vreg in self.func.branch_blockparams(block, inst, succ_idx).iter() {
                // All branch arguments should be in their spillslots at the end of the function.
                self.vreg_allocs[vreg.vreg()] = Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
            }
        }
    }

    fn alloc_inst(&mut self, block: Block, inst: Inst) {
        if self.func.is_branch(inst) {
            self.process_branch(block, inst);
        }
        let operands = self.func.inst_operands(inst);
        for (op_idx, op) in FixedLateOperands::new(operands) {
            self.process_operand_allocation(inst, op, op_idx);
        }
        for (op_idx, op) in NonFixedNonReuseLateOperands::new(operands) {
            self.process_operand_allocation(inst, op, op_idx);
        }
        for (_, op) in NonReuseLateDefOperands::new(operands) {
            self.freealloc(op.vreg(), true);
        }
        for (op_idx, op) in FixedEarlyOperands::new(operands) {
            self.process_operand_allocation(inst, op, op_idx);
        }
        for (op_idx, op) in NonFixedNonReuseEarlyOperands::new(operands) {
            self.process_operand_allocation(inst, op, op_idx);
        }
        for (_, op) in NonReuseEarlyDefOperands::new(operands) {
            self.freealloc(op.vreg(), true);
        }
        for (op_idx, op) in ReuseOperands::new(operands) {
            let OperandConstraint::Reuse(reused_idx) = op.constraint() else {
                unreachable!()
            };
            self.process_reuse_operand_allocation(inst, op, op_idx, operands[reused_idx], reused_idx);
        }
        for i in (0..self.inst_post_edits.len()).rev() {
            let (point, edit, class) = self.inst_post_edits[i].clone();
            self.process_edit(point, edit, class);
        }
        for i in (0..self.inst_pre_edits.len()).rev() {
            let (point, edit, class) = self.inst_pre_edits[i].clone();
            self.process_edit(point, edit, class);
        }
        self.inst_post_edits.clear();
        self.inst_pre_edits.clear();
        for preg in self.free_after_curr_inst.iter().cloned() {
            self.freepregs[preg.class()].insert(preg);
            self.lrus[preg.class()].append(preg.hw_enc());
        }
        for class in [RegClass::Int, RegClass::Float, RegClass::Vector] {
            // Add the remaining freed def pregs back to the free list.
            for preg in self.freed_def_pregs[class].iter().cloned() {
                self.freepregs[class].insert(preg);
            }
            self.freed_def_pregs[class].clear();
        }
        self.free_after_curr_inst.clear();
        self.vregs_allocd_in_curr_inst.clear();
        self.first_use.clear();
        self.allocs_used_by_use_ops.clear();
    }

    /// At the beginning of every block, all virtual registers that are
    /// livein are expected to be in their respective spillslots.
    /// This function sets the current allocations of livein registers
    /// to their spillslots and inserts the edits to flow livein values to
    /// the allocations where they are expected to be before the first
    /// instruction. 
    fn reload_at_begin(&mut self, block: Block) {
        // We need to check for the registers that are still live.
        // These registers are livein and they should be stack-allocated.
        let live_vregs = self.live_vregs.clone();
        for vreg in live_vregs.iter().cloned() {
            if self.vreg_spillslots[vreg.vreg()].is_invalid() {
                self.vreg_spillslots[vreg.vreg()] = self.allocstack(&vreg);
            }
            // The allocation where the vreg is expected to be before
            // the first instruction.
            let prev_alloc = self.vreg_allocs[vreg.vreg()];
            if prev_alloc.is_reg() {
                self.freealloc(vreg, true);
            }
            let slot = Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
            self.vreg_allocs[vreg.vreg()] = slot;
            if slot == prev_alloc {
                // No need to do any movements if the spillslot is where the vreg is expected to be.
                trace!("No need to reload {:?} because it's already in its expected allocation", vreg);
                continue;
            }
            trace!("Move reason: reload {:?} at begin - move from its spillslot", vreg);
            self.add_move_later(
                self.func.block_insns(block).first(),
                slot,
                prev_alloc,
                vreg.class(),
                InstPosition::Before,
                true
            );
            /*self.move_before_inst(
                self.func.block_insns(block).first(),
                vreg,
                prev_alloc,
            );*/
        }
        for i in (0..self.inst_post_edits.len()).rev() {
            let (point, edit, class) = self.inst_post_edits[i].clone();
            self.process_edit(point, edit, class);
        }
        for i in (0..self.inst_pre_edits.len()).rev() {
            let (point, edit, class) = self.inst_pre_edits[i].clone();
            self.process_edit(point, edit, class);
        }
        self.inst_post_edits.clear();
        self.inst_pre_edits.clear();
        for class in [RegClass::Int, RegClass::Float, RegClass::Vector] {
            for preg in self.freed_def_pregs[class].iter().cloned() {
                self.freepregs[class].insert(preg);
            }
            self.freed_def_pregs[class].clear();
        }
        for preg in self.free_after_curr_inst.iter().cloned() {
            self.freepregs[preg.class()].insert(preg);
            self.lrus[preg.class()].append(preg.hw_enc());
        }
        self.free_after_curr_inst.clear();
    }

    fn alloc_block(&mut self, block: Block) {
        trace!("{:?} start", block);
        for inst in self.func.block_insns(block).iter().rev() {
            self.alloc_inst(block, inst);
        }
        self.reload_at_begin(block);
        self.live_vregs.clear();
        trace!("{:?} end\n", block);
    }

    fn run(&mut self) -> Result<(), RegAllocError> {
        debug_assert_eq!(self.func.entry_block().index(), 0);
        for block in (0..self.func.num_blocks()).rev() {
            self.alloc_block(Block::new(block));
        }
        //self.edits.reverse();

        /////////////////////////////////////////////////////////////////////////////////////
        trace!("Done!");
        struct Z(usize);
        impl std::fmt::Debug for Z {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "v{}", self.0)
            }
        }
        let mut v = Vec::new();
        for i in 0..self.func.num_vregs() {
            if self.vreg_spillslots[i].is_valid() {
                v.push((Z(i), Allocation::stack(self.vreg_spillslots[i])));
            }
        }
        trace!("{:?}", v);
        trace!("\nTemp spillslots: {:?}", self.temp_spillslots);
        /////////////////////////////////////////////////////////////////////////////////////

        Ok(())
    }
}

pub fn run<F: Function>(
    func: &F,
    mach_env: &MachineEnv,
    enable_annotations: bool,
    enable_ssa_checker: bool,
) -> Result<Output, RegAllocError> {
    let cfginfo = CFGInfo::new(func)?;

    if enable_ssa_checker {
        validate_ssa(func, &cfginfo)?;
    }

    let mut env = Env::new(func, mach_env);
    env.run()?;

trace!("Final edits: {:?}", env.edits);
    Ok(Output {
        edits: env.edits.make_contiguous().to_vec(),
        allocs: env.allocs.allocs,
        inst_alloc_offsets: env.allocs.inst_alloc_offsets,
        num_spillslots: env.num_spillslots as usize,
        debug_locations: Vec::new(),
        safepoint_slots: Vec::new(),
        stats: env.stats,
    })
}
