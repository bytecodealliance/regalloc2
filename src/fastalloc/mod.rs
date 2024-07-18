use core::convert::TryInto;
use core::ops::{Index, IndexMut};
use crate::{Block, Inst, OperandKind, Operand, PReg, RegClass, VReg, SpillSlot, AllocationKind, OperandConstraint, InstPosition};
use crate::{Function, MachineEnv, ssa::validate_ssa, ProgPoint, Edit, Output};
use crate::{cfg::CFGInfo, RegAllocError, Allocation, ion::Stats};
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
    /// Free physical registers for classes Int, Float, and Vector, respectively.
    freepregs: PartedByRegClass<Vec<PReg>>,
    /// Least-recently-used caches for register classes Int, Float, and Vector, respectively.
    lrus: Lrus,
    /// `vreg_in_preg[class][i]` is the virtual register currently in physical register `i`
    /// in register class `class`.
    vreg_in_preg: PartedByRegClass<Vec<VReg>>,
    /// For parallel moves from branch args to block param spillslots.
    temp_spillslots: PartedByRegClass<Vec<SpillSlot>>,
    /// The edits to be inserted before the currently processed instruction.
    pre_edits: Vec<(ProgPoint, Edit)>,

    // Output.
    allocs: Allocs,
    edits: Vec<(ProgPoint, Edit)>,
    num_spillslots: u32,
    stats: Stats,
}

impl<'a, F: Function> Env<'a, F> {
    fn new(func: &'a F, env: &'a MachineEnv) -> Self {
        trace!("multispillslots_named_by_last_slot: {:?}", func.multi_spillslot_named_by_last_slot());
        let regs = [
            env.preferred_regs_by_class[RegClass::Int as usize].clone(),
            env.preferred_regs_by_class[RegClass::Float as usize].clone(),
            env.preferred_regs_by_class[RegClass::Vector as usize].clone(),
        ];
        use alloc::vec;
        Self {
            func,
            vreg_allocs: vec![Allocation::none(); func.num_vregs()],
            vreg_spillslots: vec![SpillSlot::invalid(); func.num_vregs()],
            live_vregs: HashSet::with_capacity(func.num_vregs()),
            freepregs: PartedByRegClass { items: regs.clone() },
            lrus: Lrus::new(
                regs[0].len(),
                regs[1].len(),
                regs[2].len()
            ),
            vreg_in_preg: PartedByRegClass { items: [
                vec![VReg::invalid(); regs[0].len()],
                vec![VReg::invalid(); regs[1].len()],
                vec![VReg::invalid(); regs[2].len()],
            ] },
            temp_spillslots: PartedByRegClass { items: [
                Vec::with_capacity(func.num_vregs()),
                Vec::with_capacity(func.num_vregs()),
                Vec::with_capacity(func.num_vregs()),
            ] },
            pre_edits: Vec::new(),
            allocs: Allocs::new(func, env),
            edits: Vec::new(),
            num_spillslots: 0,
            stats: Stats::default(),
        }
    }

    fn add_move(&mut self, inst: Inst, from: Allocation, to: Allocation, class: RegClass, pos: InstPosition) {
        if from.is_stack() && to.is_stack() {
            let mut evicted = false;
            let scratch_reg = if self.freepregs[class].is_empty() {
                evicted = true;
                self.evictreg(inst, class)
            } else {
                *self.freepregs[class].last().unwrap()
            };
            if evicted {
                self.freepregs[class].push(scratch_reg);
            }
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
    }

    fn move_after_inst(&mut self, inst: Inst, vreg: VReg, to: Allocation) {
        self.add_move(inst, self.vreg_allocs[vreg.vreg()], to, vreg.class(), InstPosition::After);
    }

    fn move_before_inst(&mut self, inst: Inst, vreg: VReg, to: Allocation) {
        self.add_move(inst, self.vreg_allocs[vreg.vreg()], to, vreg.class(), InstPosition::Before);
    }

    fn allocd_within_constraint(&self, op: Operand) -> bool {
        let curr_alloc = self.vreg_allocs[op.vreg().vreg()];
        self.alloc_meets_op_constraint(curr_alloc, op.class(), op.constraint())
    }
    
    fn alloc_meets_op_constraint(&self, alloc: Allocation, class: RegClass, constraint: OperandConstraint) -> bool {
        match constraint {
            OperandConstraint::Any => alloc.is_some(),
            OperandConstraint::Reg => alloc.is_reg() && alloc.as_reg().unwrap().class() == class,
            OperandConstraint::Stack => alloc.is_stack(),
            OperandConstraint::FixedReg(preg) => alloc.is_reg() &&
                alloc.as_reg().unwrap() == preg,
            OperandConstraint::Reuse(_) => {
                unreachable!()
            }
        }
    }

    fn evictreg(&mut self, inst: Inst, regclass: RegClass) -> PReg {
        let preg = self.lrus[regclass].pop();
        // TODO: Check if the preg has already been allocated for this
        // instruction. If it has, then there are too many stuff to
        // allocate, making allocation impossible.
        // Remember that for this to be true, the fixed registers must have
        // be allocated already. Why? Because if some register p0 has been allocated
        // and some fixed constraint register is encountered that needs p0, then
        // allocation will fail regardless of whether or not there are other free registers
        let evicted_vreg = self.vreg_in_preg[regclass][preg.hw_enc()];
        let slot = self.allocstack(&evicted_vreg);
        self.vreg_allocs[evicted_vreg.vreg()] = Allocation::stack(slot);
        trace!("Move reason: eviction");
        self.move_after_inst(inst, evicted_vreg, Allocation::reg(preg));
        preg
    }

    fn freealloc(&mut self, vreg: VReg) {
        let alloc = self.vreg_allocs[vreg.vreg()];
        match alloc.kind() {
            AllocationKind::Reg => {
                let preg = alloc.as_reg().unwrap();
                self.freepregs[vreg.class()].push(preg);
                self.vreg_in_preg[vreg.class()][preg.hw_enc()] = VReg::invalid();
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
        let preg = if self.freepregs[op.class()].is_empty() {
            self.evictreg(inst, op.class())
        } else {
            self.freepregs[op.class()].pop().unwrap()
        };
        self.lrus[op.class()].poke(preg);
        self.vreg_allocs[op.vreg().vreg()] = Allocation::reg(preg);
        self.vreg_in_preg[op.class()][preg.hw_enc()] = op.vreg();
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
                panic!("Fixed reg allocations aren't supported yet");
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
                    self.move_after_inst(inst, op.vreg(), prev_alloc);
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
                    self.move_before_inst(inst, op.vreg(), prev_alloc);
                }
            }
            trace!("Allocation for instruction {:?} and operand {:?}: {:?}", inst, op, self.allocs[(inst.index(), op_idx)]);
        } else {
            self.allocs[(inst.index(), op_idx)] = self.vreg_allocs[op.vreg().vreg()];
            trace!("Allocation for instruction {:?} and operand {:?}: {:?}", inst, op, self.allocs[(inst.index(), op_idx)]);
        }
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
        // To account for this, a move is inserted to move from the reused input's allocation
        // to the reuse operand's allocation before the instruction.
        // The allocations for both the reused input and the reuse operand are then both set
        // to l0, for this current instruction.
        //
        // Case 2: The reuse operand has not yet been allocated.
        // This could happen in a scenario such as:
        // inst 1: reuse def v0 (1), use v1
        // inst 2: use v1
        // Since v1 and v0 have to be the same allocation, then one of the following could be done:
        // 1. A new location is allocated for v0, v1 is moved into that new location before the
        // instruction, and the allocs for both v0 and v1 are set to that location.
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

        // TODO: Ensure that the reused operand is a use.
        // TODO: Ensure that the reuse operand and its reused input have the
        // same register class.
        let reused_alloc = self.allocs[(inst.index(), reused_idx)];
        if self.vreg_allocs[op.vreg().vreg()].is_some() {
            if !self.alloc_meets_op_constraint(
                self.vreg_allocs[op.vreg().vreg()],
                reused_op.class(),
                reused_op.constraint()
            ) {
                // After this instruction, op's location is expected to be `prev_alloc`
                let prev_alloc = self.vreg_allocs[op.vreg().vreg()];
                self.alloc_reg_for_operand(inst, op);
                self.add_move(
                    inst,
                    self.vreg_allocs[op.vreg().vreg()],
                    prev_alloc,
                    op.class(),
                    InstPosition::After,
                );
            }
            self.add_move(
                inst,
                reused_alloc,
                self.vreg_allocs[op.vreg().vreg()],
                op.class(),
                InstPosition::Before,
            );
            self.allocs[(inst.index(), op_idx)] = self.vreg_allocs[op.vreg().vreg()];
            self.allocs[(inst.index(), reused_idx)] = self.vreg_allocs[op.vreg().vreg()];
            // Deallocate the reuse operand.
            self.vreg_allocs[op.vreg().vreg()] = Allocation::none();
            self.live_vregs.remove(&op.vreg());
            trace!("Allocation for instruction {:?} and operand {:?}: {:?}", inst, op, self.allocs[(inst.index(), op_idx)]);
            trace!("Allocation for instruction {:?} and operand {:?} is now: {:?}", inst, reused_op, self.allocs[(inst.index(), op_idx)]);
        } else {
            let reused_op_vreg = reused_op.vreg();
            if self.vreg_spillslots[reused_op_vreg.vreg()].is_invalid() {
                self.vreg_spillslots[reused_op_vreg.vreg()] = self.allocstack(&reused_op_vreg);
            }
            let reused_op_spillslot = self.vreg_spillslots[reused_op.vreg().vreg()];
            // Remember: Edits are inserted in reverse.
            // Move back into its allocation.
            self.add_move(
                inst,
                Allocation::stack(reused_op_spillslot),
                self.vreg_allocs[reused_op_vreg.vreg()],
                op.class(),
                InstPosition::After
            );
            // Move the reused input into its spillslot before the instruction.
            self.add_move(
                inst,
                self.vreg_allocs[reused_op_vreg.vreg()],
                Allocation::stack(reused_op_spillslot),
                op.class(),
                InstPosition::Before,
            );
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

    fn place_branch_args_in_stack_allocs(&mut self, block: Block, inst: Inst, succ: Block, succ_idx: usize) {
        let succ_params = self.func.block_params(succ);

        // Used to know which temporary spillslot should be used next.
        let mut next_temp_idx = PartedByRegClass { items: [0, 0, 0] };

        fn reset_temp_idx(next_temp_idx: &mut PartedByRegClass<usize>) {
            next_temp_idx[RegClass::Int] = 0;
            next_temp_idx[RegClass::Float] = 0;
            next_temp_idx[RegClass::Vector] = 0;
        }

        // Move from temporaries to post block locations.
        for vreg in self.func.branch_blockparams(block, inst, succ_idx).iter() {
            self.live_vregs.insert(*vreg);
            if self.temp_spillslots[vreg.class()].len() == next_temp_idx[vreg.class()] {
                let newslot = self.allocstack(vreg);
                self.temp_spillslots[vreg.class()].push(newslot);
            }
            let temp_slot = self.temp_spillslots[vreg.class()][next_temp_idx[vreg.class()]];
            let temp = Allocation::stack(temp_slot);
            next_temp_idx[vreg.class()] += 1;
            let prev_alloc = self.vreg_allocs[vreg.vreg()];
            if prev_alloc.is_some() {
                trace!("{:?} which is going to be in {:?} inserting move to {:?}", vreg, temp, prev_alloc);
                self.add_move(inst, temp, prev_alloc, vreg.class(), InstPosition::Before);
            } else {
                trace!("{:?} prev alloc is none, so no moving here", vreg);
            }
        }

        reset_temp_idx(&mut next_temp_idx);

        // Move from temporaries to block param spillslots.
        for (pos, vreg) in self.func.branch_blockparams(block, inst, succ_idx).iter().enumerate() {
            let succ_param_vreg = succ_params[pos];
            let param_alloc = Allocation::stack(self.vreg_spillslots[succ_param_vreg.vreg()]);
            let temp_slot = self.temp_spillslots[vreg.class()][next_temp_idx[vreg.class()]];
            let temp = Allocation::stack(temp_slot);
            next_temp_idx[vreg.class()] += 1;
            trace!(" --- Placing branch arg {:?} in {:?}", vreg, temp);
            trace!("{:?} which is now in {:?} inserting move to {:?}", vreg, temp, param_alloc);
            self.add_move(inst, temp, param_alloc, vreg.class(), InstPosition::Before);
        }

        reset_temp_idx(&mut next_temp_idx);

        // Move from branch args spillslots to temporaries.
        for vreg in self.func.branch_blockparams(block, inst, succ_idx).iter() {
            if self.vreg_spillslots[vreg.vreg()].is_invalid() {
                self.vreg_spillslots[vreg.vreg()] = self.allocstack(vreg);
                trace!("Block arg {:?} is going to be in {:?}", vreg, Allocation::stack(self.vreg_spillslots[vreg.vreg()]));
            }
            let temp_slot = self.temp_spillslots[vreg.class()][next_temp_idx[vreg.class()]];
            let temp = Allocation::stack(temp_slot);
            next_temp_idx[vreg.class()] += 1;
            let vreg_spill = Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
            self.vreg_allocs[vreg.vreg()] = vreg_spill;
            trace!("{:?} which is going to be in {:?} inserting move to {:?}", vreg, vreg_spill, temp);
            self.add_move(inst, vreg_spill, temp, vreg.class(), InstPosition::Before);
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
        // And because edits are inserted in reverse, the algorithm has to process
        // the branch args which are not branch params first. This will result in the
        // output code processing branch args which are params before the others.

        for succ in self.func.block_succs(block).iter() {
            self.alloc_slots_for_block_params(*succ);
        }

        for (succ_idx, succ) in self.func.block_succs(block).iter().enumerate() {
            let succ_params = self.func.block_params(*succ);

            // Move from temporaries to post block locations.
            for vreg in self.func.branch_blockparams(block, inst, succ_idx).iter() {
                self.live_vregs.insert(*vreg);
                if self.temp_spillslots[vreg.class()].len() == next_temp_idx[vreg.class()] {
                    let newslot = self.allocstack(vreg);
                    self.temp_spillslots[vreg.class()].push(newslot);
                }
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
                    self.add_move(inst, temp, prev_alloc, vreg.class(), InstPosition::Before);
                } else {
                    trace!("{:?} prev alloc is none, so no moving here", vreg);
                }
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
                self.add_move(inst, temp, param_alloc, vreg.class(), InstPosition::Before);
            }
        }

        reset_temp_idx(&mut next_temp_idx);

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
                if self.vreg_spillslots[vreg.vreg()].is_invalid() {
                    self.vreg_spillslots[vreg.vreg()] = self.allocstack(vreg);
                    trace!("Block arg {:?} is going to be in {:?}", vreg, Allocation::stack(self.vreg_spillslots[vreg.vreg()]));
                }
                let temp_slot = self.temp_spillslots[vreg.class()][next_temp_idx[vreg.class()]];
                let temp = Allocation::stack(temp_slot);
                next_temp_idx[vreg.class()] += 1;
                let vreg_spill = Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
                self.vreg_allocs[vreg.vreg()] = vreg_spill;
                trace!("{:?} which is going to be in {:?} inserting move to {:?}", vreg, vreg_spill, temp);
                self.add_move(inst, vreg_spill, temp, vreg.class(), InstPosition::Before);
            }
        }
    }

    fn alloc_inst(&mut self, block: Block, inst: Inst) {
        if self.func.is_branch(inst) {
            self.process_branch(block, inst);
        }
        let operands = self.func.inst_operands(inst);
        for (op_idx, op) in NonReuseLateOperands::new(operands) {
            self.process_operand_allocation(inst, op, op_idx);
        }
        for (_, op) in NonReuseLateDefOperands::new(operands) {
            self.freealloc(op.vreg());
        }
        for (op_idx, op) in NonReuseEarlyOperands::new(operands) {
            self.process_operand_allocation(inst, op, op_idx);
        }
        for (_, op) in NonReuseEarlyDefOperands::new(operands) {
            self.freealloc(op.vreg());
        }
        for (op_idx, op) in ReuseOperands::new(operands) {
            let OperandConstraint::Reuse(reused_idx) = op.constraint() else {
                unreachable!()
            };
            self.process_reuse_operand_allocation(inst, op, op_idx, operands[reused_idx], reused_idx);
        }
        self.edits.extend(self.pre_edits.iter().cloned());
        self.pre_edits.clear();
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
        // TODO: Get rid of this clone!!!!!!!
        let live_vregs = self.live_vregs.clone();
        trace!("In reloading, live_vregs: {:?}", live_vregs);
        for vreg in live_vregs.into_iter() {
            if self.vreg_spillslots[vreg.vreg()].is_invalid() {
                self.vreg_spillslots[vreg.vreg()] = self.allocstack(&vreg);
            }
            // The allocation where the vreg is expected to be before
            // the first instruction.
            let prev_alloc = self.vreg_allocs[vreg.vreg()];
            if prev_alloc.is_reg() {
                self.freealloc(vreg);
            }
            self.vreg_allocs[vreg.vreg()] = Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
            if self.vreg_allocs[vreg.vreg()] == prev_alloc {
                // No need to do any movements if the spillslot is where the vreg is expected to be.
                trace!("No need to reload {:?} because it's already in its expected allocation", vreg);
                continue;
            }
            trace!("Move reason: reload {:?} at begin - move from its spillslot", vreg);
            self.move_before_inst(
                self.func.block_insns(block).first(),
                vreg,
                prev_alloc,
            );
        }
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
        self.edits.reverse();

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
        edits: env.edits,
        allocs: env.allocs.allocs,
        inst_alloc_offsets: env.allocs.inst_alloc_offsets,
        num_spillslots: env.num_spillslots as usize,
        debug_locations: Vec::new(),
        safepoint_slots: Vec::new(),
        stats: env.stats,
    })
}
