use crate::{cfg::CFGInfo, ion::Stats, Allocation, RegAllocError};
use crate::{ssa::validate_ssa, Edit, Function, MachineEnv, Output, ProgPoint};
use crate::{
    AllocationKind, Block, Inst, InstPosition, Operand, OperandConstraint, OperandKind, OperandPos,
    PReg, PRegSet, RegClass, SpillSlot, VReg,
};
use alloc::vec::Vec;
use core::convert::TryInto;
use core::iter::FromIterator;
use core::ops::{Index, IndexMut};

mod iter;
mod lru;
mod vregset;
use iter::*;
use lru::*;
use vregset::VRegSet;

#[cfg(test)]
mod tests;

#[derive(Debug)]
struct Allocs {
    allocs: Vec<Allocation>,
    /// `inst_alloc_offsets[i]` is the offset into `allocs` for the allocations of
    /// instruction `i`'s operands.
    inst_alloc_offsets: Vec<u32>,
}

impl Allocs {
    fn new<F: Function>(func: &F) -> (Self, u32) {
        let operand_no_guess = func.num_insts() * 3;
        let mut allocs = Vec::with_capacity(operand_no_guess);
        let mut inst_alloc_offsets = Vec::with_capacity(operand_no_guess);
        let mut max_operand_len = 0;
        for inst in 0..func.num_insts() {
            let operands_len = func.inst_operands(Inst::new(inst)).len() as u32;
            max_operand_len = max_operand_len.max(operands_len);
            inst_alloc_offsets.push(allocs.len() as u32);
            for _ in 0..operands_len {
                allocs.push(Allocation::none());
            }
        }
        (
            Self {
                allocs,
                inst_alloc_offsets,
            },
            max_operand_len,
        )
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
struct Stack<'a, F: Function> {
    num_spillslots: u32,
    func: &'a F,
}

impl<'a, F: Function> Stack<'a, F> {
    fn new(func: &'a F) -> Self {
        Self {
            num_spillslots: 0,
            func,
        }
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
}

#[derive(Debug)]
struct Edits {
    /// The final output edits.
    edits: Vec<(ProgPoint, Edit)>,
    fixed_stack_slots: PRegSet,
    /// The scratch registers being used in the instruction being
    /// currently processed.
    scratch_regs: PartedByRegClass<Option<PReg>>,
    dedicated_scratch_regs: PartedByRegClass<Option<PReg>>,
}

impl Edits {
    fn new(
        fixed_stack_slots: PRegSet,
        max_operand_len: u32,
        num_insts: usize,
        dedicated_scratch_regs: PartedByRegClass<Option<PReg>>,
    ) -> Self {
        // Some operands generate edits and some don't.
        // The operands that generate edits add no more than two.
        // Some edits are added due to clobbers, not operands.
        // Anyways, I think this may be a reasonable guess.
        let inst_edits_len_guess = max_operand_len as usize * 2;
        let total_edits_len_guess = inst_edits_len_guess * num_insts;
        Self {
            edits: Vec::with_capacity(total_edits_len_guess),
            fixed_stack_slots,
            scratch_regs: dedicated_scratch_regs.clone(),
            dedicated_scratch_regs,
        }
    }
}

impl Edits {
    fn is_stack(&self, alloc: Allocation) -> bool {
        if alloc.is_stack() {
            return true;
        }
        if alloc.is_reg() {
            return self.fixed_stack_slots.contains(alloc.as_reg().unwrap());
        }
        false
    }

    fn add_move(
        &mut self,
        inst: Inst,
        from: Allocation,
        to: Allocation,
        class: RegClass,
        pos: InstPosition,
    ) {
        trace!(
            "Recording edit: {:?}",
            (ProgPoint::new(inst, pos), Edit::Move { from, to }, class)
        );
        if self.is_stack(from) && self.is_stack(to) {
            trace!("Edit is stack-to-stack. Generating two edits with a scratch register");
            let scratch_reg = self.scratch_regs[class].unwrap();
            let scratch_alloc = Allocation::reg(scratch_reg);
            trace!("Move 1: {scratch_alloc:?} to {to:?}");
            self.edits.push((
                ProgPoint::new(inst, pos),
                Edit::Move {
                    from: scratch_alloc,
                    to,
                },
            ));
            trace!("Move 2: {from:?} to {scratch_alloc:?}");
            self.edits.push((
                ProgPoint::new(inst, pos),
                Edit::Move {
                    from,
                    to: scratch_alloc,
                },
            ));
        } else {
            self.edits
                .push((ProgPoint::new(inst, pos), Edit::Move { from, to }));
        }
    }
}

#[derive(Debug, Clone)]
struct PartedByOperandPos<T> {
    items: [T; 2],
}

impl<T> Index<OperandPos> for PartedByOperandPos<T> {
    type Output = T;
    fn index(&self, index: OperandPos) -> &Self::Output {
        &self.items[index as usize]
    }
}

impl<T> IndexMut<OperandPos> for PartedByOperandPos<T> {
    fn index_mut(&mut self, index: OperandPos) -> &mut Self::Output {
        &mut self.items[index as usize]
    }
}

use core::fmt;

impl<T: fmt::Display> fmt::Display for PartedByOperandPos<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ early: {}, late: {} }}", self.items[0], self.items[1])
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
    live_vregs: VRegSet,
    /// Least-recently-used caches for register classes Int, Float, and Vector, respectively.
    lrus: Lrus,
    /// `vreg_in_preg[i]` is the virtual register currently in the physical register
    /// with index `i`.
    vreg_in_preg: Vec<VReg>,
    /// For parallel moves from branch args to block param spillslots.
    temp_spillslots: PartedByRegClass<Vec<SpillSlot>>,
    /// `reused_input_to_reuse_op[i]` is the operand index of the reuse operand
    /// that uses the `i`th operand in the current instruction as its input.
    reused_input_to_reuse_op: Vec<usize>,
    /// The set of registers that can be used for allocation in the
    /// early and late phases of an instruction.
    /// Allocatable registers that contain no vregs, registers that can be
    /// evicted can be in the set, and fixed stack slots are in this set.
    available_pregs: PartedByOperandPos<PRegSet>,
    init_available_pregs: PRegSet,
    allocatable_regs: PRegSet,
    stack: Stack<'a, F>,
    vreg_to_live_inst_range: Vec<(ProgPoint, ProgPoint, Allocation)>,

    fixed_stack_slots: PRegSet,

    // Output.
    allocs: Allocs,
    edits: Edits,
    stats: Stats,
    debug_locations: Vec<(u32, ProgPoint, ProgPoint, Allocation)>,
}

impl<'a, F: Function> Env<'a, F> {
    fn new(func: &'a F, env: &'a MachineEnv) -> Self {
        use alloc::vec;
        let mut regs = [
            env.preferred_regs_by_class[RegClass::Int as usize].clone(),
            env.preferred_regs_by_class[RegClass::Float as usize].clone(),
            env.preferred_regs_by_class[RegClass::Vector as usize].clone(),
        ];
        regs[0].extend(
            env.non_preferred_regs_by_class[RegClass::Int as usize]
                .iter()
                .cloned(),
        );
        regs[1].extend(
            env.non_preferred_regs_by_class[RegClass::Float as usize]
                .iter()
                .cloned(),
        );
        regs[2].extend(
            env.non_preferred_regs_by_class[RegClass::Vector as usize]
                .iter()
                .cloned(),
        );
        let allocatable_regs = PRegSet::from(env);
        let init_available_pregs = {
            let mut regs = allocatable_regs;
            for preg in env.fixed_stack_slots.iter() {
                regs.add(*preg);
            }
            regs
        };
        let dedicated_scratch_regs = PartedByRegClass {
            items: [
                env.scratch_by_class[0],
                env.scratch_by_class[1],
                env.scratch_by_class[2],
            ],
        };
        trace!("{:?}", env);
        let (allocs, max_operand_len) = Allocs::new(func);
        let fixed_stack_slots = PRegSet::from_iter(env.fixed_stack_slots.iter().cloned());
        Self {
            func,
            allocatable_regs,
            vreg_allocs: vec![Allocation::none(); func.num_vregs()],
            vreg_spillslots: vec![SpillSlot::invalid(); func.num_vregs()],
            live_vregs: VRegSet::with_capacity(func.num_vregs()),
            lrus: Lrus::new(&regs[0], &regs[1], &regs[2]),
            vreg_in_preg: vec![VReg::invalid(); PReg::NUM_INDEX],
            stack: Stack::new(func),
            fixed_stack_slots,
            vreg_to_live_inst_range: vec![
                (
                    ProgPoint::invalid(),
                    ProgPoint::invalid(),
                    Allocation::none()
                );
                func.num_vregs()
            ],
            temp_spillslots: PartedByRegClass {
                items: [
                    Vec::with_capacity(func.num_vregs()),
                    Vec::with_capacity(func.num_vregs()),
                    Vec::with_capacity(func.num_vregs()),
                ],
            },
            reused_input_to_reuse_op: vec![usize::MAX; max_operand_len as usize],
            init_available_pregs,
            available_pregs: PartedByOperandPos {
                items: [init_available_pregs, init_available_pregs],
            },
            allocs,
            edits: Edits::new(
                fixed_stack_slots,
                max_operand_len,
                func.num_insts(),
                dedicated_scratch_regs,
            ),
            stats: Stats::default(),
            debug_locations: Vec::with_capacity(func.debug_value_labels().len()),
        }
    }

    fn is_stack(&self, alloc: Allocation) -> bool {
        if alloc.is_stack() {
            return true;
        }
        if alloc.is_reg() {
            return self.fixed_stack_slots.contains(alloc.as_reg().unwrap());
        }
        false
    }

    fn reset_available_pregs_and_scratch_regs(&mut self) {
        trace!("Resetting the available pregs");
        self.available_pregs = PartedByOperandPos {
            items: [self.init_available_pregs, self.init_available_pregs],
        };
        self.edits.scratch_regs = self.edits.dedicated_scratch_regs.clone();
    }

    fn get_scratch_reg(&self, class: RegClass) -> Result<PReg, RegAllocError> {
        let mut avail_regs = self.available_pregs[OperandPos::Early];
        avail_regs.intersect_from(self.available_pregs[OperandPos::Late]);
        self.lrus[class]
            .last(avail_regs)
            .ok_or(RegAllocError::TooManyLiveRegs)
    }

    fn reserve_reg_for_fixed_operand(
        &mut self,
        op: Operand,
        op_idx: usize,
        preg: PReg,
    ) -> Result<(), RegAllocError> {
        trace!("Reserving register {preg} for fixed operand {op}");
        let early_avail_pregs = self.available_pregs[OperandPos::Early];
        let late_avail_pregs = self.available_pregs[OperandPos::Late];
        match (op.pos(), op.kind()) {
            (OperandPos::Early, OperandKind::Use) => {
                if op.as_fixed_nonallocatable().is_none() && !early_avail_pregs.contains(preg) {
                    return Err(RegAllocError::TooManyLiveRegs);
                }
                self.available_pregs[OperandPos::Early].remove(preg);
                if self.reused_input_to_reuse_op[op_idx] != usize::MAX {
                    if op.as_fixed_nonallocatable().is_none() && !late_avail_pregs.contains(preg) {
                        return Err(RegAllocError::TooManyLiveRegs);
                    }
                    self.available_pregs[OperandPos::Late].remove(preg);
                }
            }
            (OperandPos::Late, OperandKind::Def) => {
                if op.as_fixed_nonallocatable().is_none() && !late_avail_pregs.contains(preg) {
                    return Err(RegAllocError::TooManyLiveRegs);
                }
                self.available_pregs[OperandPos::Late].remove(preg);
            }
            _ => {
                if op.as_fixed_nonallocatable().is_none()
                    && (!early_avail_pregs.contains(preg) || !late_avail_pregs.contains(preg))
                {
                    return Err(RegAllocError::TooManyLiveRegs);
                }
                self.available_pregs[OperandPos::Early].remove(preg);
                self.available_pregs[OperandPos::Late].remove(preg);
            }
        }
        Ok(())
    }

    fn allocd_within_constraint(&self, inst: Inst, op: Operand) -> bool {
        let alloc = self.vreg_allocs[op.vreg().vreg()];
        let alloc_is_clobber = if let Some(preg) = alloc.as_reg() {
            self.func.inst_clobbers(inst).contains(preg)
        } else {
            false
        };
        match op.constraint() {
            OperandConstraint::Any => {
                // Completely avoid assigning clobbers, if possible.
                // Assigning a clobber to a def operand that lives past the
                // current instruction makes it impossible to restore
                // the vreg.
                // And assigning a clobber to a use operand that is reused
                // by a def operand with a reuse constraint will end up
                // assigning the clobber to that def, and if it lives past
                // the current instruction, then restoration will be impossible.
                if alloc_is_clobber {
                    return false;
                }
                if let Some(preg) = alloc.as_reg() {
                    if !self.available_pregs[op.pos()].contains(preg) {
                        // If a register isn't in the available pregs list, then
                        // there are two cases: either it's reserved for a
                        // fixed register constraint or a vreg allocated in the instruction
                        // is already assigned to it.
                        //
                        // For example:
                        // 1. use v0, use v0, use v0
                        //
                        // Say p0 is assigned to v0 during the processing of the first operand.
                        // When the second v0 operand is being processed, v0 will still be in
                        // v0, so it is still allocated within constraints.
                        trace!("The vreg in {preg}: {}", self.vreg_in_preg[preg.index()]);
                        self.vreg_in_preg[preg.index()] == op.vreg()
                    } else {
                        true
                    }
                } else {
                    !alloc.is_none()
                }
            }
            OperandConstraint::Reg => {
                if alloc_is_clobber {
                    return false;
                }
                if self.is_stack(alloc) {
                    return false;
                }
                if let Some(preg) = alloc.as_reg() {
                    if !self.available_pregs[op.pos()].contains(preg) {
                        trace!("The vreg in {preg}: {}", self.vreg_in_preg[preg.index()]);
                        self.vreg_in_preg[preg.index()] == op.vreg()
                    } else {
                        true
                    }
                } else {
                    false
                }
            }
            // It is possible for an operand to have a fixed register constraint to
            // a clobber.
            OperandConstraint::FixedReg(preg) => alloc.is_reg() && alloc.as_reg().unwrap() == preg,
            OperandConstraint::Reuse(_) => {
                unreachable!()
            }
        }
    }

    fn evict_vreg_in_preg(&mut self, inst: Inst, preg: PReg) {
        trace!("Removing the vreg in preg {} for eviction", preg);
        let evicted_vreg = self.vreg_in_preg[preg.index()];
        trace!("The removed vreg: {}", evicted_vreg);
        debug_assert_ne!(evicted_vreg, VReg::invalid());
        if self.vreg_spillslots[evicted_vreg.vreg()].is_invalid() {
            self.vreg_spillslots[evicted_vreg.vreg()] = self.stack.allocstack(&evicted_vreg);
        }
        let slot = self.vreg_spillslots[evicted_vreg.vreg()];
        self.vreg_allocs[evicted_vreg.vreg()] = Allocation::stack(slot);
        trace!("Move reason: eviction");
        self.edits.add_move(
            inst,
            self.vreg_allocs[evicted_vreg.vreg()],
            Allocation::reg(preg),
            evicted_vreg.class(),
            InstPosition::After,
        );
    }

    fn freealloc(&mut self, vreg: VReg) {
        trace!("Freeing vreg {}", vreg);
        let alloc = self.vreg_allocs[vreg.vreg()];
        match alloc.kind() {
            AllocationKind::Reg => {
                let preg = alloc.as_reg().unwrap();
                self.vreg_in_preg[preg.index()] = VReg::invalid();
            }
            AllocationKind::Stack => (),
            AllocationKind::None => unreachable!("Attempting to free an unallocated operand!"),
        }
        self.vreg_allocs[vreg.vreg()] = Allocation::none();
        self.live_vregs.remove(vreg.vreg());
        trace!(
            "{} curr alloc is now {}",
            vreg,
            self.vreg_allocs[vreg.vreg()]
        );
    }

    /// Allocates a physical register for the operand `op`.
    fn alloc_reg_for_operand(
        &mut self,
        inst: Inst,
        op: Operand,
    ) -> Result<Allocation, RegAllocError> {
        trace!("available regs: {}", self.available_pregs);
        trace!("Int LRU: {:?}", self.lrus[RegClass::Int]);
        trace!("Float LRU: {:?}", self.lrus[RegClass::Float]);
        trace!("Vector LRU: {:?}", self.lrus[RegClass::Vector]);
        trace!("");
        if self.available_pregs[op.pos()].is_empty(op.class()) {
            trace!("No registers available in class {:?}", op.class());
            return Err(RegAllocError::TooManyLiveRegs);
        }
        let Some(preg) = self.lrus[op.class()].last(self.available_pregs[op.pos()]) else {
            trace!(
                "Failed to find an available {:?} register in the LRU for operand {op}",
                op.class()
            );
            return Err(RegAllocError::TooManyLiveRegs);
        };
        if self.vreg_in_preg[preg.index()] != VReg::invalid() {
            self.evict_vreg_in_preg(inst, preg);
        }
        trace!("The allocated register for vreg {}: {}", op.vreg(), preg);
        self.lrus[op.class()].poke(preg);
        self.available_pregs[op.pos()].remove(preg);
        match (op.pos(), op.kind()) {
            (OperandPos::Late, OperandKind::Use) => {
                self.available_pregs[OperandPos::Early].remove(preg);
            }
            (OperandPos::Early, OperandKind::Def) => {
                self.available_pregs[OperandPos::Late].remove(preg);
            }
            (OperandPos::Late, OperandKind::Def)
                if matches!(op.constraint(), OperandConstraint::Reuse(_)) =>
            {
                self.available_pregs[OperandPos::Early].remove(preg);
            }
            _ => (),
        };
        Ok(Allocation::reg(preg))
    }

    /// Allocates for the operand `op` with index `op_idx` into the
    /// vector of instruction `inst`'s operands.
    fn alloc_operand(
        &mut self,
        inst: Inst,
        op: Operand,
        op_idx: usize,
    ) -> Result<Allocation, RegAllocError> {
        let new_alloc = match op.constraint() {
            OperandConstraint::Any => self.alloc_reg_for_operand(inst, op)?,
            OperandConstraint::Reg => self.alloc_reg_for_operand(inst, op)?,
            OperandConstraint::FixedReg(preg) => {
                trace!("The fixed preg: {} for operand {}", preg, op);

                Allocation::reg(preg)
            }
            OperandConstraint::Reuse(_) => {
                // This is handled elsewhere.
                unreachable!();
            }
        };
        self.allocs[(inst.index(), op_idx)] = new_alloc;
        Ok(new_alloc)
    }

    /// Allocate operand the `op_idx`th operand `op` in instruction `inst` within its constraint.
    /// Since only fixed register constraints are allowed, `fixed_spillslot` is used when a
    /// fixed stack allocation is needed, like when transferring a stack allocation from a
    /// reuse operand allocation to the reused input.
    fn process_operand_allocation(
        &mut self,
        inst: Inst,
        op: Operand,
        op_idx: usize,
    ) -> Result<(), RegAllocError> {
        if let Some(preg) = op.as_fixed_nonallocatable() {
            self.allocs[(inst.index(), op_idx)] = Allocation::reg(preg);
            trace!(
                "Allocation for instruction {:?} and operand {}: {}",
                inst,
                op,
                self.allocs[(inst.index(), op_idx)]
            );
            return Ok(());
        }
        if !self.allocd_within_constraint(inst, op) {
            trace!("{op} isn't allocated within constraints.");
            let curr_alloc = self.vreg_allocs[op.vreg().vreg()];
            let new_alloc = self.alloc_operand(inst, op, op_idx)?;
            if curr_alloc.is_none() {
                self.live_vregs.insert(op.vreg());
                self.vreg_to_live_inst_range[op.vreg().vreg()].1 = match (op.pos(), op.kind()) {
                    (OperandPos::Late, OperandKind::Use) | (_, OperandKind::Def) => {
                        // Live range ends just before the early phase of the
                        // next instruction.
                        ProgPoint::before(Inst::new(inst.index() + 1))
                    }
                    (OperandPos::Early, OperandKind::Use) => {
                        // Live range ends just before the late phase of the current instruction.
                        ProgPoint::after(inst)
                    }
                };
                self.vreg_to_live_inst_range[op.vreg().vreg()].2 = new_alloc;

                trace!("Setting vreg_allocs[{op}] to {new_alloc:?}");
                self.vreg_allocs[op.vreg().vreg()] = new_alloc;
                if let Some(preg) = new_alloc.as_reg() {
                    self.vreg_in_preg[preg.index()] = op.vreg();
                }
            }
            // Need to insert a move to propagate flow from the current
            // allocation to the subsequent places where the value was
            // used (in `prev_alloc`, that is).
            else if curr_alloc.is_some() {
                trace!("Move reason: Prev allocation doesn't meet constraints");
                if self.is_stack(new_alloc)
                    && self.is_stack(curr_alloc)
                    && self.edits.scratch_regs[op.class()].is_none()
                {
                    let reg = self.get_scratch_reg(op.class())?;
                    self.edits.scratch_regs[op.class()] = Some(reg);
                    self.available_pregs[OperandPos::Early].remove(reg);
                    self.available_pregs[OperandPos::Late].remove(reg);
                }
                if op.kind() == OperandKind::Def {
                    trace!("Adding edit from {new_alloc:?} to {curr_alloc:?} after inst {inst:?} for {op}");
                    self.edits.add_move(
                        inst,
                        new_alloc,
                        curr_alloc,
                        op.class(),
                        InstPosition::After,
                    );
                    // No need to set vreg_in_preg because it will be set during
                    // `freealloc` if needed.
                }
                // Edits for use operands are added later to avoid inserting
                // edits out of order.

                if let Some(preg) = new_alloc.as_reg() {
                    // Don't change the allocation.
                    self.vreg_in_preg[preg.index()] = VReg::invalid();
                }
            }
            trace!(
                "Allocation for instruction {:?} and operand {}: {}",
                inst,
                op,
                self.allocs[(inst.index(), op_idx)]
            );
        } else {
            trace!("{op} is already allocated within constraints");
            self.allocs[(inst.index(), op_idx)] = self.vreg_allocs[op.vreg().vreg()];
            if let Some(preg) = self.allocs[(inst.index(), op_idx)].as_reg() {
                if self.allocatable_regs.contains(preg) {
                    self.lrus[preg.class()].poke(preg);
                }
                self.available_pregs[op.pos()].remove(preg);
                match (op.pos(), op.kind()) {
                    (OperandPos::Late, OperandKind::Use) => {
                        self.available_pregs[OperandPos::Early].remove(preg);
                    }
                    (OperandPos::Early, OperandKind::Def) => {
                        self.available_pregs[OperandPos::Late].remove(preg);
                    }
                    _ => (),
                };
            }
            trace!(
                "Allocation for instruction {:?} and operand {}: {}",
                inst,
                op,
                self.allocs[(inst.index(), op_idx)]
            );
        }
        trace!(
            "Late available regs: {}",
            self.available_pregs[OperandPos::Late]
        );
        trace!(
            "Early available regs: {}",
            self.available_pregs[OperandPos::Early]
        );
        Ok(())
    }

    fn remove_clobbers_from_available_pregs(&mut self, clobbers: PRegSet) {
        trace!("Removing clobbers {clobbers} from available reg sets");
        // Don't let defs get allocated to clobbers.
        // Consider a scenario:
        //
        // 1. (early|late) def v0 (reg). Clobbers: [p0]
        // 2. use v0 (fixed: p0)
        //
        // If p0 isn't removed from the both available reg sets, then
        // p0 could get allocated to v0 in inst 1, making it impossible
        // to restore it after the instruction.
        // To avoid this scenario, clobbers should be removed from both late
        // and early reg sets.
        let all_but_clobbers = clobbers.invert();
        self.available_pregs[OperandPos::Late].intersect_from(all_but_clobbers);
        self.available_pregs[OperandPos::Early].intersect_from(all_but_clobbers);
    }

    /// If instruction `inst` is a branch in `block`,
    /// this function places branch arguments in the spillslots
    /// expected by the destination blocks.
    ///
    /// The process used to do this is as follows:
    ///
    /// 1. Move all branch arguments into corresponding temporary spillslots.
    /// 2. Move values from the temporary spillslots to corresponding block param spillslots.
    ///
    /// These temporaries are used because the moves have to be parallel in the case where
    /// a block parameter of the successor block is a branch argument.
    fn process_branch(&mut self, block: Block, inst: Inst) -> Result<(), RegAllocError> {
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

        trace!("Adding temp to block params spillslots for branch args");
        for (succ_idx, succ) in self.func.block_succs(block).iter().enumerate() {
            let succ_params = self.func.block_params(*succ);

            // Move from temporaries to block param spillslots.
            for (pos, vreg) in self
                .func
                .branch_blockparams(block, inst, succ_idx)
                .iter()
                .enumerate()
            {
                if self.temp_spillslots[vreg.class()].len() == next_temp_idx[vreg.class()] {
                    let newslot = self.stack.allocstack(vreg);
                    self.temp_spillslots[vreg.class()].push(newslot);
                }
                let succ_param_vreg = succ_params[pos];
                if self.vreg_spillslots[succ_param_vreg.vreg()].is_invalid() {
                    self.vreg_spillslots[succ_param_vreg.vreg()] =
                        self.stack.allocstack(&succ_param_vreg);
                    trace!(
                        "Block param {} is in {}",
                        vreg,
                        Allocation::stack(self.vreg_spillslots[vreg.vreg()])
                    );
                }
                let param_alloc = Allocation::stack(self.vreg_spillslots[succ_param_vreg.vreg()]);
                let temp_slot = self.temp_spillslots[vreg.class()][next_temp_idx[vreg.class()]];
                let temp = Allocation::stack(temp_slot);
                next_temp_idx[vreg.class()] += 1;
                trace!(" Branch arg {vreg} from {temp} to {param_alloc}");
                if self.edits.scratch_regs[vreg.class()].is_none() {
                    let reg = self.get_scratch_reg(vreg.class())?;
                    // No need to remove the scratch register from the available reg sets
                    // because branches are processed last.
                    self.edits.scratch_regs[vreg.class()] = Some(reg);
                }
                self.edits
                    .add_move(inst, temp, param_alloc, vreg.class(), InstPosition::Before);
            }
        }

        reset_temp_idx(&mut next_temp_idx);

        for (succ_idx, _) in self.func.block_succs(block).iter().enumerate() {
            // Move from branch args spillslots to temporaries.
            //
            // Consider a scenario:
            //
            // block entry:
            //      goto Y(...)
            //
            // block Y(vp)
            //      goto X
            //
            // block X
            //      use vp
            //      goto Y(va)
            //
            // block X branches to block Y and block Y branches to block X.
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
                    self.vreg_spillslots[vreg.vreg()] = self.stack.allocstack(vreg);
                    trace!(
                        "Block arg {} is going to be in {}",
                        vreg,
                        Allocation::stack(self.vreg_spillslots[vreg.vreg()])
                    );
                }
                let temp_slot = self.temp_spillslots[vreg.class()][next_temp_idx[vreg.class()]];
                let temp = Allocation::stack(temp_slot);
                next_temp_idx[vreg.class()] += 1;
                let vreg_spill = Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
                trace!(
                    "{} which is going to be in {} inserting move to {}",
                    vreg,
                    vreg_spill,
                    temp
                );

                self.edits
                    .add_move(inst, vreg_spill, temp, vreg.class(), InstPosition::Before);
                // All branch arguments should be in their spillslots at the end of the function.
                if self.vreg_allocs[vreg.vreg()].is_none() {
                    self.live_vregs.insert(*vreg);
                    let slot = self.vreg_spillslots[vreg.vreg()];
                    self.vreg_allocs[vreg.vreg()] = Allocation::stack(slot);
                    self.vreg_to_live_inst_range[vreg.vreg()].1 = ProgPoint::before(inst);
                } else if self.vreg_allocs[vreg.vreg()] != vreg_spill {
                    self.edits.add_move(
                        inst,
                        self.vreg_allocs[vreg.vreg()],
                        vreg_spill,
                        vreg.class(),
                        InstPosition::Before,
                    );
                }
            }
        }

        Ok(())
    }

    fn alloc_inst(&mut self, block: Block, inst: Inst) -> Result<(), RegAllocError> {
        trace!("Allocating instruction {:?}", inst);
        let operands = Operands::new(self.func.inst_operands(inst));
        let clobbers = self.func.inst_clobbers(inst);

        for (op_idx, op) in operands.reuse() {
            trace!("Initializing reused_input_to_reuse_op for {op}");
            let OperandConstraint::Reuse(reused_idx) = op.constraint() else {
                unreachable!()
            };
            self.reused_input_to_reuse_op[reused_idx] = op_idx;
        }
        for (op_idx, op) in operands.fixed() {
            let OperandConstraint::FixedReg(preg) = op.constraint() else {
                unreachable!();
            };
            self.reserve_reg_for_fixed_operand(op, op_idx, preg)?;
            if self.allocatable_regs.contains(preg) {
                self.lrus[preg.class()].poke(preg);
            }
        }
        for (_, op) in operands.fixed() {
            let OperandConstraint::FixedReg(preg) = op.constraint() else {
                unreachable!();
            };
            // Eviction has to be done separately to avoid using a fixed register
            // as a scratch register.
            if self.vreg_in_preg[preg.index()] != VReg::invalid()
                && self.vreg_in_preg[preg.index()] != op.vreg()
            {
                trace!(
                    "Evicting {} from fixed register {preg}",
                    self.vreg_in_preg[preg.index()]
                );
                if self.fixed_stack_slots.contains(preg)
                    && self.edits.scratch_regs[preg.class()].is_none()
                {
                    let reg = self.get_scratch_reg(preg.class())?;
                    self.edits.scratch_regs[preg.class()] = Some(reg);
                    self.available_pregs[OperandPos::Early].remove(reg);
                    self.available_pregs[OperandPos::Late].remove(reg);
                }
                self.evict_vreg_in_preg(inst, preg);
                self.vreg_in_preg[preg.index()] = VReg::invalid();
            }
        }
        self.remove_clobbers_from_available_pregs(clobbers);
        for preg in clobbers {
            if self.vreg_in_preg[preg.index()] != VReg::invalid() {
                trace!(
                    "Evicting {} from clobber {preg}",
                    self.vreg_in_preg[preg.index()]
                );
                if self.fixed_stack_slots.contains(preg)
                    && self.edits.scratch_regs[preg.class()].is_none()
                {
                    let reg = self.get_scratch_reg(preg.class())?;
                    self.edits.scratch_regs[preg.class()] = Some(reg);
                    self.available_pregs[OperandPos::Early].remove(reg);
                    self.available_pregs[OperandPos::Late].remove(reg);
                }
                self.evict_vreg_in_preg(inst, preg);
                self.vreg_in_preg[preg.index()] = VReg::invalid();
            }
        }
        for (_, op) in operands.non_fixed_use() {
            if op.as_fixed_nonallocatable().is_some() {
                continue;
            }
            if let Some(preg) = self.vreg_allocs[op.vreg().vreg()].as_reg() {
                trace!("Removing {op}'s current reg allocation {preg} from reg sets");
                // The current allocation, vreg_allocs[op.vreg], doesn't change,
                // so it should be removed from the available reg sets to avoid
                // allocating it to some other operand in the instruction.
                //
                // For example:
                // 1. def v0 (reuse: 1), use v1, use v2
                // 2. use v1 (fixed: p0)
                //
                // When inst 1 is about to be processed, vreg_allocs[v1] will be p0.
                // Suppose p1 is allocated to v0: this will create a fixed constraint for
                // v1 and p1 will also be allocated to it.
                // When it's time to process the v2 operand, vreg_allocs[v1] will still be p0
                // because it doesn't change (except by an explicit fixed reg constraint which
                // will not be a problem here) and it's possible for v2 to get p0 as an allocation,
                // which is wrong. That will lead to the following scenario:
                //
                // move from p0 to p1   // Inserted due to reuse constraints
                //                      // (vreg_allocs[v1] == p0)
                // 1. def v0 (reuse: 1), use v1, use v2 // v0: p1, v1: p1, v2: p0
                // move from stack_v0 to p0 // Eviction here because v0 is still in p0 when
                //                          // v2's processing picked p0 from available regs
                // 2. use v1 (fixed: p0)
                //
                // To avoid this scenario, the register is removed from the available set.
                self.available_pregs[op.pos()].remove(preg);
                if let (OperandPos::Late, OperandKind::Use) = (op.pos(), op.kind()) {
                    self.available_pregs[OperandPos::Early].remove(preg);
                }
            }
        }
        for (op_idx, op) in operands.def_ops() {
            trace!("Allocating def operands {op}");
            if let OperandConstraint::Reuse(reused_idx) = op.constraint() {
                let reused_op = operands[reused_idx];
                let new_reuse_op =
                    Operand::new(op.vreg(), reused_op.constraint(), op.kind(), op.pos());
                trace!("allocating reuse op {op} as {new_reuse_op}");
                self.process_operand_allocation(inst, new_reuse_op, op_idx)?;
            } else {
                self.process_operand_allocation(inst, op, op_idx)?;
            }
            let slot = self.vreg_spillslots[op.vreg().vreg()];
            if slot.is_valid() {
                self.vreg_to_live_inst_range[op.vreg().vreg()].2 = Allocation::stack(slot);
                let curr_alloc = self.vreg_allocs[op.vreg().vreg()];
                let vreg_slot = self.vreg_spillslots[op.vreg().vreg()];
                let (is_stack_to_stack, src_and_dest_are_same) =
                    if let Some(curr_alloc) = curr_alloc.as_stack() {
                        (true, curr_alloc == vreg_slot)
                    } else {
                        (self.is_stack(curr_alloc), false)
                    };
                if !src_and_dest_are_same {
                    if is_stack_to_stack && self.edits.scratch_regs[op.class()].is_none() {
                        let reg = self.get_scratch_reg(op.class())?;
                        self.edits.scratch_regs[op.class()] = Some(reg);
                        self.available_pregs[OperandPos::Early].remove(reg);
                        self.available_pregs[OperandPos::Late].remove(reg);
                    };
                    self.edits.add_move(
                        inst,
                        self.vreg_allocs[op.vreg().vreg()],
                        Allocation::stack(self.vreg_spillslots[op.vreg().vreg()]),
                        op.class(),
                        InstPosition::After,
                    );
                }
            }
            self.vreg_to_live_inst_range[op.vreg().vreg()].0 = ProgPoint::after(inst);
            self.freealloc(op.vreg());
        }
        for (op_idx, op) in operands.use_ops() {
            trace!("Allocating use operand {op}");
            if self.reused_input_to_reuse_op[op_idx] != usize::MAX {
                let reuse_op_idx = self.reused_input_to_reuse_op[op_idx];
                let reuse_op_alloc = self.allocs[(inst.index(), reuse_op_idx)];
                let Some(preg) = reuse_op_alloc.as_reg() else {
                    unreachable!();
                };
                let new_reused_input_constraint = OperandConstraint::FixedReg(preg);
                let new_reused_input =
                    Operand::new(op.vreg(), new_reused_input_constraint, op.kind(), op.pos());
                trace!("Allocating reused input {op} as {new_reused_input}");
                self.process_operand_allocation(inst, new_reused_input, op_idx)?;
            } else {
                self.process_operand_allocation(inst, op, op_idx)?;
            }
        }
        for (op_idx, op) in operands.use_ops() {
            if op.as_fixed_nonallocatable().is_some() {
                continue;
            }
            if self.vreg_allocs[op.vreg().vreg()] != self.allocs[(inst.index(), op_idx)] {
                let curr_alloc = self.vreg_allocs[op.vreg().vreg()];
                let new_alloc = self.allocs[(inst.index(), op_idx)];
                trace!("Adding edit from {curr_alloc:?} to {new_alloc:?} before inst {inst:?} for {op}");
                self.edits.add_move(
                    inst,
                    curr_alloc,
                    new_alloc,
                    op.class(),
                    InstPosition::Before,
                );
            }
        }
        if self.func.is_branch(inst) {
            self.process_branch(block, inst)?;
        }
        for entry in self.reused_input_to_reuse_op.iter_mut() {
            *entry = usize::MAX;
        }
        if trace_enabled!() {
            self.log_post_inst_processing_state(inst);
        }
        Ok(())
    }

    /// At the beginning of every block, all virtual registers that are
    /// livein are expected to be in their respective spillslots.
    /// This function sets the current allocations of livein registers
    /// to their spillslots and inserts the edits to flow livein values to
    /// the allocations where they are expected to be before the first
    /// instruction.
    fn reload_at_begin(&mut self, block: Block) -> Result<(), RegAllocError> {
        trace!(
            "Reloading live registers at the beginning of block {:?}",
            block
        );
        trace!(
            "Live registers at the beginning of block {:?}: {:?}",
            block,
            self.live_vregs
        );
        trace!(
            "Block params at block {:?} beginning: {:?}",
            block,
            self.func.block_params(block)
        );
        trace!(
            "Available pregs: {}",
            self.available_pregs[OperandPos::Early]
        );
        let mut available_regs_for_scratch = self.available_pregs[OperandPos::Early];
        let first_inst = self.func.block_insns(block).first();
        // We need to check for the registers that are still live.
        // These registers are either livein or block params
        // Liveins should be stack-allocated and block params should be freed.
        for vreg in self.func.block_params(block).iter().cloned() {
            trace!("Processing {}", vreg);
            if self.vreg_allocs[vreg.vreg()] == Allocation::none() {
                // If this block param was never used, its allocation will
                // be none at this point.
                continue;
            }
            if self.vreg_spillslots[vreg.vreg()].is_invalid() {
                self.vreg_spillslots[vreg.vreg()] = self.stack.allocstack(&vreg);
            }
            // The allocation where the vreg is expected to be before
            // the first instruction.
            let prev_alloc = self.vreg_allocs[vreg.vreg()];
            let slot = Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
            self.vreg_to_live_inst_range[vreg.vreg()].2 = slot;
            self.vreg_to_live_inst_range[vreg.vreg()].0 = ProgPoint::before(first_inst);
            trace!("{} is a block param. Freeing it", vreg);
            // A block's block param is not live before the block.
            // And `vreg_allocs[i]` of a virtual register i is none for
            // dead vregs.
            self.freealloc(vreg);
            if let Some(preg) = prev_alloc.as_reg() {
                available_regs_for_scratch.remove(preg);
            } else if slot == prev_alloc {
                // No need to do any movements if the spillslot is where the vreg is expected to be.
                trace!(
                    "No need to reload {} because it's already in its expected allocation",
                    vreg
                );
                continue;
            }
            trace!(
                "Move reason: reload {} at begin - move from its spillslot",
                vreg
            );
            if self.is_stack(prev_alloc) && self.edits.scratch_regs[vreg.class()].is_none() {
                let reg = self.get_scratch_reg(vreg.class())?;
                self.edits.scratch_regs[vreg.class()] = Some(reg);
            }
            self.edits.add_move(
                self.func.block_insns(block).first(),
                slot,
                prev_alloc,
                vreg.class(),
                InstPosition::Before,
            );
        }
        for vreg in self.live_vregs.iter() {
            trace!("Processing {}", vreg);
            trace!(
                "{} is not a block param. It's a liveout vreg from some predecessor",
                vreg
            );
            if self.vreg_spillslots[vreg.vreg()].is_invalid() {
                self.vreg_spillslots[vreg.vreg()] = self.stack.allocstack(&vreg);
            }
            // The allocation where the vreg is expected to be before
            // the first instruction.
            let prev_alloc = self.vreg_allocs[vreg.vreg()];
            let slot = Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
            trace!("Setting {}'s current allocation to its spillslot", vreg);
            self.vreg_allocs[vreg.vreg()] = slot;
            if let Some(preg) = prev_alloc.as_reg() {
                trace!("{} was in {}. Removing it", preg, vreg);
                // Nothing is in that preg anymore.
                self.vreg_in_preg[preg.index()] = VReg::invalid();
                available_regs_for_scratch.remove(preg);
            }
            if slot == prev_alloc {
                // No need to do any movements if the spillslot is where the vreg is expected to be.
                trace!(
                    "No need to reload {} because it's already in its expected allocation",
                    vreg
                );
                continue;
            }
            trace!(
                "Move reason: reload {} at begin - move from its spillslot",
                vreg
            );
            if self.is_stack(prev_alloc) && self.edits.scratch_regs[vreg.class()].is_none() {
                let mut avail_regs = self.available_pregs[OperandPos::Early];
                avail_regs.intersect_from(self.available_pregs[OperandPos::Late]);
                let reg = self.lrus[vreg.class()]
                    .last(avail_regs)
                    .ok_or(RegAllocError::TooManyLiveRegs)?;
                self.edits.scratch_regs[vreg.class()] = Some(reg);
            }
            self.edits.add_move(
                self.func.block_insns(block).first(),
                slot,
                prev_alloc,
                vreg.class(),
                InstPosition::Before,
            );
        }
        if trace_enabled!() {
            self.log_post_reload_at_begin_state(block);
        }
        Ok(())
    }

    fn log_post_reload_at_begin_state(&self, block: Block) {
        use alloc::format;
        use hashbrown::HashMap;
        trace!("");
        trace!("State after instruction reload_at_begin of {:?}", block);
        let mut map = HashMap::new();
        for (vreg_idx, alloc) in self.vreg_allocs.iter().enumerate() {
            if *alloc != Allocation::none() {
                map.insert(format!("vreg{vreg_idx}"), alloc);
            }
        }
        trace!("vreg_allocs: {:?}", map);
        let mut map = HashMap::new();
        for i in 0..self.vreg_in_preg.len() {
            if self.vreg_in_preg[i] != VReg::invalid() {
                map.insert(PReg::from_index(i), self.vreg_in_preg[i]);
            }
        }
        trace!("vreg_in_preg: {:?}", map);
        trace!("Int LRU: {:?}", self.lrus[RegClass::Int]);
        trace!("Float LRU: {:?}", self.lrus[RegClass::Float]);
        trace!("Vector LRU: {:?}", self.lrus[RegClass::Vector]);
    }

    fn log_post_inst_processing_state(&self, inst: Inst) {
        use alloc::format;
        use hashbrown::HashMap;
        trace!("");
        trace!("State after instruction {:?}", inst);
        let mut map = HashMap::new();
        for (vreg_idx, alloc) in self.vreg_allocs.iter().enumerate() {
            if *alloc != Allocation::none() {
                map.insert(format!("vreg{vreg_idx}"), alloc);
            }
        }
        trace!("vreg_allocs: {:?}", map);
        let mut v = Vec::new();
        for i in 0..self.vreg_in_preg.len() {
            if self.vreg_in_preg[i] != VReg::invalid() {
                v.push(format!(
                    "{}: {}, ",
                    PReg::from_index(i),
                    self.vreg_in_preg[i]
                ));
            }
        }
        trace!("vreg_in_preg: {:?}", v);
        trace!("Int LRU: {:?}", self.lrus[RegClass::Int]);
        trace!("Float LRU: {:?}", self.lrus[RegClass::Float]);
        trace!("Vector LRU: {:?}", self.lrus[RegClass::Vector]);
        trace!("");
    }

    fn alloc_block(&mut self, block: Block) -> Result<(), RegAllocError> {
        trace!("{:?} start", block);
        for inst in self.func.block_insns(block).iter().rev() {
            // Reset has to be before `alloc_inst` not after because
            // available pregs is needed after processing the first
            // instruction in the block during `reload_at_begin`.
            self.reset_available_pregs_and_scratch_regs();
            self.alloc_inst(block, inst)?;
        }
        self.reload_at_begin(block)?;
        trace!("{:?} end\n", block);
        Ok(())
    }

    fn build_debug_info(&mut self) {
        trace!("Building debug location info");
        for &(vreg, start, end, label) in self.func.debug_value_labels() {
            let (point_start, point_end, alloc) = self.vreg_to_live_inst_range[vreg.vreg()];
            if point_start.inst() <= start && end <= point_end.inst().next() {
                self.debug_locations
                    .push((label, point_start, point_end, alloc));
            }
        }
        self.debug_locations.sort_by_key(|loc| loc.0);
    }

    fn run(&mut self) -> Result<(), RegAllocError> {
        debug_assert_eq!(self.func.entry_block().index(), 0);
        for block in (0..self.func.num_blocks()).rev() {
            self.reset_available_pregs_and_scratch_regs();
            self.alloc_block(Block::new(block))?;
        }
        self.edits.edits.reverse();
        self.build_debug_info();
        // Ought to check if there are livein registers
        // then throw an error, but will that be expensive?
        Ok(())
    }
}

fn log_function<F: Function>(func: &F) {
    trace!("Processing a new function");
    for block in 0..func.num_blocks() {
        let block = Block::new(block);
        trace!(
            "Block {:?}. preds: {:?}. succs: {:?}, params: {:?}",
            block,
            func.block_preds(block),
            func.block_succs(block),
            func.block_params(block)
        );
        for inst in func.block_insns(block).iter() {
            let clobbers = func.inst_clobbers(inst);
            trace!(
                "inst{:?}: {:?}. Clobbers: {}",
                inst.index(),
                func.inst_operands(inst),
                clobbers
            );
            if func.is_branch(inst) {
                trace!("Block args: ");
                for (succ_idx, _succ) in func.block_succs(block).iter().enumerate() {
                    trace!(" {:?}", func.branch_blockparams(block, inst, succ_idx));
                }
            }
        }
        trace!("");
    }
}

fn log_output<'a, F: Function>(env: &Env<'a, F>) {
    trace!("Done!");
    use alloc::format;
    let mut v = Vec::new();
    for i in 0..env.func.num_vregs() {
        if env.vreg_spillslots[i].is_valid() {
            v.push((
                format!("{}", VReg::new(i, RegClass::Int)),
                format!("{}", Allocation::stack(env.vreg_spillslots[i])),
            ));
        }
    }
    let mut temp_slots = Vec::new();
    for class in [RegClass::Int, RegClass::Float, RegClass::Vector] {
        for slot in env.temp_spillslots[class].iter() {
            temp_slots.push(format!("{slot}"));
        }
    }
    trace!("VReg spillslots: {:?}", v);
    trace!("Temp spillslots: {:?}", temp_slots);
    trace!("Final edits: {:?}", env.edits.edits);
}

pub fn run<F: Function>(
    func: &F,
    mach_env: &MachineEnv,
    verbose_log: bool,
    enable_ssa_checker: bool,
) -> Result<Output, RegAllocError> {
    if enable_ssa_checker {
        let cfginfo = CFGInfo::new(func)?;
        validate_ssa(func, &cfginfo)?;
    }

    if trace_enabled!() || verbose_log {
        log_function(func);
    }

    let mut env = Env::new(func, mach_env);
    env.run()?;

    if trace_enabled!() || verbose_log {
        log_output(&env);
    }

    Ok(Output {
        edits: env.edits.edits,
        allocs: env.allocs.allocs,
        inst_alloc_offsets: env.allocs.inst_alloc_offsets,
        num_spillslots: env.stack.num_spillslots as usize,
        debug_locations: env.debug_locations,
        stats: env.stats,
    })
}
