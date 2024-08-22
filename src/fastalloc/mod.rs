use crate::{cfg::CFGInfo, ion::Stats, Allocation, RegAllocError};
use crate::{ssa::validate_ssa, Edit, Function, MachineEnv, Output, ProgPoint};
use crate::{
    AllocationKind, Block, Inst, InstPosition, Operand, OperandConstraint, OperandKind, OperandPos,
    PReg, PRegSet, RegClass, SpillSlot, VReg,
};
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::convert::TryInto;
use core::iter::FromIterator;
use core::ops::{Index, IndexMut};

mod bitset;
mod iter;
mod lru;
mod vregset;
use bitset::BitSet;
use iter::*;
use lru::*;
use vregset::VRegSet;

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

fn remove_any_from_pregset(set: &mut PRegSet) -> Option<PReg> {
    if let Some(preg) = set.into_iter().next() {
        set.remove(preg);
        Some(preg)
    } else {
        None
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
    /// The edits to be inserted before the currently processed instruction.
    inst_pre_edits: VecDeque<(ProgPoint, Edit, RegClass)>,
    /// The edits to be inserted after the currently processed instruction.
    inst_post_edits: VecDeque<(ProgPoint, Edit, RegClass)>,
    /// The final output edits.
    edits: VecDeque<(ProgPoint, Edit)>,
    /// Used to determine if a scratch register is needed for an
    /// instruction's moves during the `process_edit` calls.
    inst_needs_scratch_reg: PartedByRegClass<bool>,
    fixed_stack_slots: PRegSet,
    /// All pregs used as the source or destination in an edit
    /// for the current instruction.
    pregs_mentioned_in_edit: PRegSet,
}

impl Edits {
    fn new(fixed_stack_slots: PRegSet, max_operand_len: u32, num_insts: usize) -> Self {
        // Some operands generate edits and some don't.
        // The operands that generate edits add no more than two.
        // Some edits are added due to clobbers, not operands.
        // Anyways, I think this may be a reasonable guess.
        let inst_edits_len_guess = max_operand_len as usize * 2;
        let total_edits_len_guess = inst_edits_len_guess * num_insts;
        Self {
            inst_pre_edits: VecDeque::with_capacity(inst_edits_len_guess),
            inst_post_edits: VecDeque::with_capacity(inst_edits_len_guess),
            edits: VecDeque::with_capacity(total_edits_len_guess),
            fixed_stack_slots,
            inst_needs_scratch_reg: PartedByRegClass {
                items: [false, false, false],
            },
            pregs_mentioned_in_edit: PRegSet::empty(),
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

    fn process_edits(&mut self, scratch_regs: PartedByRegClass<Option<PReg>>) {
        for i in (0..self.inst_post_edits.len()).rev() {
            let (point, edit, class) = self.inst_post_edits[i].clone();
            self.process_edit(point, edit, scratch_regs[class]);
        }
        for i in (0..self.inst_pre_edits.len()).rev() {
            let (point, edit, class) = self.inst_pre_edits[i].clone();
            self.process_edit(point, edit, scratch_regs[class]);
        }
        for class in [RegClass::Int, RegClass::Float, RegClass::Vector] {
            self.inst_needs_scratch_reg[class] = false;
        }
        self.inst_post_edits.clear();
        self.inst_pre_edits.clear();
        self.pregs_mentioned_in_edit = PRegSet::empty();
    }

    fn process_edit(&mut self, point: ProgPoint, edit: Edit, scratch_reg: Option<PReg>) {
        trace!("Processing edit: {:?}", edit);
        let Edit::Move { from, to } = edit;
        if self.is_stack(from) && self.is_stack(to) {
            let scratch_reg = scratch_reg.unwrap();
            trace!(
                "Edit is stack-to-stack, generating two moves with a scratch register {}",
                scratch_reg
            );
            let scratch_alloc = Allocation::reg(scratch_reg);
            trace!(
                "Processed Edit: {:?}",
                (
                    point,
                    Edit::Move {
                        from: scratch_alloc,
                        to,
                    }
                )
            );
            self.edits.push_front((
                point,
                Edit::Move {
                    from: scratch_alloc,
                    to,
                },
            ));
            trace!(
                "Processed Edit: {:?}",
                (
                    point,
                    Edit::Move {
                        from,
                        to: scratch_alloc,
                    }
                )
            );
            self.edits.push_front((
                point,
                Edit::Move {
                    from,
                    to: scratch_alloc,
                },
            ));
        } else {
            trace!("Edit is not stack-to-stack. Adding it directly:");
            trace!("Processed Edit: {:?}", (point, Edit::Move { from, to }));
            self.edits.push_front((point, Edit::Move { from, to }));
        }
    }

    fn add_move_later(
        &mut self,
        inst: Inst,
        from: Allocation,
        to: Allocation,
        class: RegClass,
        pos: InstPosition,
        prepend: bool,
    ) {
        trace!(
            "Recording edit to add later: {:?}",
            (ProgPoint::new(inst, pos), Edit::Move { from, to }, class)
        );
        if let Some(preg) = from.as_reg() {
            self.pregs_mentioned_in_edit.add(preg);
        }
        if let Some(preg) = to.as_reg() {
            self.pregs_mentioned_in_edit.add(preg);
        }
        if from == to {
            trace!("Deciding not to record the edit, since the source and dest are the same");
            return;
        }
        if self.is_stack(from) && self.is_stack(to) {
            self.inst_needs_scratch_reg[class] = true;
        }
        let target_edits = match pos {
            InstPosition::After => &mut self.inst_post_edits,
            InstPosition::Before => &mut self.inst_pre_edits,
        };
        if prepend {
            target_edits.push_front((ProgPoint::new(inst, pos), Edit::Move { from, to }, class));
        } else {
            target_edits.push_back((ProgPoint::new(inst, pos), Edit::Move { from, to }, class));
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
        write!(
            f,
            "{{ early: {}, late: {} }}",
            self.items[0], self.items[1]
        )
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
    /// Allocatable free physical registers for classes Int, Float, and Vector, respectively.
    freepregs: PartedByRegClass<PRegSet>,
    /// Least-recently-used caches for register classes Int, Float, and Vector, respectively.
    lrus: Lrus,
    /// `vreg_in_preg[i]` is the virtual register currently in the physical register
    /// with index `i`.
    vreg_in_preg: Vec<VReg>,
    /// For parallel moves from branch args to block param spillslots.
    temp_spillslots: PartedByRegClass<Vec<SpillSlot>>,
    /// Used to keep track of which used vregs are seen for the first time
    /// in the instruction, that is, if the vregs live past the current instruction.
    /// This is used to determine whether or not reused operands
    /// for reuse-input constraints should be restored after an instruction.
    /// It's also used to determine if the an early operand can reuse a freed def operand's
    /// allocation. And it's also used to determine the edits to be inserted when
    /// allocating a use operand.
    vregs_first_seen_in_curr_inst: BitSet,
    /// Used to keep track of which vregs have been allocated in the current instruction.
    /// This is used to determine which edits to insert when allocating a use operand.
    vregs_allocd_in_curr_inst: BitSet,
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
    dedicated_scratch_regs: PartedByRegClass<Option<PReg>>,
    stack: Stack<'a, F>,

    fixed_stack_slots: PRegSet,

    // Output.
    allocs: Allocs,
    edits: Edits,
    stats: Stats,
}

impl<'a, F: Function> Env<'a, F> {
    fn new(func: &'a F, env: &'a MachineEnv) -> Self {
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
        use alloc::vec;
        trace!("{:?}", env);
        let (allocs, max_operand_len) = Allocs::new(func);
        let fixed_stack_slots = PRegSet::from_iter(env.fixed_stack_slots.iter().cloned());
        Self {
            func,
            allocatable_regs,
            vreg_allocs: vec![Allocation::none(); func.num_vregs()],
            vreg_spillslots: vec![SpillSlot::invalid(); func.num_vregs()],
            live_vregs: VRegSet::with_capacity(func.num_vregs()),
            freepregs: PartedByRegClass {
                items: [
                    PRegSet::from_iter(regs[0].iter().cloned()),
                    PRegSet::from_iter(regs[1].iter().cloned()),
                    PRegSet::from_iter(regs[2].iter().cloned()),
                ],
            },
            lrus: Lrus::new(&regs[0], &regs[1], &regs[2]),
            vreg_in_preg: vec![VReg::invalid(); PReg::NUM_INDEX],
            stack: Stack::new(func),
            fixed_stack_slots,
            temp_spillslots: PartedByRegClass {
                items: [
                    Vec::with_capacity(func.num_vregs()),
                    Vec::with_capacity(func.num_vregs()),
                    Vec::with_capacity(func.num_vregs()),
                ],
            },
            vregs_allocd_in_curr_inst: BitSet::with_capacity(func.num_vregs()),
            vregs_first_seen_in_curr_inst: BitSet::with_capacity(func.num_vregs()),
            reused_input_to_reuse_op: vec![usize::MAX; max_operand_len as usize],
            dedicated_scratch_regs: PartedByRegClass {
                items: [
                    env.scratch_by_class[0],
                    env.scratch_by_class[1],
                    env.scratch_by_class[2],
                ],
            },
            init_available_pregs,
            available_pregs: PartedByOperandPos {
                items: [
                    init_available_pregs,
                    init_available_pregs,
                ]
            },
            allocs,
            edits: Edits::new(fixed_stack_slots, max_operand_len, func.num_insts()),
            stats: Stats::default(),
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

    fn reset_available_pregs(&mut self) {
        trace!("Resetting the available pregs");
        self.available_pregs = PartedByOperandPos {
            items: [self.init_available_pregs, self.init_available_pregs]
        };
    }

    /// The scratch registers needed for processing the edits generated
    /// during a `reload_at_begin` call.
    ///
    /// This function is only called when all instructions in a block have
    /// already been processed. The only edits being processed will be for the
    /// ones to move a liveout vreg or block param from its spillslot to its
    /// expected allocation.
    fn get_scratch_regs_for_reloading(
        &self,
        inst_needs_scratch_reg: PartedByRegClass<bool>,
        avail_pregs: PRegSet,
    ) -> Result<PartedByRegClass<Option<PReg>>, RegAllocError> {
        trace!("Getting scratch registers for reload_at_begin");
        let mut scratch_regs = PartedByRegClass {
            items: [None, None, None],
        };
        for class in [RegClass::Int, RegClass::Float, RegClass::Vector] {
            if inst_needs_scratch_reg[class] {
                trace!("{:?} class needs a scratch register", class);
                if self.dedicated_scratch_regs[class].is_some() {
                    trace!("Using the dedicated scratch register for class {:?}", class);
                    scratch_regs[class] = self.dedicated_scratch_regs[class];
                } else {
                    trace!("No dedicated scratch register for class {:?}. Using the last free register", class);
                    scratch_regs[class] = self.lrus[class].last_satisfying(|preg| {
                        avail_pregs.contains(preg)
                        && self.vreg_in_preg[preg.index()] == VReg::invalid()
                    });
                    if scratch_regs[class].is_none() {
                        trace!("Unable to find a scratch register for class {class:?}");
                        return Err(RegAllocError::TooManyLiveRegs);
                    }
                }
            }
        }
        Ok(scratch_regs)
    }

    /// The scratch registers needed for processing edits generated while
    /// processing instructions.
    fn get_scratch_regs(
        &mut self,
        inst: Inst,
        inst_needs_scratch_reg: PartedByRegClass<bool>,
        avail_pregs: PRegSet,
    ) -> Result<PartedByRegClass<Option<PReg>>, RegAllocError> {
        trace!("Getting scratch registers for instruction {:?}", inst);
        let mut scratch_regs = PartedByRegClass {
            items: [None, None, None],
        };
        for class in [RegClass::Int, RegClass::Float, RegClass::Vector] {
            if inst_needs_scratch_reg[class] {
                trace!("{:?} class needs a scratch register", class);
                if let Some(reg) = self.dedicated_scratch_regs[class] {
                    trace!("Using the dedicated scratch register for class {:?}", class);
                    scratch_regs[class] = Some(reg);
                } else {
                    trace!("class {:?} has no dedicated scratch register", class);
                    if let Some(preg) = self.lrus[class].last_satisfying(|preg| {
                        avail_pregs.contains(preg)
                        // Consider a scenario:
                        //
                        // 1. use v0 (fixed: stack0), use v1 (fixed: p1)
                        // 2. use v1 (fixed: p2), use v0 (fixed: stack1)
                        //
                        // In the above, during the processing of inst 1, v0 is already
                        // in stack1 and v1 is already in p2.
                        // An edit will be inserted after the instruction to move
                        // from stack0 to stack1. Afterwards, when the v1 operand
                        // is being processed, `vreg_in_preg[p2]` will be set to `VReg::invalid`.
                        // The end result is:
                        //
                        // move from p1 to stack_v1 // Save v1 (inserted by `process_operand_allocation`)
                        // 1. use v0 (fixed: stack0), use v1 (fixed: p1)
                        // move from stack_v1 to p2 // Restore v1 (inserted by `process_operand_allocation`)
                        // move from stack0 to scratch // scratch could be p2
                        // move from scratch to stack1
                        // 2. use v1 (fixed: p2), use v0 (fixed: stack1)
                        //
                        // p2 could be used as a scratch register because it will be
                        // marked available in the `avail_regs` since it's not allocated
                        // to any operand in inst 1.
                        // If p2 is used as the scratch register, then v0 will
                        // be used in the place of v1 in inst 2, which is incorrect.
                        // To avoid this scenario, all pregs either used as the source
                        // or destination in the instruction are avoided.
                        && !self.edits.pregs_mentioned_in_edit.contains(preg)
                    }) {
                        if self.vreg_in_preg[preg.index()] != VReg::invalid() {
                            // A register used as scratch may be used for instructions
                            // added before or after the instruction.
                            //
                            // This will work because `preg` hasn't been allocated in the current
                            // instruction (`avail_pregs` only contains such registers).
                            self.evict_vreg_in_preg(inst, preg);
                            self.vreg_in_preg[preg.index()] = VReg::invalid();
                        }
                        trace!("The scratch register: {preg}");
                        scratch_regs[class] = Some(preg);
                    } else {
                        trace!("Unable to find a scratch register for class {class:?}");
                        return Err(RegAllocError::TooManyLiveRegs);
                    }
                }
            } else {
                trace!("{:?} class does not need a scratch register", class);
            }
        }
        Ok(scratch_regs)
    }

    fn allocd_within_constraint(&self, inst: Inst, op: Operand, fixed_spillslot: Option<SpillSlot>) -> bool {
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
                        // In the case where the vreg is used multiple times:
                        // use v0 (reg), use v0 (reg), use v0 (reg)
                        // During processing of the first operand, the allocated
                        // register is removed from the available regs list, but the
                        // vreg is added to the `vregs_allocd_in_curr_inst` set.
                        // This check is necessary to ensure
                        // that multiple pregs aren't assigned in these situations when
                        // they aren't needed.
                        self.vreg_in_preg[preg.index()] == op.vreg()
                            && self.vregs_allocd_in_curr_inst.contains(op.vreg().vreg())
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
                        self.vreg_in_preg[preg.index()] == op.vreg()
                            && self.vregs_allocd_in_curr_inst.contains(op.vreg().vreg())
                    } else {
                        true
                    }
                } else {
                    false
                }
            }
            OperandConstraint::Stack => if let Some(slot) = fixed_spillslot {
                alloc == Allocation::stack(slot)
            } else {
                self.is_stack(alloc)
            },
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
        self.edits.add_move_later(
            inst,
            self.vreg_allocs[evicted_vreg.vreg()],
            Allocation::reg(preg),
            evicted_vreg.class(),
            InstPosition::After,
            false,
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
    fn alloc_reg_for_operand(&mut self, inst: Inst, op: Operand) -> Result<(), RegAllocError> {
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
            trace!("Failed to find an available {:?} register in the LRU for operand {op}", op.class());
            return Err(RegAllocError::TooManyLiveRegs);
        };
        if self.vreg_in_preg[preg.index()] != VReg::invalid() {
            self.evict_vreg_in_preg(inst, preg);
        }
        trace!("The allocated register for vreg {}: {}", op.vreg(), preg);
        self.lrus[op.class()].poke(preg);
        self.vreg_allocs[op.vreg().vreg()] = Allocation::reg(preg);
        self.vreg_in_preg[preg.index()] = op.vreg();
        self.available_pregs[op.pos()].remove(preg);
        match (op.pos(), op.kind()) {
            (OperandPos::Late, OperandKind::Use) => {
                self.available_pregs[OperandPos::Early].remove(preg);
            }
            (OperandPos::Early, OperandKind::Def) => {
                self.available_pregs[OperandPos::Late].remove(preg);
            }
            _ => ()
        };
        Ok(())
    }

    fn alloc_fixed_reg_for_operand(
        &mut self,
        inst: Inst,
        op: Operand,
        preg: PReg,
    ) {
        trace!("The fixed preg: {} for operand {}", preg, op);

        // It is an error for a fixed register clobber to be used for a defined vreg
        // that outlives the instruction, because it will be impossible to restore it.
        // But checking for that will be expensive?

        if self.vreg_in_preg[preg.index()] != VReg::invalid() {
            // Something is already in that register. Evict it.
            self.evict_vreg_in_preg(inst, preg);
        }
        if self.allocatable_regs.contains(preg) {
            self.lrus[op.class()].poke(preg);
        }
        self.vreg_allocs[op.vreg().vreg()] = Allocation::reg(preg);
        self.vreg_in_preg[preg.index()] = op.vreg();
        trace!("vreg {} is now in preg {}", op.vreg(), preg);
    }

    /// Allocates for the operand `op` with index `op_idx` into the
    /// vector of instruction `inst`'s operands.
    fn alloc_operand(
        &mut self,
        inst: Inst,
        op: Operand,
        op_idx: usize,
        fixed_spillslot: Option<SpillSlot>,
    ) -> Result<(), RegAllocError> {
        match op.constraint() {
            OperandConstraint::Any => {
                self.alloc_reg_for_operand(inst, op)?;
            }
            OperandConstraint::Reg => {
                self.alloc_reg_for_operand(inst, op)?;
            }
            OperandConstraint::Stack => {
                let slot = if let Some(spillslot) = fixed_spillslot {
                    spillslot
                } else {
                    if self.vreg_spillslots[op.vreg().vreg()].is_invalid() {
                        self.vreg_spillslots[op.vreg().vreg()] = self.stack.allocstack(&op.vreg());
                    }
                    self.vreg_spillslots[op.vreg().vreg()]
                };
                self.vreg_allocs[op.vreg().vreg()] = Allocation::stack(slot);
            }
            OperandConstraint::FixedReg(preg) => {
                self.alloc_fixed_reg_for_operand(inst, op, preg);
            }
            OperandConstraint::Reuse(_) => {
                // This is handled elsewhere.
                unreachable!();
            }
        }
        self.allocs[(inst.index(), op_idx)] = self.vreg_allocs[op.vreg().vreg()];
        Ok(())
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
        fixed_spillslot: Option<SpillSlot>,
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
        if !self.allocd_within_constraint(inst, op, fixed_spillslot) {
            trace!("{op} isn't allocated within constraints.");
            let prev_alloc = self.vreg_allocs[op.vreg().vreg()];
            if prev_alloc.is_none() {
                self.vregs_first_seen_in_curr_inst.insert(op.vreg().vreg());
                self.live_vregs.insert(op.vreg());
            }
            self.alloc_operand(inst, op, op_idx, fixed_spillslot)?;
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
                    self.edits.add_move_later(
                        inst,
                        self.vreg_allocs[op.vreg().vreg()],
                        prev_alloc,
                        op.class(),
                        InstPosition::After,
                        true,
                    );
                } else {
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

                    if !self.vregs_allocd_in_curr_inst.contains(op.vreg().vreg())
                        // Don't restore after the instruction if it doesn't live past
                        // this instruction.
                        && !self.vregs_first_seen_in_curr_inst.contains(op.vreg().vreg())
                    {
                        if self.vreg_spillslots[op.vreg().vreg()].is_invalid() {
                            self.vreg_spillslots[op.vreg().vreg()] =
                                self.stack.allocstack(&op.vreg());
                        }
                        let op_spillslot =
                            Allocation::stack(self.vreg_spillslots[op.vreg().vreg()]);
                        self.edits.add_move_later(
                            inst,
                            self.vreg_allocs[op.vreg().vreg()],
                            op_spillslot,
                            op.class(),
                            InstPosition::Before,
                            false,
                        );
                        self.edits.add_move_later(
                            inst,
                            op_spillslot,
                            prev_alloc,
                            op.class(),
                            InstPosition::After,
                            true,
                        );
                    } else {
                        self.edits.add_move_later(
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
                    // Free the previous allocation so that it can be reused.
                    let preg = prev_alloc.as_reg().unwrap();
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
                if self.func.inst_clobbers(inst).contains(preg) {
                    // It is possible for the first use of a vreg in an instruction
                    // to be some clobber p0 and the expected location of that vreg
                    // after the instruction is also p0:
                    //
                    // 1. use v0 (fixed: p0), use v0 (fixed: p1). clobbers: [p0]
                    // 2. use v0 (fixed: p0)
                    //
                    // When the second use of v0 is encountered in inst 1, a save and restore is
                    // not inserted because it's not the first use of v0 in the instruction. Instead,
                    // a single edit to move from p1 to p0 is inserted before the instruction:
                    //
                    // move from p1 to p0
                    // 1. use v0 (fixed: p0), use v0 (fixed: p1). clobbers: [p0]
                    // 2. use v0 (fixed: p0)
                    //
                    // To avoid this scenario, a save and restore is added here.
                    if !self.vregs_allocd_in_curr_inst.contains(op.vreg().vreg())
                        && !self
                            .vregs_first_seen_in_curr_inst
                            .contains(op.vreg().vreg())
                    {
                        if self.vreg_spillslots[op.vreg().vreg()].is_invalid() {
                            self.vreg_spillslots[op.vreg().vreg()] =
                                self.stack.allocstack(&op.vreg());
                        }
                        let op_spillslot =
                            Allocation::stack(self.vreg_spillslots[op.vreg().vreg()]);
                        self.edits.add_move_later(
                            inst,
                            self.vreg_allocs[op.vreg().vreg()],
                            op_spillslot,
                            op.class(),
                            InstPosition::Before,
                            false,
                        );
                        self.edits.add_move_later(
                            inst,
                            op_spillslot,
                            self.vreg_allocs[op.vreg().vreg()],
                            op.class(),
                            InstPosition::After,
                            true,
                        );
                    }
                }
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
                    _ => ()
                };
            }
            trace!(
                "Allocation for instruction {:?} and operand {}: {}",
                inst,
                op,
                self.allocs[(inst.index(), op_idx)]
            );
        }
        trace!("Late available regs: {}", self.available_pregs[OperandPos::Late]);
        trace!("Early available regs: {}", self.available_pregs[OperandPos::Early]);
        self.vregs_allocd_in_curr_inst.insert(op.vreg().vreg());
        Ok(())
    }

    fn alloc_slots_for_block_params(&mut self, succ: Block) {
        for vreg in self.func.block_params(succ) {
            if self.vreg_spillslots[vreg.vreg()].is_invalid() {
                self.vreg_spillslots[vreg.vreg()] = self.stack.allocstack(vreg);
                trace!(
                    "Block param {} is in {}",
                    vreg,
                    Allocation::stack(self.vreg_spillslots[vreg.vreg()])
                );
            }
        }
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

    fn save_and_restore_clobbered_registers(&mut self, inst: Inst) {
        trace!("Adding save and restore edits for vregs in clobbered registers");
        for clobbered_preg in self.func.inst_clobbers(inst) {
            // If the instruction clobbers a register holding a live vreg,
            // insert edits to save the live reg and restore it
            // after the instruction.
            // For example:
            //
            // 1. def v2
            // 2. use v0, use v1 - clobbers p0
            // 3. use v2 (fixed: p0)
            //
            // In the above, v2 is assigned to p0 first. During the processing of inst 2,
            // p0 is clobbered, so v2 is no longer in it and p0 no longer contains v2 at inst 2.
            // p0 is allocated to the v2 def operand in inst 1. The flow ends up wrong because of
            // the clobbering.
            //
            //
            // It is also possible for a clobbered register to be allocated to an operand
            // in an instruction. No edits need to be inserted here because
            // `process_operand_allocation` has already done all the insertions.

            let vreg = self.vreg_in_preg[clobbered_preg.index()];
            if vreg != VReg::invalid() {
                let vreg_isnt_mentioned_in_curr_inst =
                    !self.vregs_allocd_in_curr_inst.contains(vreg.vreg());
                if vreg_isnt_mentioned_in_curr_inst {
                    trace!("Adding save and restore edits for {}", vreg);
                    let preg_alloc = Allocation::reg(clobbered_preg);
                    let slot = if self.vreg_spillslots[vreg.vreg()].is_valid() {
                        self.vreg_spillslots[vreg.vreg()]
                    } else {
                        self.vreg_spillslots[vreg.vreg()] = self.stack.allocstack(&vreg);
                        self.vreg_spillslots[vreg.vreg()]
                    };
                    let slot_alloc = Allocation::stack(slot);
                    self.edits.add_move_later(
                        inst,
                        preg_alloc,
                        slot_alloc,
                        vreg.class(),
                        InstPosition::Before,
                        true,
                    );
                    self.edits.add_move_later(
                        inst,
                        slot_alloc,
                        preg_alloc,
                        vreg.class(),
                        InstPosition::After,
                        false,
                    );
                }
            }
        }
        trace!("Done adding edits for clobbered registers");
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

        for succ in self.func.block_succs(block).iter() {
            self.alloc_slots_for_block_params(*succ);
        }

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
                if self.temp_spillslots[vreg.class()].len() == next_temp_idx[vreg.class()] {
                    let newslot = self.stack.allocstack(vreg);
                    self.temp_spillslots[vreg.class()].push(newslot);
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
                // Assuming that vregs defined in the current branch instruction can't be
                // used as branch args for successors, else inserting the moves before, instead
                // of after will be wrong. But the edits are inserted before because the fuzzer
                // doesn't recognize moves inserted after branch instructions.
                self.edits.add_move_later(
                    inst,
                    vreg_spill,
                    temp,
                    vreg.class(),
                    InstPosition::Before,
                    false,
                );
            }
        }

        reset_temp_idx(&mut next_temp_idx);

        for (succ_idx, succ) in self.func.block_succs(block).iter().enumerate() {
            let succ_params = self.func.block_params(*succ);

            // Move from temporaries to block param spillslots.
            for (pos, vreg) in self
                .func
                .branch_blockparams(block, inst, succ_idx)
                .iter()
                .enumerate()
            {
                let succ_param_vreg = succ_params[pos];
                let param_alloc = Allocation::stack(self.vreg_spillslots[succ_param_vreg.vreg()]);
                let temp_slot = self.temp_spillslots[vreg.class()][next_temp_idx[vreg.class()]];
                let temp = Allocation::stack(temp_slot);
                next_temp_idx[vreg.class()] += 1;
                trace!(" --- Placing branch arg {} in {}", vreg, temp);
                trace!(
                    "{} which is now in {} inserting move to {}",
                    vreg,
                    temp,
                    param_alloc
                );
                self.edits.add_move_later(
                    inst,
                    temp,
                    param_alloc,
                    vreg.class(),
                    InstPosition::Before,
                    false,
                );

                // All branch arguments should be in their spillslots at the end of the function.
                //
                // The invariants posed by `vregs_first_seen_in_curr_inst` and
                // `vregs_allocd_in_curr_inst` must be maintained in order to
                // insert edits in the correct order when vregs used as branch args
                // are also used as operands.
                if self.vreg_allocs[vreg.vreg()].is_none() {
                    self.vregs_first_seen_in_curr_inst.insert(vreg.vreg());
                    self.live_vregs.insert(*vreg);
                    self.vregs_allocd_in_curr_inst.insert(vreg.vreg());
                }
                self.vreg_allocs[vreg.vreg()] =
                    Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
            }
        }
    }

    fn alloc_inst(&mut self, block: Block, inst: Inst) -> Result<(), RegAllocError> {
        trace!("Allocating instruction {:?}", inst);
        if self.func.requires_refs_on_stack(inst) && !self.func.reftype_vregs().is_empty() {
            panic!("Safepoint instructions aren't supported");
        }
        if self.func.is_branch(inst) {
            self.process_branch(block, inst);
        }
        let operands = Operands::new(self.func.inst_operands(inst));
        let clobbers = self.func.inst_clobbers(inst);

        for (op_idx, op) in operands.reuse() {
            trace!("Initializing reused_input_to_reuse_op");
            let OperandConstraint::Reuse(reused_idx) = op.constraint() else {
                unreachable!()
            };
            self.reused_input_to_reuse_op[reused_idx] = op_idx;
        }
        for (op_idx, op) in operands.fixed() {
            let OperandConstraint::FixedReg(preg) = op.constraint() else {
                unreachable!();
            };
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
        }
        self.remove_clobbers_from_available_pregs(clobbers);
        for (op_idx, op) in operands.def_ops() {
            trace!("Allocating def operands {op}");
            if let OperandConstraint::Reuse(reused_idx) = op.constraint() {
                let reused_op = operands[reused_idx];
                let new_reuse_op = Operand::new(
                    op.vreg(),
                    reused_op.constraint(),
                    op.kind(),
                    op.pos(),
                );
                trace!("allocating reuse op {op} as {new_reuse_op}");
                self.process_operand_allocation(inst, new_reuse_op, op_idx, None)?;
                if let Some(preg) = self.allocs[(inst.index(), op_idx)].as_reg() {
                    // The reused input is going to be processed as a fixed register for this
                    // preg.
                    self.available_pregs[OperandPos::Early].remove(preg);
                }
            } else {
                self.process_operand_allocation(inst, op, op_idx, None)?;
            }
            self.freealloc(op.vreg());
        }
        for (op_idx, op) in operands.use_ops() {
            trace!("Allocating use operand {op}");
            if self.reused_input_to_reuse_op[op_idx] != usize::MAX {
                let reuse_op_idx = self.reused_input_to_reuse_op[op_idx];
                let reuse_op_alloc = self.allocs[(inst.index(), reuse_op_idx)];
                let new_reused_input_constraint;
                let mut fixed_slot = None;
                if let Some(preg) = reuse_op_alloc.as_reg() {
                    new_reused_input_constraint = OperandConstraint::FixedReg(preg);
                } else {
                    new_reused_input_constraint = OperandConstraint::Stack;
                    fixed_slot = Some(reuse_op_alloc.as_stack().unwrap());
                }
                let new_reused_input = Operand::new(
                    op.vreg(),
                    new_reused_input_constraint,
                    op.kind(),
                    op.pos(),
                );
                trace!("Allocating reused input {op} as {new_reused_input}, (fixed spillslot: {fixed_slot:?})");
                self.process_operand_allocation(inst, new_reused_input, op_idx, fixed_slot)?;
            } else {
                self.process_operand_allocation(inst, op, op_idx, None)?;
            }
        }
        self.save_and_restore_clobbered_registers(inst);
        let mut avail_for_scratch = self.available_pregs[OperandPos::Early];
        avail_for_scratch.intersect_from(self.available_pregs[OperandPos::Late]);
        let scratch_regs =
            self.get_scratch_regs(inst, self.edits.inst_needs_scratch_reg.clone(), avail_for_scratch)?;
        self.edits.process_edits(scratch_regs);
        self.vregs_first_seen_in_curr_inst.clear();
        self.vregs_allocd_in_curr_inst.clear();
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
        trace!("Available pregs: {}", self.available_pregs[OperandPos::Early]);
        let mut available_regs_for_scratch = self.available_pregs[OperandPos::Early];
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
            self.edits.add_move_later(
                self.func.block_insns(block).first(),
                slot,
                prev_alloc,
                vreg.class(),
                InstPosition::Before,
                true,
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
            self.edits.add_move_later(
                self.func.block_insns(block).first(),
                slot,
                prev_alloc,
                vreg.class(),
                InstPosition::Before,
                true,
            );
        }
        let scratch_regs =
            self.get_scratch_regs_for_reloading(self.edits.inst_needs_scratch_reg.clone(), available_regs_for_scratch)?;
        self.edits.process_edits(scratch_regs);
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
                v.push(format!("{}: {}, ", PReg::from_index(i), self.vreg_in_preg[i]));
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
            self.reset_available_pregs();
            self.alloc_inst(block, inst)?;
        }
        self.reload_at_begin(block)?;
        trace!("{:?} end\n", block);
        Ok(())
    }

    fn run(&mut self) -> Result<(), RegAllocError> {
        debug_assert_eq!(self.func.entry_block().index(), 0);
        for block in (0..self.func.num_blocks()).rev() {
            self.reset_available_pregs();
            self.alloc_block(Block::new(block))?;
        }
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
    enable_annotations: bool,
    enable_ssa_checker: bool,
) -> Result<Output, RegAllocError> {
    if enable_ssa_checker {
        let cfginfo = CFGInfo::new(func)?;
        validate_ssa(func, &cfginfo)?;
    }

    if trace_enabled!() {
        log_function(func);
    }

    let mut env = Env::new(func, mach_env);
    env.run()?;

    if trace_enabled!() {
        log_output(&env);
    }

    Ok(Output {
        edits: env.edits.edits.make_contiguous().to_vec(),
        allocs: env.allocs.allocs,
        inst_alloc_offsets: env.allocs.inst_alloc_offsets,
        num_spillslots: env.stack.num_spillslots as usize,
        // TODO: Handle debug locations.
        debug_locations: Vec::new(),
        safepoint_slots: Vec::new(),
        stats: env.stats,
    })
}
