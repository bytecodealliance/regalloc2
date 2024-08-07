use core::convert::TryInto;
use core::iter::FromIterator;
use core::ops::{Index, IndexMut};
use crate::{AllocationKind, Block, Inst, InstPosition, Operand, OperandConstraint, OperandKind, OperandPos, PReg, PRegSet, RegClass, SpillSlot, VReg};
use crate::{Function, MachineEnv, ssa::validate_ssa, ProgPoint, Edit, Output};
use crate::{cfg::CFGInfo, RegAllocError, Allocation, ion::Stats};
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use hashbrown::{HashSet, HashMap};

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
    fn new<F: Function>(func: &F) -> Self {
        let operand_no_guess = func.num_insts() * 3;
        let mut allocs = Vec::with_capacity(operand_no_guess);
        let mut inst_alloc_offsets = Vec::with_capacity(operand_no_guess);
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

fn remove_any_from_pregset(set: &mut PRegSet) -> Option<PReg> {
    if let Some(preg) = set.into_iter().next() {
        set.remove(preg);
        Some(preg)
    } else {
        None
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
    freepregs: PartedByRegClass<PRegSet>,
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
    free_after_curr_inst: PartedByRegClass<PRegSet>,
    /// The virtual registers of use operands that have been allocated in the current instruction
    /// and for which edits had to be inserted to save and restore them because their constraint
    /// doesn't allow the allocation they are expected to be in after the instruction.
    /// 
    /// This needs to be kept track of to generate the correct moves in the case where a
    /// single virtual register is used multiple times in a single instruction with
    /// different constraints.
    use_vregs_saved_and_restored_in_curr_inst: HashSet<VReg>,
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
    freed_def_pregs: PartedByRegClass<PRegSet>,
    /// Used to keep track of which used vregs are seen for the first time
    /// in the instruction, that is, if the vregs live past the current instruction.
    /// This is used to determine whether or not reused operands
    /// for reuse-input constraints should be restored after an instruction.
    /// It's also used to determine if the an early operand can reuse a freed def operand's
    /// allocation. And it's also used to determine the edits to be inserted when
    /// allocating a use operand.
    vregs_first_seen_in_curr_inst: HashSet<VReg>,
    /// Used to keep track of which vregs have been allocated in the current instruction.
    /// This is used to determine which edits to insert when allocating a use operand.
    vregs_allocd_in_curr_inst: HashSet<VReg>,
    /// All the liveout vregs encountered during allocation.
    /// When allocation is completed, this contains all the liveout vregs in
    /// the function.
    /// This is used to build the stackmap after allocation is complete.
    liveout_vregs: HashSet<VReg>,
    /// Used to determine if a scratch register is needed for an
    /// instruction's moves during the `process_edit` calls.
    inst_needs_scratch_reg: PartedByRegClass<bool>,
    /// The operand indexes of the inputs reused for reuse operands.
    /// This is used to avoid assigning a freed def preg to a reused input.
    reused_inputs_in_curr_inst: Vec<usize>,
    /// The vregs defined or used in the current instruction.
    vregs_in_curr_inst: HashSet<VReg>,
    allocatable_regs: PRegSet,
    dedicated_scratch_regs: PartedByRegClass<Option<PReg>>,

    fixed_stack_slots: Vec<PReg>,

    // Output.
    allocs: Allocs,
    edits: VecDeque<(ProgPoint, Edit)>,
    safepoint_slots: Vec<(ProgPoint, Allocation)>,
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
            allocatable_regs: PRegSet::from(env),
            vreg_allocs: vec![Allocation::none(); func.num_vregs()],
            vreg_spillslots: vec![SpillSlot::invalid(); func.num_vregs()],
            live_vregs: HashSet::with_capacity(func.num_vregs()),
            freepregs: PartedByRegClass {
                items: [
                    PRegSet::from_iter(regs[0].iter().cloned()),
                    PRegSet::from_iter(regs[1].iter().cloned()),
                    PRegSet::from_iter(regs[2].iter().cloned()),
                ]
            },
            lrus: Lrus::new(
                &regs[0],
                &regs[1],
                &regs[2]
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
            free_after_curr_inst: PartedByRegClass { items: [PRegSet::empty(), PRegSet::empty(), PRegSet::empty()] },
            vregs_allocd_in_curr_inst: HashSet::new(),
            use_vregs_saved_and_restored_in_curr_inst: HashSet::new(),
            freed_def_pregs: PartedByRegClass { items: [PRegSet::empty(), PRegSet::empty(), PRegSet::empty()] },
            vregs_first_seen_in_curr_inst: HashSet::new(),
            liveout_vregs: HashSet::new(),
            inst_needs_scratch_reg: PartedByRegClass { items: [false, false, false] },
            reused_inputs_in_curr_inst: Vec::new(),
            vregs_in_curr_inst: HashSet::new(),
            dedicated_scratch_regs: PartedByRegClass { items: [
                env.scratch_by_class[0],
                env.scratch_by_class[1],
                env.scratch_by_class[2],
            ] },
            allocs: Allocs::new(func),
            edits: VecDeque::new(),
            safepoint_slots: Vec::new(),
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

    fn add_freed_regs_to_freelist(&mut self) {
        for class in [RegClass::Int, RegClass::Float, RegClass::Vector] {
            for preg in self.freed_def_pregs[class] {
                self.lrus[class].append(preg.hw_enc());
            }
            self.freepregs[class].union_from(self.freed_def_pregs[class]);
            self.freed_def_pregs[class] = PRegSet::empty();
            
            for preg in self.free_after_curr_inst[class] {
                self.lrus[preg.class()].append(preg.hw_enc());
            }
            self.freepregs[class].union_from(self.free_after_curr_inst[class]);
            self.free_after_curr_inst[class] = PRegSet::empty();
        }
    }

    /// The scratch registers needed for processing the edits generated
    /// during a `reload_at_begin` call.
    ///
    /// This function is only called when all instructions in a block have
    /// already been processed. The only edits being processed will be for the
    /// ones to move a liveout vreg or block param from its spillslot to its
    /// expected allocation.
    fn get_scratch_regs_for_reloading(&self) -> PartedByRegClass<Option<PReg>> {
        trace!("Getting scratch registers for reload_at_begin");
        let mut scratch_regs = PartedByRegClass{ items: [None, None, None] };
        for class in [RegClass::Int, RegClass::Float, RegClass::Vector] {
            if self.inst_needs_scratch_reg[class] {
                trace!("{:?} class needs a scratch register", class);
                if self.dedicated_scratch_regs[class].is_some() {
                    trace!("Using the dedicated scratch register for class {:?}", class);
                    scratch_regs[class] = self.dedicated_scratch_regs[class];
                } else {
                    trace!("No dedicated scratch register for class {:?}. Using the last free register", class);
                    scratch_regs[class] = Some(self.freepregs[class]
                        .into_iter()
                        .next()
                        .expect("Allocation impossible?"));
                }
            }
        }
        scratch_regs
    }

    /// The scratch registers needed for processing edits generated while
    /// processing instructions.
    fn get_scratch_regs(&mut self, inst: Inst) -> PartedByRegClass<Option<PReg>> {
        trace!("Getting scratch registers for instruction {:?}", inst);
        let mut scratch_regs = PartedByRegClass { items: [None, None, None] };
        for class in [RegClass::Int, RegClass::Float, RegClass::Vector] {
            if self.inst_needs_scratch_reg[class] {
                trace!("{:?} class needs a scratch register", class);
                if let Some(reg) = self.dedicated_scratch_regs[class] {
                    trace!("Using the dedicated scratch register for class {:?}", class);
                    scratch_regs[class] = Some(reg);
                } else {
                    trace!("class {:?} has no dedicated scratch register", class);
                    let reg = if let Some(preg) = self.freepregs[class].into_iter().next() {
                        trace!("Using the last free {:?} register for scratch", class);
                        preg
                    } else {
                        trace!("No free {:?} registers. Evicting a register", class);
                        self.evict_any_reg(inst, class)
                    };
                    scratch_regs[class] = Some(reg);
                }
            } else {
                trace!("{:?} class does not need a scratch register", class);
            }
        }
        scratch_regs
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
    }

    fn process_edit(&mut self, point: ProgPoint, edit: Edit, scratch_reg: Option<PReg>) {
        trace!("Processing edit: {:?}", edit);
        let Edit::Move { from, to } = edit;
        if self.is_stack(from) && self.is_stack(to) {
            let scratch_reg = scratch_reg.unwrap();
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
        trace!("Recording edit to add later: {:?}", (ProgPoint::new(inst, pos), Edit::Move {
            from,
            to
        }, class));
        if from == to {
            trace!("Deciding not to record the edit, since the source and dest are the same");
            return;
        }
        if self.is_stack(from) && self.is_stack(to) {
            self.inst_needs_scratch_reg[class] = true;
        }
        let target_edits = match pos {
            InstPosition::After => &mut self.inst_post_edits,
            InstPosition::Before => &mut self.inst_pre_edits
        };
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

    fn move_after_inst(&mut self, inst: Inst, vreg: VReg, to: Allocation) {
        self.add_move_later(inst, self.vreg_allocs[vreg.vreg()], to, vreg.class(), InstPosition::After, false);
    }

    fn move_before_inst(&mut self, inst: Inst, vreg: VReg, to: Allocation) {
        self.add_move_later(inst, self.vreg_allocs[vreg.vreg()], to, vreg.class(), InstPosition::Before, false);
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
                alloc.is_some() && !alloc_is_clobber
            },
            OperandConstraint::Reg => {
                let alloc_is_reg = alloc.is_reg() && alloc.as_reg().unwrap().class() == op.class()
                    && !self.is_stack(alloc);
                alloc_is_reg && !alloc_is_clobber
            },
            OperandConstraint::Stack => self.is_stack(alloc),
            // It is possible for an operand to have a fixed register constraint to
            // a clobber.
            OperandConstraint::FixedReg(preg) => alloc.is_reg() &&
                alloc.as_reg().unwrap() == preg,
            OperandConstraint::Reuse(_) => {
                unreachable!()
            }
        }
    }

    fn evict_vreg_in_preg(&mut self, inst: Inst, preg: PReg) {
        trace!("Removing the vreg in preg {:?} for eviction", preg);
        let evicted_vreg = self.vreg_in_preg[preg.index()];
        trace!("The removed vreg: {:?}", evicted_vreg);
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
        trace!("Evicting a register in evict_any_reg for class {:?}", regclass);
        let preg = self.lrus[regclass].pop();
        trace!("Selected register from lru: {:?}", preg);
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

    fn freealloc(&mut self, vreg: VReg, clobbers: PRegSet) {
        trace!("Freeing vreg {:?}", vreg);
        let alloc = self.vreg_allocs[vreg.vreg()];
        match alloc.kind() {
            AllocationKind::Reg => {
                let preg = alloc.as_reg().unwrap();
                self.vreg_in_preg[preg.index()] = VReg::invalid();
                // If it's a fixed stack slot, then it's not allocatable.
                if !self.is_stack(alloc) {
                    if clobbers.contains(preg) {
                        // For a defined vreg to be restored to the location it's expected to
                        // be in after the instruction, it cannot be allocated to a clobber because that
                        // will make the restoration impossible.
                        // In the case where a reuse operand reuses an input allocated to a clobber,
                        // the defined vreg will be allocated to a clobber
                        // and if the vreg lives past the instruction, restoration will be impossible.
                        // To avoid this, simply make it impossible for a clobber to be allocated to
                        // a vreg with "any" or "any reg" constraints.
                        // By adding it to this list, instead of freed_def_pregs, the only way
                        // a clobber can be newly allocated to a vreg in the instruction is to
                        // use a fixed register constraint.
                        self.free_after_curr_inst[preg.class()].add(preg);
                        // No need to remove the preg from the LRU because clobbers
                        // have already been removed from the LRU.
                    } else {
                        // Added to the freed def pregs list, not the free pregs
                        // list to avoid a def's allocated register being used
                        // as a scratch register.
                        self.freed_def_pregs[vreg.class()].add(preg);
                        // Don't allow this register to be evicted.
                        self.lrus[vreg.class()].remove(preg.hw_enc());
                    }
                }
            }
            AllocationKind::Stack => (),
            AllocationKind::None => panic!("Attempting to free an unallocated operand!")
        }
        self.vreg_allocs[vreg.vreg()] = Allocation::none();
        self.live_vregs.remove(&vreg);
        trace!("{:?} curr alloc is now {:?}", vreg, self.vreg_allocs[vreg.vreg()]);
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
    fn alloc_reg_for_operand(&mut self, inst: Inst, op: Operand, op_idx: usize) {
        trace!("freepregs int: {:?}", self.freepregs[RegClass::Int]);
        trace!("freepregs vector: {:?}", self.freepregs[RegClass::Vector]);
        trace!("freepregs float: {:?}", self.freepregs[RegClass::Float]);
        trace!("freed_def_pregs int: {:?}", self.freed_def_pregs[RegClass::Int]);
        trace!("freed_def_pregs vector: {:?}", self.freed_def_pregs[RegClass::Vector]);
        trace!("freed_def_pregs float: {:?}", self.freed_def_pregs[RegClass::Float]);
        trace!("");
        let mut allocd = false;
        // The only way a freed def preg can be reused for an operand is if
        // the operand uses or defines a vreg in the early phase and the vreg doesn't
        // live past the instruction. If the vreg lives past the instruction, then the
        // defined value will overwrite it.
        if op.pos() == OperandPos::Early && self.vregs_first_seen_in_curr_inst.contains(&op.vreg())
            // A reused input should not have the same allocation as a non-reused def operand.
            // For example:
            // 1. dev v0 (fixed: p0), def v1 (reuse: 2), use v2
            // 2. use v0, use v1
            // In the above, p0 is assigned to v0 in inst 1. Say, after v0 is freed,
            // p0 is re-assigned to v2. Then, when reuse operands are processed, p0
            // will also end up being v1's allocation in inst 1.
            // The end result will be allocating two defined vregs, v0 and v1, to the
            // same allocation p0.
            // To avoid this, don't allow a reused input to have the same allocation as
            // a freed def operand.
            && !self.reused_inputs_in_curr_inst.contains(&op_idx) 
        {
            if let Some(freed_def_preg) = remove_any_from_pregset(&mut self.freed_def_pregs[op.class()]) {
                trace!("Reusing the freed def preg: {:?}", freed_def_preg);
                self.lrus[freed_def_preg.class()].append_and_poke(freed_def_preg);
                self.vreg_allocs[op.vreg().vreg()] = Allocation::reg(freed_def_preg);
                self.vreg_in_preg[freed_def_preg.index()] = op.vreg();
                allocd = true;
            }
        }
        if !allocd {
            let preg = if self.freepregs[op.class()] == PRegSet::empty() {
                trace!("Evicting a register");
                self.evict_any_reg(inst, op.class())
            } else {
                trace!("Getting a register from freepregs");
                remove_any_from_pregset(&mut self.freepregs[op.class()]).unwrap()
            };
            trace!("The allocated register for vreg {:?}: {:?}", preg, op.vreg());
            self.lrus[op.class()].poke(preg);
            self.vreg_allocs[op.vreg().vreg()] = Allocation::reg(preg);
            self.vreg_in_preg[preg.index()] = op.vreg();
        }
    }

    fn alloc_fixed_reg_for_operand(&mut self, inst: Inst, op: Operand, preg: PReg) {
        trace!("The fixed preg: {:?} for operand {:?}", preg, op);

        // It is an error for a fixed register clobber to be used for a defined vreg
        // that outlives the instruction, because it will be impossible to restore it.
        if self.func.inst_clobbers(inst).contains(preg) && op.kind() == OperandKind::Def
            && (!self.vregs_first_seen_in_curr_inst.contains(&op.vreg())
                || self.liveout_vregs.contains(&op.vreg()))
        {
            panic!("Invalid input");
        }
        let is_allocatable = !self.is_stack(Allocation::reg(preg))
            && !self.func.inst_clobbers(inst).contains(preg);
        if self.vreg_in_preg[preg.index()] != VReg::invalid() {
            // Something is already in that register. Evict it.
            // TODO: Check if the evicted register is a register in the
            // current instruction. If it is, then there's a problem.
            self.evict_vreg_in_preg(inst, preg);
        } else if self.freed_def_pregs[preg.class()].contains(preg) {
            // Consider the scenario:
            // def v0 (fixed: p0), use v1 (fixed: p0)
            // In the above, p0 has already been used for v0, and since it's a
            // def operand, the register has been freed and kept in `freed_def_pregs`,
            // so it can be added back to the free pregs list after the instruction 
            // has finished processing.
            // To avoid the preg being added back to the free list, it must be removed
            // from `freed_def_pregs` here.
            self.freed_def_pregs[preg.class()].remove(preg);
            self.lrus[preg.class()].append(preg.hw_enc());
        } else if self.free_after_curr_inst[preg.class()].contains(preg) {
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
            trace!("{:?} is now using preg {:?}. Removing it from the free after instruction list", op.vreg(), preg);
            self.free_after_curr_inst[preg.class()].remove(preg);
            if is_allocatable {
                self.lrus[preg.class()].append(preg.hw_enc());
            }
        } else {
            // Find the register in the list of free registers (if it's there).
            // If it's not there, then it must be be a fixed stack slot or
            // a clobber, since clobbers are removed from the free preg list before allocation begins.
            self.freepregs[op.class()].remove(preg);
        }
        if is_allocatable {
            self.lrus[op.class()].poke(preg);
        }
        self.vreg_allocs[op.vreg().vreg()] = Allocation::reg(preg);
        self.vreg_in_preg[preg.index()] = op.vreg();
        trace!("vreg {:?} is now in preg {:?}", op.vreg(), preg);
    }

    /// Allocates for the operand `op` with index `op_idx` into the
    /// vector of instruction `inst`'s operands.
    /// Only non reuse-input operands.
    fn alloc_operand(&mut self, inst: Inst, op: Operand, op_idx: usize) {
        match op.constraint() {
            OperandConstraint::Any => {
                self.alloc_reg_for_operand(inst, op, op_idx);
            }
            OperandConstraint::Reg => {
                self.alloc_reg_for_operand(inst, op, op_idx);
            }
            OperandConstraint::Stack => {
                if self.vreg_spillslots[op.vreg().vreg()].is_invalid() {
                    self.vreg_spillslots[op.vreg().vreg()] = self.allocstack(&op.vreg());
                }
                self.vreg_allocs[op.vreg().vreg()] = Allocation::stack(self.vreg_spillslots[op.vreg().vreg()]);
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
        if let Some(preg) = op.as_fixed_nonallocatable() {
            self.allocs[(inst.index(), op_idx)] = Allocation::reg(preg);
            trace!("Allocation for instruction {:?} and operand {:?}: {:?}", inst, op, self.allocs[(inst.index(), op_idx)]);
            return;
        }
        self.vregs_in_curr_inst.insert(op.vreg());
        self.live_vregs.insert(op.vreg());
        if !self.allocd_within_constraint(inst, op) {
            let prev_alloc = self.vreg_allocs[op.vreg().vreg()];
            if prev_alloc.is_none() {
                self.vregs_first_seen_in_curr_inst.insert(op.vreg());
            }
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
                    
                    if !self.use_vregs_saved_and_restored_in_curr_inst.contains(&op.vreg())
                        && !self.vregs_allocd_in_curr_inst.contains(&op.vreg())
                        // Don't restore after the instruction if it doesn't live past
                        // this instruction.
                        && !self.vregs_first_seen_in_curr_inst.contains(&op.vreg())
                    {
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
                            true,
                        );
                        self.use_vregs_saved_and_restored_in_curr_inst.insert(op.vreg());
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
                    // Free the previous allocation so that it can be reused.
                    let preg = prev_alloc.as_reg().unwrap();
                    self.vreg_in_preg[preg.index()] = VReg::invalid();
                    // If it's a fixed stack slot, then it's not allocatable.
                    if !self.is_stack(prev_alloc) {
                        trace!("{:?} is no longer using preg {:?}, so freeing it after instruction", op.vreg(), preg);
                        // A clobber will have already been removed from the LRU
                        // and will be freed after the instruction has completed processing
                        // if no vreg is still present in it.
                        if !self.func.inst_clobbers(inst).contains(preg) {
                            self.free_after_curr_inst[preg.class()].add(preg);
                            self.lrus[preg.class()].remove(preg.hw_enc());
                        } else {
                            trace!("{:?} is a clobber, so not bothering with the state update", preg);
                        }
                    }
                }
            }
            trace!("Allocation for instruction {:?} and operand {:?}: {:?}", inst, op, self.allocs[(inst.index(), op_idx)]);
        } else {
            self.allocs[(inst.index(), op_idx)] = self.vreg_allocs[op.vreg().vreg()];
            trace!("Allocation for instruction {:?} and operand {:?}: {:?}", inst, op, self.allocs[(inst.index(), op_idx)]);
        }
        self.vregs_allocd_in_curr_inst.insert(op.vreg());
    }

    fn process_reuse_operand_allocation(
        &mut self,
        inst: Inst,
        op: Operand,
        op_idx: usize,
        reused_op: Operand,
        clobbers: PRegSet,
    ) {
        debug_assert!(matches!(op.constraint(), OperandConstraint::Reuse(_)));
        self.vregs_in_curr_inst.insert(op.vreg());
        // To handle reuse operands, the reused input's allocation is always used for
        // both operands. This is to avoid having to think about how to handle fixed
        // register constraints. For example:
        // 1. def v0 (reuse: 1), use v1 (fixed: p0)
        // p0 is simply used for both operands, regardless of where v0 is expected to be
        // after the instruction.
        //
        //
        // A check has to be made to see if the reuse operand has already been allocated,
        // in a previous alloc_inst call. There are 2 cases that need to be considered here:
        //
        // Case 1: The reuse operand has already been allocated.
        // Case 2: The reuse operand has no allocation.
        //
        // For case 1:
        // An example:
        //
        // inst 1: def v0 (reuse: 1), use v1
        // inst 2: use v0
        // In the above example, v0 will have already been allocated by the time inst 1
        // is about to be processed.
        // After this inst 1, v0 is expected to be in some location l0.
        // Depending on whether v1 lives past this instruction, there are also two subcases:
        // Case 1.1: The reused input, v1, lives past the instruction and is expected to be in some
        //          location l1 afterwards.
        // Case 1.2: The reused input v1, doesn't live past the instruction.
        //
        // In case 1.1, both v0 and v1 live past the instruction, so edits have to be inserted to
        // ensure that both end up in their expected after instruction locations.
        // Again, depending on whether or not l0 is used by another operand in the current instruction,
        // there are two subcases here:
        // Case 1.1.1: l0 is used by another operand in the instruction.
        // Case 1.1.2: l0 isn't use by another operand in the instruction.
        //
        // In case 1.1.1:
        //
        // 1. def v0 (reuse: 1), use v1, use v2
        // 2. use v0, use v1
        //
        // At inst 2, v0 is expected to be at l0 and v1 is expected to be at l1.
        // During the processing of inst 1, the v1 and v2 operands are processed before the reuse operand
        // v0. If by some fixed register constraint either v1 or v2 get l0 as an allocation, then v0
        // will be evicted and an edit will be added after inst 1 to move from v0's spillslot to l0
        // by the eviction procedure. Then we'll have:
        //
        // 1. def v0 (reuse: 1), use v1, use v2
        // move from stack_v0 to l0   // Added during eviction.
        // 2. use v0, use v1
        //
        // And v0's current allocation, at the point of invoking this reuse operand allocation
        // procedure will be stack_v0 (vreg_allocs[v0] == stack_v0).
        // Suppose v1 is allocated to p0.
        // For the flow to be correct, two things have to occur:
        // 1. The reused input v1 has to be saved and restored, because the definition of v0
        // overwrites it in inst 1.
        // 2. v0 has to be moved into its current allocation (stack_v0) before the eviction edit.
        //
        // The edits added by this procedure will be like so:
        //
        // move from p0 to stack_v1     // Saving v1.
        // 1. def v0 (reuse: 1), use v1, use v2  // p0 is used for both v1 and v0.
        // move from p0 to vreg_allocs[v0] (== stack_v0)   // Move into v0's current allocation.
        // move from stack_v0 to l0   // Added during eviction.
        // move from stack_v1 to l1   // Restoring v1.
        // 2. use v0, use v1
        //
        // Hence, the edits have to be inserted in the following order:
        // 1. The edit to save the reused input is preprended to the other edits before the instruction.
        // 2. The edit to move v0 from its reused input into its current allocation (spillslot)
        // will be prepended to the edits after the instruction.
        // 3. The edit to restore v1 will be appended to the edits after the instruction.
        //
        // In the case 1.1.2, l0 isn't used by any other operand in the instruction, so no eviction
        // occurs and vreg_allocs[v0] at this point == l0.
        // Here, the only edits that needs to be added is the move from the reused input allocation
        // into vreg_allocs[v0] (which is l0) and the save and restore of v1.
        // For example:
        //
        // 1. def v0 (reuse: 1), use v1, use v2 // Both v0 and v1 use p0
        // 2. use v0, use v1
        //
        // Becomes:
        //
        // move from p0 to stack_v1     // Save v1.
        // 1. def v0 (reuse: 1), use v1, use v2
        // move from p0 to vreg_allocs[v0] (== l0) // Move to v0's current allocation.
        // move from stack_v1 to l1     // Restore v1.
        // 2. use v0, use v1
        //
        // So, the procedure for case 1.1.1 is the same for case 1.1.2.
        //
        // In case 1.2, the reused input doesn't live past the instruction. Only v0
        // does. In this case, the only edit added is the one to move v0 from the reused
        // input allocation p0 to l0. The edits to save and restore v1 are omitted.
        // For example:
        //
        // 1. def v0 (reuse: 1), use v1, use v2 // Both v0 and v1 use p0
        // 2. use v0        // Only v0 lives past inst 1
        //
        // The only edit inserted is the one to move from p0 to vreg_allocs[v0] (l0):
        //
        // 1. def v0 (reuse: 1), use v1, use v2 // Both v0 and v1 use p0
        // move from p0 to vreg_allocs[v0]  // Move to v0's current allocation
        // 2. use v0, use v1
        //
        // In case 2, v0 has no allocation, meaning it doesn't live past the instruction.
        // There are two subcases here:
        // Case 2.1: The reused input v1 lives past the instruction.
        // Case 2.2: The reused input v1 doesn't live past the instruction.
        //
        // In case 2.1, the only edits that need to be added are the ones to save and restore v1.
        // For example:
        //
        // 1. def v0 (reuse: 1), use v1, use v2 // Both v0 and v1 use p0
        // 2. use v1
        //
        // Becomes:
        //
        // move from p0 to stack_v1     // Save v1.
        // 1. def v0 (reuse: 1), use v1, use v2 // Both v0 and v1 use p0.
        // move from stack_v1 to p0     // Restore v1.
        // 2. use v1
        //
        // In case 2.2, no edits should be inserted at all since none of them live past the
        // instruction.
        //
        // From these cases, it can be concluded that edits to save and restore the reused input
        // should be inserted whenever the reused input lives past the current instruction.
        // And the edit to move the reuse operand into its after-instruction-allocation should
        // only be inserted if it lives past the current instruction.
        // And because of interactions with other edits relating to eviction, the edits,
        // when inserted, should be in the following order:
        //
        // 1. The edit to save the reused input is preprended to the other edits before the instruction.
        // 2. The edit to move v0 from its reused input into its current allocation (spillslot)
        // will be prepended to the edits after the instruction.
        // 3. The edit to restore v1 will be appended to the edits after the instruction.
        //
        trace!("Move Reason: Reuse constraints");
        
        if reused_op.kind() != OperandKind::Use || op.kind() != OperandKind::Def
            || reused_op.pos() != OperandPos::Early || op.pos() != OperandPos::Late
            || reused_op.class() != op.class()
        {
            panic!("Invalid input");
        }
        let reused_input_lives_past_curr_inst = !self.vregs_first_seen_in_curr_inst.contains(&reused_op.vreg());
        if self.vreg_allocs[op.vreg().vreg()].is_some() {
            let reused_op_vreg = reused_op.vreg();
            // The only way that a vreg can be assigned a clobber is if a fixed register
            // constraint demands it.
            // Reusing an input assigned to a clobber will result in a clobber being assigned
            // to the vreg being defined. Since the vreg is live after this instruction, this
            // must be an error, because it will be impossible to restore the defined vreg
            // afterwards.
            if let Some(preg) = self.vreg_allocs[reused_op_vreg.vreg()].as_reg() {
                if clobbers.contains(preg) {
                    panic!("Invalid input");
                }
            }
            let op_prev_alloc = self.vreg_allocs[op.vreg().vreg()];
            if self.vreg_spillslots[reused_op_vreg.vreg()].is_invalid() {
                self.vreg_spillslots[reused_op_vreg.vreg()] = self.allocstack(&reused_op_vreg);
            }
            let reused_op_spillslot = self.vreg_spillslots[reused_op.vreg().vreg()];

            // Save the reused input only if it lives past the current instruction.
            if reused_input_lives_past_curr_inst {
                self.add_move_later(
                    inst,
                    self.vreg_allocs[reused_op_vreg.vreg()],
                    Allocation::stack(reused_op_spillslot),
                    op.class(),
                    InstPosition::Before,
                    true,
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
                true,
            );

            // Restore the reused input only if it lives past the current instruction.
            if reused_input_lives_past_curr_inst {
                self.add_move_later(
                    inst,
                    Allocation::stack(reused_op_spillslot),
                    self.vreg_allocs[reused_op_vreg.vreg()],
                    op.class(),
                    InstPosition::After,
                    false,
                );
                self.use_vregs_saved_and_restored_in_curr_inst.insert(reused_op.vreg());
            }

            self.allocs[(inst.index(), op_idx)] = self.vreg_allocs[reused_op_vreg.vreg()];
            self.freealloc(op.vreg(), clobbers);
            trace!("Allocation for instruction {:?} and operand {:?}: {:?}", inst, op, self.allocs[(inst.index(), op_idx)]);
        } else {
            let reused_op_vreg = reused_op.vreg();
            if reused_input_lives_past_curr_inst {
                if self.vreg_spillslots[reused_op_vreg.vreg()].is_invalid() {
                    self.vreg_spillslots[reused_op_vreg.vreg()] = self.allocstack(&reused_op_vreg);
                }
                let reused_op_spillslot = self.vreg_spillslots[reused_op.vreg().vreg()];
                // Save the reused input to its spillslot before the instruction.
                self.add_move_later(
                    inst,
                    self.vreg_allocs[reused_op_vreg.vreg()],
                    Allocation::stack(reused_op_spillslot),
                    op.class(),
                    InstPosition::Before,
                    true,
                );
                // Restore the reused input.
                self.add_move_later(
                    inst,
                    Allocation::stack(reused_op_spillslot),
                    self.vreg_allocs[reused_op_vreg.vreg()],
                    op.class(),
                    InstPosition::After,
                    false,
                );
                self.use_vregs_saved_and_restored_in_curr_inst.insert(reused_op.vreg());
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
            // in an instruction. In this case, edits only need to be inserted if the
            // following conditions are met:
            //
            // 1. All the operands assigned the clobber are all uses of the same vreg 
            // with the same constraint (no defs should be assigned the clobber).
            // 2. No other operand in the instruction uses that vreg with a different constraint.
            // 3. The used vreg lives past the instruction.
            // 4. The expected allocation of the vreg after the instruction is the clobber.
            //
            // Because of the way operand allocation works, edits to save and restore a vreg
            // will have already been inserted during operand allocation if any of the following
            // conditions are met:
            // 1. The expected allocation afterwards is not a clobber.
            // 2. There are multiple operands using the vreg with different constraints.
            // 3. A def operand has the same clobber allocation assigned to it and
            // the vreg lives past the instruction.
            // Therefore, the presence of the vreg in `use_vregs_saved_and_restored`
            // implies that it violates one of the conditions for the edits to be inserted.

            let vreg = self.vreg_in_preg[clobbered_preg.index()];
            if vreg != VReg::invalid() {
                let vreg_isnt_mentioned_in_curr_inst = !self.vregs_in_curr_inst.contains(&vreg);
                let vreg_lives_past_curr_inst = !self.vregs_first_seen_in_curr_inst.contains(&vreg);
                if vreg_isnt_mentioned_in_curr_inst
                    || (!self.use_vregs_saved_and_restored_in_curr_inst.contains(&vreg)
                        && vreg_lives_past_curr_inst)
                {
                    trace!("Adding save and restore edits for {:?}", vreg);
                    let preg_alloc = Allocation::reg(clobbered_preg);
                    let slot = if self.vreg_spillslots[vreg.vreg()].is_valid() {
                        self.vreg_spillslots[vreg.vreg()]
                    } else {
                        self.vreg_spillslots[vreg.vreg()] = self.allocstack(&vreg);
                        self.vreg_spillslots[vreg.vreg()]
                    };
                    let slot_alloc = Allocation::stack(slot);
                    self.add_move_later(
                        inst,
                        preg_alloc,
                        slot_alloc,
                        vreg.class(),
                        InstPosition::Before,
                        true
                    );
                    self.add_move_later(
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

        for (succ_idx, _) in self.func.block_succs(block).iter().enumerate() {
            for vreg in self.func.branch_blockparams(block, inst, succ_idx).iter() {
                // All branch arguments should be in their spillslots at the end of the function.
                self.vreg_allocs[vreg.vreg()] = Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
                self.live_vregs.insert(*vreg);
            }
        }
    }

    fn alloc_inst(&mut self, block: Block, inst: Inst) {
        trace!("Allocating instruction {:?}", inst);
        if self.func.is_branch(inst) {
            self.process_branch(block, inst);
        }
        let operands = Operands::new(self.func.inst_operands(inst));
        let clobbers = self.func.inst_clobbers(inst);
        for preg in clobbers {
            // To avoid allocating clobbers, they are removed from the
            // free register list. To also avoid a clobber being evicted,
            // it's also removed from the LRU.
            // The only way a clobber can be marked as the allocation of
            // an operand is through a fixed register constraint to the clobber
            // or a reused input constraint of an operand with a fixed register
            // constraint to use a clobber.
            if self.allocatable_regs.contains(preg) {
                trace!("Removing {:?} from the freelist because it's a clobber", preg);
                self.freepregs[preg.class()].remove(preg);
                self.lrus[preg.class()].remove(preg.hw_enc());
            }
        }
        for (_, op) in operands.reuse() {
            let OperandConstraint::Reuse(reused_idx) = op.constraint() else {
                unreachable!()
            };
            self.reused_inputs_in_curr_inst.push(reused_idx);
        }
        for (op_idx, op) in operands.fixed_late() {
            self.process_operand_allocation(inst, op, op_idx);
        }
        for (op_idx, op) in operands.non_fixed_non_reuse_late_def() {
            self.process_operand_allocation(inst, op, op_idx);
        }
        for (_, op) in operands.non_reuse_late_def() {
            self.freealloc(op.vreg(), clobbers);
        }
        for (op_idx, op) in operands.fixed_early() {
            self.process_operand_allocation(inst, op, op_idx);
        }
        for (op_idx, op) in operands.non_fixed_non_reuse_late_use() {
            self.process_operand_allocation(inst, op, op_idx);
        }
        for (op_idx, op) in operands.non_fixed_non_reuse_early() {
            self.process_operand_allocation(inst, op, op_idx);
        }
        for (_, op) in operands.non_reuse_early_def() {
            self.freealloc(op.vreg(), clobbers);
        }
        for (op_idx, op) in operands.reuse() {
            let OperandConstraint::Reuse(reused_idx) = op.constraint() else {
                unreachable!()
            };
            self.process_reuse_operand_allocation(inst, op, op_idx, operands.0[reused_idx], clobbers);
        }
        self.save_and_restore_clobbered_registers(inst);
        for preg in self.func.inst_clobbers(inst) {
            if self.allocatable_regs.contains(preg) {
                if self.vreg_in_preg[preg.index()] == VReg::invalid() {
                    // In the case where the clobbered register is allocated to
                    // something, don't add the register to the freelist, cause
                    // it isn't free.
                    trace!("Adding clobbered {:?} to free after inst list", preg);
                    // Consider a scenario:
                    //
                    // 1. use v0 (fixed: p1). Clobbers: [p0]
                    // 2. use v0 (fixed: p0)
                    //
                    // In the above, v0 is first allocated to p0 at inst 2.
                    // At inst 1, v0's allocation is changed to p1 and edits are inserted
                    // to save and restore v0:
                    //
                    // move from p1 to stack_v0
                    // 1. use v0 (fixed: p1). Clobbers: [p0]
                    // move from stack_v0 to p0
                    // 2. use v0 (fixed: p0)
                    //
                    // Suppose some other edits need to be inserted before/after inst 1
                    // and scratch registers are needed.
                    // If the clobber p0 is added back to the free list directly,
                    // p0 may end up be being used as a scratch register and get overwritten
                    // before inst 2 is reached. This could happen if inst 1 is a safepoint and
                    // edits to save and restore reftypes are prepended before the inst
                    // and after resulting in the following scenario:
                    //
                    // --- p0 is overwritten ---
                    // move from p1 to stack_v0
                    // 1. use v0 (fixed: p1). Clobbers: [p0]
                    // move from stack_v0 to p0
                    // --- p0 is overwritten ---
                    // 2. use v0 (fixed: p0)
                    //
                    // To avoid this scenario, the registers are added to the
                    // `free_after_curr_inst` instead, to ensure that it isn't used as
                    // a scratch register.
                    self.free_after_curr_inst[preg.class()].add(preg);
                } else {
                    // Something is still in the clobber.
                    // After this instruction, it's no longer a clobber.
                    // Add it back to the LRU.
                    trace!("Something is still in the clobber {:?}. Adding it back to the LRU directly.", preg);
                    self.lrus[preg.class()].append_and_poke(preg);
                }
            }
        }
        trace!("After the allocation:");
        trace!("freed_def_pregs: {:?}", self.freed_def_pregs);
        trace!("free after curr inst: {:?}", self.free_after_curr_inst);
        trace!("");
        let scratch_regs = self.get_scratch_regs(inst);
        self.process_edits(scratch_regs);
        self.add_freed_regs_to_freelist();
        self.use_vregs_saved_and_restored_in_curr_inst.clear();
        self.vregs_first_seen_in_curr_inst.clear();
        self.vregs_allocd_in_curr_inst.clear();
        self.reused_inputs_in_curr_inst.clear();
        self.vregs_in_curr_inst.clear();

        if trace_enabled!() {
            self.log_post_inst_processing_state(inst);
        }
    }

    /// At the beginning of every block, all virtual registers that are
    /// livein are expected to be in their respective spillslots.
    /// This function sets the current allocations of livein registers
    /// to their spillslots and inserts the edits to flow livein values to
    /// the allocations where they are expected to be before the first
    /// instruction. 
    fn reload_at_begin(&mut self, block: Block) {
        trace!("Reloading live registers at the beginning of block {:?}", block);
        trace!("Live registers at the beginning of block {:?}: {:?}", block, self.live_vregs);
        trace!("Block params at block {:?} beginning: {:?}", block, self.func.block_params(block));
        // We need to check for the registers that are still live.
        // These registers are livein and they should be stack-allocated.
        let live_vregs = self.live_vregs.clone();
        for vreg in live_vregs.iter().cloned() {
            trace!("Processing {:?}", vreg);
            if self.vreg_spillslots[vreg.vreg()].is_invalid() {
                self.vreg_spillslots[vreg.vreg()] = self.allocstack(&vreg);
            }
            // The allocation where the vreg is expected to be before
            // the first instruction.
            let prev_alloc = self.vreg_allocs[vreg.vreg()];
            let slot = Allocation::stack(self.vreg_spillslots[vreg.vreg()]);
            if self.func.block_params(block).contains(&vreg) {
                trace!("{:?} is a block param. Freeing it", vreg);
                // A block's block param is not live before the block.
                // And `vreg_allocs[i]` of a virtual register i is none for
                // dead vregs.
                self.freealloc(vreg, PRegSet::empty());
            } else {
                trace!("{:?} is not a block param. It's a liveout vreg from some predecessor", vreg);
                trace!("Setting {:?}'s current allocation to its spillslot", vreg);
                // It is a liveout vreg from a predecessor.
                self.vreg_allocs[vreg.vreg()] = slot;
                trace!("Recording that {:?} is a liveout", vreg);
                // Need to remember that this is a liveout vreg so that its
                // spillslot, if it's a reftype, can be recorded in the stackmap
                // later.
                self.liveout_vregs.insert(vreg);
                if let Some(preg) = prev_alloc.as_reg() {
                    trace!("{:?} was in {:?}. Removing it", preg, vreg);
                    // Nothing is in that preg anymore. Return it to
                    // the free preg list.
                    self.vreg_in_preg[preg.index()] = VReg::invalid();
                    if !self.is_stack(prev_alloc) {
                        trace!("{:?} is not a fixed stack slot. Recording it in the freed def pregs list", prev_alloc);
                        // Using this instead of directly adding it to
                        // freepregs to prevent allocated registers from being
                        // used as scratch registers.
                        self.freed_def_pregs[preg.class()].add(preg);
                        self.lrus[preg.class()].remove(preg.hw_enc());
                    }
                }
            }
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
        }
        for block_param_vreg in self.func.block_params(block) {
            trace!("Removing block param {:?} from the live regs set", block_param_vreg);
            self.live_vregs.remove(block_param_vreg);
        }
        self.process_edits(self.get_scratch_regs_for_reloading());
        self.add_freed_regs_to_freelist();

        if trace_enabled!() {
            self.log_post_reload_at_begin_state(block);
        }
    }

    fn log_post_reload_at_begin_state(&self, block: Block) {
        use alloc::format;
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
        trace!("Free int pregs: {:?}", self.freepregs[RegClass::Int]);
        trace!("Free float pregs: {:?}", self.freepregs[RegClass::Float]);
        trace!("Free vector pregs: {:?}", self.freepregs[RegClass::Vector]);
    }

    fn log_post_inst_processing_state(&self, inst: Inst) {
        use alloc::format;
        trace!("");
        trace!("State after instruction {:?}", inst);
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
        trace!("Free int pregs: {:?}", self.freepregs[RegClass::Int]);
        trace!("Free float pregs: {:?}", self.freepregs[RegClass::Float]);
        trace!("Free vector pregs: {:?}", self.freepregs[RegClass::Vector]);
    }

    fn alloc_block(&mut self, block: Block) {
        trace!("{:?} start", block);
        for inst in self.func.block_insns(block).iter().rev() {
            self.alloc_inst(block, inst);
        }
        self.reload_at_begin(block);
        trace!("{:?} end\n", block);
    }

    fn run(&mut self) -> Result<(), RegAllocError> {
        debug_assert_eq!(self.func.entry_block().index(), 0);
        for block in (0..self.func.num_blocks()).rev() {
            self.alloc_block(Block::new(block));
        }

        Ok(())
    }
}

fn log_function<F: Function>(func: &F) {
    trace!("Processing a new function");
    for block in 0..func.num_blocks() {
        let block = Block::new(block);
        trace!("Block {:?}. preds: {:?}. succs: {:?}, params: {:?}",
            block, func.block_preds(block), func.block_succs(block),
            func.block_params(block)
        );
        for inst in func.block_insns(block).iter() {
            let clobbers = func.inst_clobbers(inst);
            let mut cls = Vec::new();
            for c in clobbers {
                cls.push(c);
            }
            trace!("inst{:?}: {:?}. Clobbers: {:?}", inst.index(), func.inst_operands(inst), cls);
            if func.is_branch(inst) {
                trace!("Block args: ");
                for (succ_idx, _succ) in func.block_succs(block).iter().enumerate() {
                    trace!(" {:?}", func.branch_blockparams(block, inst, succ_idx));
                }
            }
            trace!("");
        }
        trace!("");
    }
}

fn log_output<'a, F: Function>(env: &Env<'a, F>) {
    trace!("Done!");
    let mut v = Vec::new();
    for i in 0..env.func.num_vregs() {
        if env.vreg_spillslots[i].is_valid() {
            v.push((
                VReg::new(i, RegClass::Int),
                Allocation::stack(env.vreg_spillslots[i])
            ));
        }
    }
    trace!("VReg spillslots: {:?}", v);
    trace!("Temp spillslots: {:?}", env.temp_spillslots);
    trace!("Final edits: {:?}", env.edits);
    trace!("safepoint_slots: {:?}\n", env.safepoint_slots);
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
        edits: env.edits.make_contiguous().to_vec(),
        allocs: env.allocs.allocs,
        inst_alloc_offsets: env.allocs.inst_alloc_offsets,
        num_spillslots: env.num_spillslots as usize,
        // TODO: Handle debug locations.
        debug_locations: Vec::new(),
        safepoint_slots: env.safepoint_slots,
        stats: env.stats,
    })
}
