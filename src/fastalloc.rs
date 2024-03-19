use core::convert::TryInto;

use crate::{Block, InstRange, Inst, OperandKind, Operand, PReg, RegClass, VReg, SpillSlot, FxHashMap, AllocationKind};
use crate::{Function, MachineEnv, ssa::validate_ssa, ProgPoint, Edit, Output};
use crate::{cfg::CFGInfo, RegAllocError, Allocation, ion::Stats};
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

use std::println;

/// A least recently used cache organized as a linked list based on a vector
#[derive(Debug)]
struct Lru {
    /// The list of node information
    ///
    /// Each node corresponds to a physical register.
    /// The index of a node is the `address` from the perspective of the linked list.
    data: Vec<LruNode>,
    /// Index of the most recently used register
    head: usize,
    class: RegClass,
}

#[derive(Clone, Copy, Debug)]
struct LruNode {
    /// The previous physical register in the list
    prev: usize,
    /// The next physical register in the list
    next: usize,
}

impl Lru {
    fn new(no_of_regs: usize) -> Self {
        let mut data = Vec::with_capacity(no_of_regs);
        for _ in 0..no_of_regs {
            data.push(LruNode { prev: 0, next: 0 });
        }
        let mut lru = Self {
            head: 0,
            data,
            class: RegClass::Int
        };
        for i in 0..no_of_regs {
            lru.data[i].prev = i.checked_sub(1).unwrap_or(no_of_regs - 1);
            lru.data[i].next = (i + 1) % no_of_regs;
        }
        lru
    }

    /// Marks the physical register `i` as the most recently used
    /// and sets `vreg` as the virtual register it contains
    fn poke(&mut self, preg: PReg) {
        let prev_newest = self.head;
        let i = preg.hw_enc();
        if i == prev_newest {
            return;
        }
        if self.data[prev_newest].prev != i {
            self.remove(i);
            self.insert_before(i, self.head);
        }
        self.head = i;
    }

    /// Gets the least recently used physical register.
    fn pop(&mut self) -> PReg {
        let oldest = self.data[self.head].prev;
        PReg::new(oldest, self.class)
    }

    /// Splices out a node from the list
    fn remove(&mut self, i: usize) {
        let (iprev, inext) = (self.data[i].prev, self.data[i].next);
        self.data[iprev].next = self.data[i].next;
        self.data[inext].prev = self.data[i].prev;
    }

    /// Insert node `i` before node `j` in the list
    fn insert_before(&mut self, i: usize, j: usize) {
        let prev = self.data[j].prev;
        self.data[prev].next = i;
        self.data[j].prev = i;
        self.data[i] = LruNode {
            next: j,
            prev,
        };
    }
}

/// Info about the operand currently in a `PReg`.
#[derive(Debug)]
struct OpInfo {
    /// The instruction the operand is in.
    inst: Inst,
    /// The index of the operand in the instruction.
    op_idx: usize,
    /// The `VReg` in the `PReg`.
    vreg: VReg,
}

#[derive(Debug)]
struct LiveRegInfo {
    /// The current allocation.
    alloc: Allocation,
}

#[derive(Debug)]
pub struct Env<'a, F: Function> {
    func: &'a F,
    env: &'a MachineEnv,
    /// What do we know about the live registers?
    livevregs: HashMap<VReg, LiveRegInfo>,
    /// Which virtual registers are held in physical registers.
    preg_info: HashMap<PReg, VReg>,
    /// Free registers for each register class: Int, Float and Vector,
    /// respectively.
    freepregs: [Vec<PReg>; 3],
    /// Least recently used cache, for eviction.
    lru: Lru,
    /// All registers used in the current instruction being allocated.
    regs_used_in_curr_inst: HashSet<PReg>,
    /// Spillslots allocated to virtual registers.
    allocd_stackslots: HashMap<VReg, SpillSlot>,
    /// Offset for the next stack slot allocation.
    next_freestack_offset: usize,

    // Output.
    allocs: Vec<Allocation>,
    inst_alloc_offsets: Vec<u32>,
    edits: Vec<(ProgPoint, Edit)>,
    num_spillslots: u32,

    stats: Stats,
}

impl<'a, F: Function> Env<'a, F> {
    fn new(func: &'a F, env: &'a MachineEnv) -> Self {
        let freepregs = [
            env.preferred_regs_by_class[RegClass::Int as usize].clone(),
            env.preferred_regs_by_class[RegClass::Float as usize].clone(),
            env.preferred_regs_by_class[RegClass::Vector as usize].clone(),
        ];
        let lru = Lru::new(freepregs[RegClass::Int as usize].len());
        let inst_alloc_offsets: Vec<u32> = (0..func.num_insts())
            .map(|_| 0).collect();
        Self {
            func,
            env,
            allocs: Vec::with_capacity(func.num_vregs()),
            edits: Vec::new(),
            inst_alloc_offsets,
            num_spillslots: 0,
            stats: Stats::default(),
            livevregs: HashMap::with_capacity(func.num_vregs()),
            freepregs,
            lru,
            regs_used_in_curr_inst: HashSet::new(),
            allocd_stackslots: HashMap::new(),
            preg_info: HashMap::new(),
            next_freestack_offset: 0,
        }
    }

    fn init_operands_allocs(&mut self, operands: &[Operand], inst: Inst) {
        let no_of_operands: u32 = operands.len().try_into().unwrap();
        let offset = self.allocs.len();
        for _ in 0..no_of_operands {
            self.allocs.push(Allocation::none());
        }
        self.inst_alloc_offsets[inst.index()] = offset.try_into().unwrap();
    }

    fn assigned_reg(&self, operand: &Operand) -> Option<PReg> {
        self.livevregs.get(&operand.vreg())
            .and_then(|info| info.alloc.as_reg().map(|reg| reg))
    }

    fn no_of_operands(&self, inst: Inst) -> usize {
        self.func.inst_operands(inst).len()
    }

    fn curralloc_mut(&mut self, inst: Inst, op_idx: usize) -> &mut Allocation {
        let inst_offset = self.inst_alloc_offsets[inst.index()];
        let inst_offset: usize = inst_offset.try_into().unwrap();
        let no_of_operands = self.no_of_operands(inst);
        // The end results will be reversed
        // So, the operands have to be put in reverse order to
        // avoid breaking the external API.
        &mut self.allocs[inst_offset + (no_of_operands - op_idx - 1)]
    }

    fn allocstack(&mut self, vreg: VReg) -> Allocation {
        let ss = if let Some(ss) = self.allocd_stackslots.get(&vreg) {
            *ss
        } else {
            let size = self.func.spillslot_size(vreg.class());
            let offset = self.next_freestack_offset;
            let slot = (offset + size - 1) & !(size - 1);
            self.next_freestack_offset = offset + size;
            let ss = SpillSlot::new(slot);
            self.allocd_stackslots.insert(vreg, ss);
            ss
        };
        Allocation::stack(ss)
    }

    fn evictreg(&mut self, inst: Inst) -> PReg {
        let preg = self.lru.pop();
        let evicted_vreg = self.preg_info[&preg];
        let stackloc = self.allocstack(evicted_vreg);
        self.edits.push((ProgPoint::after(inst), Edit::Move {
            from: stackloc,
            to: Allocation::reg(preg),
        }));
        self.livevregs.get_mut(&evicted_vreg).unwrap().alloc = stackloc;
        preg
    }

    fn freealloc(&mut self, operand: &Operand) {
        let livereg = self.livevregs.get(&operand.vreg())
            .expect("Trying to free an unallocated vreg");
        match livereg.alloc.kind() {
            AllocationKind::Reg => {
                let preg = livereg.alloc.as_reg().unwrap();
                self.freepregs[operand.class() as usize].push(preg);
                self.preg_info.remove(&preg);
                self.livevregs.remove(&operand.vreg());
            }
            _ => unimplemented!()
        };
    }

    /// Allocate a physical register for `operand` at index `idx` in
    /// instruction `inst`.
    fn allocreg(&mut self, operand: &Operand, inst: Inst, idx: usize) -> PReg {
        let freepregs_idx = operand.class() as usize;
        let preg = if self.freepregs[freepregs_idx].is_empty() {
            self.evictreg(inst)
        } else{
            self.freepregs[freepregs_idx].pop().unwrap()
        };
        self.lru.poke(preg);
        self.livevregs.insert(operand.vreg(), LiveRegInfo {
            alloc: Allocation::reg(preg)
        });
        self.preg_info.insert(preg, operand.vreg());
        *self.curralloc_mut(inst, idx) = Allocation::reg(preg);
        preg
    }

    fn alloc_inst(&mut self, inst: Inst) -> Result<(), RegAllocError> {
        let operands = self.func.inst_operands(inst);
        self.init_operands_allocs(operands, inst);
        let mut def_operands = Vec::with_capacity(operands.len());
        let mut use_operands = Vec::with_capacity(operands.len());
        for (i, operand) in operands.iter().enumerate() {
            if operand.kind() == OperandKind::Def {
                def_operands.push(i);
            } else {
                use_operands.push(i);
            }
        }

        for idx in def_operands {
            if let Some(preg) = self.assigned_reg(&operands[idx]) {
                *self.curralloc_mut(inst, idx) = Allocation::reg(preg);
            } else {
                self.allocreg(&operands[idx], inst, idx);
            }
            self.freealloc(&operands[idx]);
        }
        
        for idx in use_operands {
            let operand = &operands[idx];
            let prevalloc = if let Some(livereg) = self.livevregs.get(&operand.vreg()) {
                Some(livereg.alloc.clone())
            } else {
                None
            };
            let assigned_reg = self.assigned_reg(operand)
                .map(|reg| reg.clone());
            let preg = if let Some(preg) = assigned_reg {
                *self.curralloc_mut(inst, idx) = Allocation::reg(preg);
                self.lru.poke(preg);
                preg
            } else {
                let preg = self.allocreg(operand, inst, idx);
                if self.regs_used_in_curr_inst.contains(&preg) {
                    return Err(RegAllocError::TooManyLiveRegs);
                }
                self.regs_used_in_curr_inst.insert(preg);
                preg
            };
            if let Some(prevalloc) = prevalloc {
                if prevalloc.is_stack() {
                    self.edits.push((ProgPoint::before(inst), Edit::Move {
                        from: Allocation::reg(preg),
                        to: Allocation::stack(prevalloc.as_stack().unwrap()),
                    }));
                }
            }
        }
        self.regs_used_in_curr_inst.clear();
        Ok(())
    }

    /// Allocates instructions in reverse order.
    fn alloc_basic_block(&mut self, block: Block) -> Result<(), RegAllocError> {
        for inst in self.func.block_insns(block).iter().rev() {
            self.alloc_inst(inst)?;
        }
        // Reversing the result to conform to the external API
        self.reverse_results();
        Ok(())
    }

    fn reverse_results(&mut self) {
        let mut offset = 0;
        let mut prev_end: u32 = self.allocs.len().try_into().unwrap();
        for i in 0..self.inst_alloc_offsets.len() - 1 {
            let diff = prev_end as u32 - self.inst_alloc_offsets[i];
            prev_end = self.inst_alloc_offsets[i];
            self.inst_alloc_offsets[i] = offset;
            offset += diff;
        }
        *self.inst_alloc_offsets.last_mut().unwrap() = offset;
        self.allocs.reverse();
        self.edits.reverse();
    }

    fn run(&mut self) -> Result<(), RegAllocError> {
        for blocknum in 0..self.func.num_blocks() {
            self.alloc_basic_block(Block::new(blocknum))?;
        }
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

    Ok(Output {
        edits: env.edits,
        allocs: env.allocs,
        inst_alloc_offsets: env.inst_alloc_offsets,
        num_spillslots: env.num_spillslots as usize,
        debug_locations: Vec::new(),
        safepoint_slots: Vec::new(),
        stats: env.stats,
    })
}
