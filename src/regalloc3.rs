use core::iter::{self, FromIterator};

use alloc::vec::Vec;
use regalloc3::entity::{PackedOption, PrimaryMap};
use regalloc3::function::TerminatorKind;
use regalloc3::reginfo::{PhysRegSet, RegGroupSet};
use regalloc3::{
    function::{
        Block, Function, Inst, InstRange, Operand, OperandConstraint, OperandKind, RematCost,
        Value, ValueGroup,
    },
    output::{Allocation, AllocationKind, OutputInst},
    reginfo::{PhysReg, RegBank, RegClass, RegClassSet, RegGroup, RegInfo, RegUnit, SpillSlotSize},
    RegAllocError, RegisterAllocator,
};

use crate::cfg::CFGInfoCtx;
use crate::{cfg::CFGInfo, ion::Stats, Edit, MachineEnv, Output, PReg, PRegSet, ProgPoint, VReg};

// We create 6 register classes in 3 banks:
// - Int { Any, Reg }
// - Float { Any, Reg }
// - Vector { Any, Reg }
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ClassKind {
    Any = 0,
    Reg = 1,
}
impl ClassKind {
    fn from_class(class: RegClass) -> Self {
        match class.index() % 2 {
            0 => Self::Any,
            1 => Self::Reg,
            _ => unreachable!(),
        }
    }

    fn to_class(self, bank: RegBank) -> RegClass {
        RegClass::new(bank.index() * 2 + self as usize)
    }
}

struct RegData {
    is_stack: bool,
    allocatable: bool,
}

struct RegBankData {
    spillslot_size: SpillSlotSize,
}

struct RegClassData {
    regs: PhysRegSet,
    subclasses: RegClassSet,
    allocation_order: Vec<PhysReg>,
}

#[derive(Default)]
struct RegInfoImpl {
    regs: PrimaryMap<PhysReg, RegData>,
    classes: PrimaryMap<RegClass, RegClassData>,
    banks: PrimaryMap<RegBank, RegBankData>,
}

fn class_to_bank(class: crate::RegClass) -> RegBank {
    match class {
        crate::RegClass::Int => RegBank::new(0),
        crate::RegClass::Float => RegBank::new(1),
        crate::RegClass::Vector => RegBank::new(2),
    }
}

impl RegInfo for RegInfoImpl {
    #[inline]
    fn num_banks(&self) -> usize {
        3
    }

    #[inline]
    fn top_level_class(&self, bank: RegBank) -> RegClass {
        ClassKind::Any.to_class(bank)
    }

    #[inline]
    fn stack_to_stack_class(&self, bank: RegBank) -> RegClass {
        ClassKind::Reg.to_class(bank)
    }

    #[inline]
    fn bank_for_class(&self, class: RegClass) -> RegBank {
        RegBank::new(class.index() / 2)
    }

    #[inline]
    fn bank_for_reg(&self, reg: PhysReg) -> Option<RegBank> {
        if self.regs[reg].allocatable {
            Some(class_to_bank(PReg::from_index(reg.index()).class()))
        } else {
            None
        }
    }

    #[inline]
    fn spillslot_size(&self, bank: RegBank) -> SpillSlotSize {
        self.banks[bank].spillslot_size
    }

    #[inline]
    fn num_classes(&self) -> usize {
        6
    }

    #[inline]
    fn class_members(&self, class: RegClass) -> PhysRegSet {
        self.classes[class].regs
    }

    fn class_group_members(&self, _class: RegClass) -> RegGroupSet {
        RegGroupSet::new()
    }

    #[inline]
    fn class_includes_spillslots(&self, class: RegClass) -> bool {
        ClassKind::from_class(class) != ClassKind::Reg
    }

    #[inline]
    fn class_spill_cost(&self, class: RegClass) -> f32 {
        match ClassKind::from_class(class) {
            ClassKind::Any => 0.5,
            ClassKind::Reg => 1.0,
        }
    }

    #[inline]
    fn allocation_order(&self, class: RegClass) -> &[PhysReg] {
        &self.classes[class].allocation_order
    }

    fn group_allocation_order(&self, _class: RegClass) -> &[RegGroup] {
        &[]
    }

    #[inline]
    fn sub_classes(&self, class: RegClass) -> RegClassSet {
        self.classes[class].subclasses
    }

    #[inline]
    fn class_group_size(&self, _class: RegClass) -> usize {
        1
    }

    #[inline]
    fn num_regs(&self) -> usize {
        self.regs.len()
    }

    #[inline]
    fn reg_units(&self, reg: PhysReg) -> impl Iterator<Item = RegUnit> {
        iter::once(RegUnit::new(reg.index()))
    }

    #[inline]
    fn is_memory(&self, reg: PhysReg) -> bool {
        self.regs[reg].is_stack
    }

    #[inline]
    fn num_reg_groups(&self) -> usize {
        0
    }

    #[inline]
    fn reg_group_members(&self, _group: RegGroup) -> &[PhysReg] {
        &[]
    }

    #[inline]
    fn group_for_reg(
        &self,
        _reg: PhysReg,
        _group_index: usize,
        _class: RegClass,
    ) -> Option<RegGroup> {
        None
    }
}

impl RegInfoImpl {
    fn build(&mut self, func: &impl crate::Function, mach_env: &MachineEnv) {
        self.regs.clear();
        self.classes.clear();
        self.banks.clear();

        for class in [
            crate::RegClass::Int,
            crate::RegClass::Float,
            crate::RegClass::Vector,
        ] {
            // spillslot_size panics in cranelift if called with RegClass::Vector
            let spillslot_size = if mach_env.preferred_regs_by_class[class as usize].is_empty()
                && mach_env.non_preferred_regs_by_class[class as usize].is_empty()
            {
                SpillSlotSize::new(1)
            } else {
                SpillSlotSize::new(func.spillslot_size(class).max(1) as u32)
            };
            let bank = self.banks.push(RegBankData { spillslot_size });

            let preferred_regs = || {
                mach_env.preferred_regs_by_class[class as usize]
                    .iter()
                    .map(|preg| PhysReg::new(preg.index()))
            };
            let non_preferred_regs = || {
                mach_env.non_preferred_regs_by_class[class as usize]
                    .iter()
                    .map(|preg| PhysReg::new(preg.index()))
            };
            let stack_regs = || {
                mach_env
                    .fixed_stack_slots
                    .iter()
                    .filter(|preg| preg.class() == class)
                    .map(|preg| PhysReg::new(preg.index()))
            };

            // Any
            self.classes.push(RegClassData {
                regs: PhysRegSet::from_iter(
                    preferred_regs()
                        .chain(non_preferred_regs())
                        .chain(stack_regs()),
                ),
                subclasses: RegClassSet::from_iter([
                    ClassKind::Any.to_class(bank),
                    ClassKind::Reg.to_class(bank),
                ]),
                allocation_order: preferred_regs().chain(non_preferred_regs()).collect(),
            });
            // Reg
            self.classes.push(RegClassData {
                regs: PhysRegSet::from_iter(preferred_regs().chain(non_preferred_regs())),
                subclasses: RegClassSet::from_iter([ClassKind::Reg.to_class(bank)]),
                allocation_order: preferred_regs().chain(non_preferred_regs()).collect(),
            });
        }
        for _ in 0..PReg::NUM_INDEX {
            self.regs.push(RegData {
                is_stack: false,
                allocatable: false,
            });
        }
        for &preg in &mach_env.fixed_stack_slots {
            self.regs[PhysReg::new(preg.index())].allocatable = true;
            self.regs[PhysReg::new(preg.index())].is_stack = true;
        }
        for &preg in mach_env.preferred_regs_by_class.iter().flatten() {
            self.regs[PhysReg::new(preg.index())].allocatable = true;
        }
        for &preg in mach_env.non_preferred_regs_by_class.iter().flatten() {
            self.regs[PhysReg::new(preg.index())].allocatable = true;
        }
        for &preg in mach_env.scratch_by_class.iter().flatten() {
            self.regs[PhysReg::new(preg.index())].allocatable = true;
        }
    }
}

#[derive(Clone)]
struct BlockData {
    insts: InstRange,
    preds: u32,
    succs: u32,
    block_params_in: u32,
    block_params_out: u32,
    idom: PackedOption<Block>,
    frequency: f32,
}

#[derive(Clone)]
struct InstData {
    block: Block,
    operands: u32,
    clobbers: u32,
    terminator_kind: Option<TerminatorKind>,
}

#[derive(Default)]
struct Values {
    bank: PrimaryMap<Value, RegBank>,
    mapping: Vec<PackedOption<Value>>,
}

impl Values {
    fn clear(&mut self, func: &impl crate::Function) {
        self.bank.clear();
        self.mapping.clear();
        self.mapping.resize(func.num_vregs(), None.into());
    }

    // regalloc3 requires all values to be used at least once in the function
    // and only allows blockparams for blocks with multiple predecessors.
    fn map(&mut self, vreg: VReg) -> Value {
        if let Some(value) = self.mapping[vreg.vreg()].expand() {
            value
        } else {
            let class = vreg.class();
            let bank = class_to_bank(class);
            let value = self.bank.push(bank);
            self.mapping[vreg.vreg()] = Some(value).into();
            value
        }
    }

    fn alias_blockparam(&mut self, blockparam_out: VReg, blockparam_in: VReg) {
        let value = self.map(blockparam_out);
        self.mapping[blockparam_in.vreg()] = Some(value).into();
    }
}

#[derive(Default)]
pub struct FunctionAdapter {
    blocks: PrimaryMap<Block, BlockData>,
    insts: PrimaryMap<Inst, InstData>,
    values: Values,

    // Storage for lists
    preds: Vec<Block>,
    succs: Vec<Block>,
    block_params_in: Vec<Value>,
    block_params_out: Vec<Value>,
    operands: Vec<Operand>,
    clobbers: Vec<RegUnit>,
}

impl Function for FunctionAdapter {
    #[inline]
    fn num_insts(&self) -> usize {
        self.insts.len()
    }

    #[inline]
    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    #[inline]
    fn block_insts(&self, block: Block) -> InstRange {
        self.blocks[block].insts
    }

    #[inline]
    fn inst_block(&self, inst: Inst) -> Block {
        self.insts[inst].block
    }

    #[inline]
    fn block_succs(&self, block: Block) -> &[Block] {
        let start = self.blocks[block].succs as usize;
        let end = match self.blocks.get(block.next()) {
            Some(b) => b.succs as usize,
            None => self.succs.len(),
        };
        &self.succs[start..end]
    }

    #[inline]
    fn block_preds(&self, block: Block) -> &[Block] {
        let start = self.blocks[block].preds as usize;
        let end = match self.blocks.get(block.next()) {
            Some(b) => b.preds as usize,
            None => self.preds.len(),
        };
        &self.preds[start..end]
    }

    #[inline]
    fn block_immediate_dominator(&self, block: Block) -> Option<Block> {
        self.blocks[block].idom.into()
    }

    #[inline]
    fn block_params(&self, block: Block) -> &[Value] {
        let start = self.blocks[block].block_params_in as usize;
        let end = match self.blocks.get(block.next()) {
            Some(b) => b.block_params_in as usize,
            None => self.block_params_in.len(),
        };
        &self.block_params_in[start..end]
    }

    #[inline]
    fn terminator_kind(&self, inst: Inst) -> Option<TerminatorKind> {
        self.insts[inst].terminator_kind
    }

    #[inline]
    fn jump_blockparams(&self, block: Block) -> &[Value] {
        let start = self.blocks[block].block_params_out as usize;
        let end = match self.blocks.get(block.next()) {
            Some(b) => b.block_params_out as usize,
            None => self.block_params_out.len(),
        };
        &self.block_params_out[start..end]
    }

    #[inline]
    fn block_frequency(&self, block: Block) -> f32 {
        self.blocks[block].frequency
    }

    #[inline]
    fn block_is_critical_edge(&self, block: Block) -> bool {
        self.block_insts(block).len() == 1 && self.block_succs(block).len() == 1
    }

    #[inline]
    fn inst_operands(&self, inst: Inst) -> &[Operand] {
        let start = self.insts[inst].operands as usize;
        let end = match self.insts.get(inst.next()) {
            Some(i) => i.operands as usize,
            None => self.operands.len(),
        };
        &self.operands[start..end]
    }

    #[inline]
    fn inst_clobbers(&self, inst: Inst) -> impl Iterator<Item = RegUnit> {
        let start = self.insts[inst].clobbers as usize;
        let end = match self.insts.get(inst.next()) {
            Some(i) => i.clobbers as usize,
            None => self.clobbers.len(),
        };
        self.clobbers[start..end].iter().copied()
    }

    #[inline]
    fn num_values(&self) -> usize {
        self.values.bank.len()
    }

    #[inline]
    fn value_bank(&self, value: Value) -> RegBank {
        self.values.bank[value]
    }

    #[inline]
    fn num_value_groups(&self) -> usize {
        0
    }

    #[inline]
    fn value_group_members(&self, _group: ValueGroup) -> &[Value] {
        &[]
    }

    #[inline]
    fn can_rematerialize(&self, _value: Value) -> Option<(RematCost, RegClass)> {
        None
    }

    #[inline]
    fn can_eliminate_dead_inst(&self, _inst: Inst) -> bool {
        false
    }
}

impl FunctionAdapter {
    fn build(&mut self, func: &impl crate::Function, cfg: &CFGInfo) {
        self.blocks.clear();
        self.insts.clear();
        self.values.clear(func);
        self.preds.clear();
        self.succs.clear();
        self.block_params_in.clear();
        self.block_params_out.clear();
        self.operands.clear();
        self.clobbers.clear();

        // Eliminate blockparams for blocks with only one predecessor.
        for &block in cfg.postorder.iter().rev() {
            let block_insn = func.block_insns(block);
            for (succ_idx, &succ) in func.block_succs(block).iter().enumerate() {
                if func.block_preds(succ).len() == 1 {
                    for (&blockparam_out, &blockparam_in) in func
                        .branch_blockparams(block, block_insn.last(), succ_idx)
                        .iter()
                        .zip(func.block_params(succ))
                    {
                        self.values.alias_blockparam(blockparam_out, blockparam_in);
                    }
                }
            }
        }

        for i in 0..func.num_insts() {
            let inst = crate::Inst::new(i);
            let operands_start = self.operands.len();
            let inst_data = InstData {
                block: Block::ENTRY_BLOCK, // This is filled in later.
                operands: operands_start as u32,
                clobbers: self.clobbers.len() as u32,
                terminator_kind: None, // This is filled in later.
            };
            self.insts.push(inst_data);
            let mut have_late_use = false;
            let mut fixed_def = PRegSet::empty();
            self.operands
                .extend(func.inst_operands(inst).iter().map(|op| {
                    if let Some(preg) = op.as_fixed_nonallocatable() {
                        return Operand::fixed_nonallocatable(PhysReg::new(preg.index()));
                    }
                    let value = self.values.map(op.vreg());
                    let kind = match (op.kind(), op.pos()) {
                        (crate::OperandKind::Def, crate::OperandPos::Early) => {
                            OperandKind::EarlyDef(value)
                        }
                        (crate::OperandKind::Def, crate::OperandPos::Late) => {
                            OperandKind::Def(value)
                        }
                        (crate::OperandKind::Use, crate::OperandPos::Early) => {
                            OperandKind::Use(value)
                        }
                        (crate::OperandKind::Use, crate::OperandPos::Late) => {
                            have_late_use = true;
                            OperandKind::Use(value)
                        }
                    };
                    let bank = class_to_bank(op.class());
                    let constraint = match op.constraint() {
                        crate::OperandConstraint::Any => {
                            OperandConstraint::Class(ClassKind::Any.to_class(bank))
                        }
                        crate::OperandConstraint::Reg => {
                            OperandConstraint::Class(ClassKind::Reg.to_class(bank))
                        }
                        crate::OperandConstraint::FixedReg(preg) => {
                            if op.kind() == crate::OperandKind::Def {
                                fixed_def.add(preg);
                            }
                            OperandConstraint::Fixed(PhysReg::new(preg.index()))
                        }
                        crate::OperandConstraint::Reuse(reuse) => {
                            match func.inst_operands(inst)[reuse].constraint() {
                                crate::OperandConstraint::FixedReg(preg) => {
                                    OperandConstraint::Fixed(PhysReg::new(preg.index()))
                                }
                                _ => OperandConstraint::Reuse(reuse),
                            }
                        }
                    };
                    Operand::new(kind, constraint)
                }));
            if have_late_use {
                // regalloc3 doesn't support late use so instead force all defs
                // to be early defs if there is a late use in the instruction.
                for op in &mut self.operands[operands_start..] {
                    if let OperandKind::Def(value) = op.kind() {
                        *op = Operand::new(OperandKind::EarlyDef(value), op.constraint());
                    }
                }
            }
            for preg in func.inst_clobbers(inst) {
                // regalloc3 doesn't allow clobbers to overlap with fixed defs.
                if fixed_def.contains(preg) {
                    continue;
                }
                fixed_def.add(preg);
                self.clobbers.push(RegUnit::new(preg.index()));
            }
        }
        for i in 0..func.num_blocks() {
            let block = crate::Block::new(i);
            let block_insts = func.block_insns(block);
            let idom = if cfg.domtree[i].is_valid() {
                Some(Block::new(cfg.domtree[i].index())).into()
            } else {
                None.into()
            };
            let block_data = BlockData {
                insts: InstRange::new(
                    Inst::new(block_insts.first().index()),
                    Inst::new(block_insts.last().next().index()),
                ),
                preds: self.preds.len() as u32,
                succs: self.succs.len() as u32,
                block_params_in: self.block_params_in.len() as u32,
                block_params_out: self.block_params_out.len() as u32,
                idom,
                frequency: (0..cfg.approx_loop_depth[i]).fold(1000.0, |a, _| a * 4.0),
            };
            let ra3_block = self.blocks.push(block_data);
            for inst in block_insts.iter() {
                self.insts[Inst::new(inst.index())].block = ra3_block;
            }
            self.insts[Inst::new(block_insts.last().index())].terminator_kind =
                Some(match func.block_succs(block) {
                    &[] => TerminatorKind::Ret,
                    &[succ] if func.block_preds(succ).len() != 1 => {
                        self.block_params_out.extend(
                            func.branch_blockparams(block, block_insts.last(), 0)
                                .iter()
                                .map(|&vreg| self.values.map(vreg)),
                        );
                        TerminatorKind::Jump
                    }
                    _ => TerminatorKind::Branch,
                });
            if func.block_preds(block).len() != 1 {
                self.block_params_in.extend(
                    func.block_params(block)
                        .iter()
                        .map(|&vreg| self.values.map(vreg)),
                );
            };
            self.preds.extend(
                func.block_preds(block)
                    .iter()
                    .map(|b| Block::new(b.index())),
            );
            self.succs.extend(
                func.block_succs(block)
                    .iter()
                    .map(|b| Block::new(b.index())),
            );
        }
    }
}

#[derive(Default)]
pub struct Regalloc3Ctx {
    adapter: FunctionAdapter,
    reginfo: RegInfoImpl,
    regalloc: RegisterAllocator,
}

impl Regalloc3Ctx {
    pub fn run<F: crate::Function>(
        &mut self,
        func: &F,
        mach_env: &MachineEnv,
        cfg: &mut CFGInfo,
        cfginfo_ctx: &mut CFGInfoCtx,
        output: &mut Output,
    ) -> Result<(), crate::RegAllocError> {
        cfg.init(func, cfginfo_ctx)?;
        self.reginfo.build(func, mach_env);
        self.adapter.build(func, cfg);

        if cfg!(debug_assertions) {
            regalloc3::debug_utils::validate_function(&self.adapter, &self.reginfo).unwrap();
        }

        let ra3_output = self
            .regalloc
            .allocate_registers(&self.adapter, &self.reginfo, &Default::default())
            .map_err(|err| match err {
                RegAllocError::TooManyLiveRegs => crate::RegAllocError::TooManyLiveRegs,
                RegAllocError::FunctionTooBig => panic!("Input function too big for regalloc3"),
                err => panic!("regalloc3 error: {:?}", err),
            })?;

        output.num_spillslots = ra3_output.stack_layout().spillslot_area_size() as usize;
        output.edits.clear();
        output.allocs.clear();
        output.inst_alloc_offsets.clear();
        output.debug_locations.clear();
        output.stats = Stats::default();

        if cfg!(debug_assertions) {
            regalloc3::debug_utils::check_output(&ra3_output).unwrap();
        }

        let map_allocation = |alloc: Allocation| match alloc.kind() {
            AllocationKind::PhysReg(reg) => crate::Allocation::reg(PReg::from_index(reg.index())),
            AllocationKind::SpillSlot(slot) => crate::Allocation::stack(crate::SpillSlot::new(
                ra3_output.stack_layout().spillslot_offset(slot) as usize,
            )),
        };

        for block in self.adapter.blocks() {
            for inst in ra3_output.output_insts(block) {
                match inst {
                    OutputInst::Inst {
                        inst,
                        operand_allocs,
                    } => {
                        debug_assert_eq!(inst.index(), output.inst_alloc_offsets.len());
                        output.inst_alloc_offsets.push(output.allocs.len() as u32);
                        output
                            .allocs
                            .extend(operand_allocs.iter().map(|&alloc| map_allocation(alloc)));
                    }
                    OutputInst::Rematerialize { value: _, to: _ } => unreachable!(),
                    OutputInst::Move { from, to, value: _ } => {
                        output.edits.push((
                            ProgPoint::before(crate::Inst(output.inst_alloc_offsets.len() as u32)),
                            Edit::Move {
                                from: map_allocation(from),
                                to: map_allocation(to),
                            },
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}
