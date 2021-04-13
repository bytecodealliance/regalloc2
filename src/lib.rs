/*
 * The fellowing license applies to this file, which derives many
 * details (register and constraint definitions, for example) from the
 * files `BacktrackingAllocator.h`, `BacktrackingAllocator.cpp`,
 * `LIR.h`, and possibly definitions in other related files in
 * `js/src/jit/`:
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#![allow(dead_code)]

pub mod bitvec;
pub mod cfg;
pub mod domtree;
pub mod ion;
pub mod moves;
pub mod postorder;
pub mod ssa;

#[macro_use]
pub mod index;
pub use index::{Block, Inst, InstRange, InstRangeIter};

pub mod checker;
pub mod fuzzing;

/// Register classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegClass {
    Int = 0,
    Float = 1,
}

/// A physical register. Contains a physical register number and a class.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PReg(u8, RegClass);

impl PReg {
    pub const MAX_BITS: usize = 5;
    pub const MAX: usize = (1 << Self::MAX_BITS) - 1;

    /// Create a new PReg. The `hw_enc` range is 6 bits.
    #[inline(always)]
    pub fn new(hw_enc: usize, class: RegClass) -> Self {
        assert!(hw_enc <= Self::MAX);
        PReg(hw_enc as u8, class)
    }

    /// The physical register number, as encoded by the ISA for the particular register class.
    #[inline(always)]
    pub fn hw_enc(self) -> usize {
        self.0 as usize
    }

    /// The register class.
    #[inline(always)]
    pub fn class(self) -> RegClass {
        self.1
    }

    /// Get an index into the (not necessarily contiguous) index space of
    /// all physical registers. Allows one to maintain an array of data for
    /// all PRegs and index it efficiently.
    #[inline(always)]
    pub fn index(self) -> usize {
        ((self.1 as u8 as usize) << 6) | (self.0 as usize)
    }

    #[inline(always)]
    pub fn from_index(index: usize) -> Self {
        let class = (index >> 6) & 1;
        let class = match class {
            0 => RegClass::Int,
            1 => RegClass::Float,
            _ => unreachable!(),
        };
        let index = index & Self::MAX;
        PReg::new(index, class)
    }

    #[inline(always)]
    pub fn invalid() -> Self {
        PReg::new(Self::MAX, RegClass::Int)
    }
}

impl std::fmt::Debug for PReg {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "PReg(hw = {}, class = {:?}, index = {})",
            self.hw_enc(),
            self.class(),
            self.index()
        )
    }
}

impl std::fmt::Display for PReg {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let class = match self.class() {
            RegClass::Int => "i",
            RegClass::Float => "f",
        };
        write!(f, "p{}{}", self.hw_enc(), class)
    }
}

/// A virtual register. Contains a virtual register number and a class.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VReg(u32);

impl VReg {
    pub const MAX_BITS: usize = 20;
    pub const MAX: usize = (1 << Self::MAX_BITS) - 1;

    #[inline(always)]
    pub fn new(virt_reg: usize, class: RegClass) -> Self {
        assert!(virt_reg <= Self::MAX);
        VReg(((virt_reg as u32) << 1) | (class as u8 as u32))
    }

    #[inline(always)]
    pub fn vreg(self) -> usize {
        (self.0 >> 1) as usize
    }

    #[inline(always)]
    pub fn class(self) -> RegClass {
        match self.0 & 1 {
            0 => RegClass::Int,
            1 => RegClass::Float,
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn invalid() -> Self {
        VReg::new(Self::MAX, RegClass::Int)
    }
}

impl std::fmt::Debug for VReg {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "VReg(vreg = {}, class = {:?})",
            self.vreg(),
            self.class()
        )
    }
}

impl std::fmt::Display for VReg {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "v{}", self.vreg())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpillSlot(u32);

impl SpillSlot {
    #[inline(always)]
    pub fn new(slot: usize, class: RegClass) -> Self {
        assert!(slot < (1 << 24));
        SpillSlot((slot as u32) | (class as u8 as u32) << 24)
    }
    #[inline(always)]
    pub fn index(self) -> usize {
        (self.0 & 0x00ffffff) as usize
    }
    #[inline(always)]
    pub fn class(self) -> RegClass {
        match (self.0 >> 24) as u8 {
            0 => RegClass::Int,
            1 => RegClass::Float,
            _ => unreachable!(),
        }
    }
    #[inline(always)]
    pub fn plus(self, offset: usize) -> Self {
        SpillSlot::new(self.index() + offset, self.class())
    }
}

impl std::fmt::Display for SpillSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "stack{}", self.index())
    }
}

/// An `Operand` encodes everything about a mention of a register in
/// an instruction: virtual register number, and any constraint/policy
/// that applies to the register at this program point.
///
/// An Operand may be a use or def (this corresponds to `LUse` and
/// `LAllocation` in Ion).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    /// Bit-pack into 31 bits. This allows a `Reg` to encode an
    /// `Operand` or an `Allocation` in 32 bits.
    ///
    /// op-or-alloc:1 pos:2 kind:1 policy:2 class:1 preg:5 vreg:20
    bits: u32,
}

impl Operand {
    #[inline(always)]
    pub fn new(vreg: VReg, policy: OperandPolicy, kind: OperandKind, pos: OperandPos) -> Self {
        let (preg_field, policy_field): (u32, u32) = match policy {
            OperandPolicy::Any => (0, 0),
            OperandPolicy::Reg => (0, 1),
            OperandPolicy::FixedReg(preg) => {
                assert_eq!(preg.class(), vreg.class());
                (preg.hw_enc() as u32, 2)
            }
            OperandPolicy::Reuse(which) => {
                assert!(which <= PReg::MAX);
                (which as u32, 3)
            }
        };
        let class_field = vreg.class() as u8 as u32;
        let pos_field = pos as u8 as u32;
        let kind_field = kind as u8 as u32;
        Operand {
            bits: vreg.vreg() as u32
                | (preg_field << 20)
                | (class_field << 25)
                | (policy_field << 26)
                | (kind_field << 28)
                | (pos_field << 29),
        }
    }

    #[inline(always)]
    pub fn reg_use(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandPolicy::Reg,
            OperandKind::Use,
            OperandPos::Before,
        )
    }
    #[inline(always)]
    pub fn reg_use_at_end(vreg: VReg) -> Self {
        Operand::new(vreg, OperandPolicy::Reg, OperandKind::Use, OperandPos::Both)
    }
    #[inline(always)]
    pub fn reg_def(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandPolicy::Reg,
            OperandKind::Def,
            OperandPos::After,
        )
    }
    #[inline(always)]
    pub fn reg_def_at_start(vreg: VReg) -> Self {
        Operand::new(vreg, OperandPolicy::Reg, OperandKind::Def, OperandPos::Both)
    }
    #[inline(always)]
    pub fn reg_temp(vreg: VReg) -> Self {
        Operand::new(vreg, OperandPolicy::Reg, OperandKind::Def, OperandPos::Both)
    }
    #[inline(always)]
    pub fn reg_reuse_def(vreg: VReg, idx: usize) -> Self {
        Operand::new(
            vreg,
            OperandPolicy::Reuse(idx),
            OperandKind::Def,
            OperandPos::Both,
        )
    }
    #[inline(always)]
    pub fn reg_fixed_use(vreg: VReg, preg: PReg) -> Self {
        Operand::new(
            vreg,
            OperandPolicy::FixedReg(preg),
            OperandKind::Use,
            OperandPos::Before,
        )
    }
    #[inline(always)]
    pub fn reg_fixed_def(vreg: VReg, preg: PReg) -> Self {
        Operand::new(
            vreg,
            OperandPolicy::FixedReg(preg),
            OperandKind::Def,
            OperandPos::After,
        )
    }

    #[inline(always)]
    pub fn vreg(self) -> VReg {
        let vreg_idx = ((self.bits as usize) & VReg::MAX) as usize;
        VReg::new(vreg_idx, self.class())
    }

    #[inline(always)]
    pub fn class(self) -> RegClass {
        let class_field = (self.bits >> 25) & 1;
        match class_field {
            0 => RegClass::Int,
            1 => RegClass::Float,
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn kind(self) -> OperandKind {
        let kind_field = (self.bits >> 28) & 1;
        match kind_field {
            0 => OperandKind::Def,
            1 => OperandKind::Use,
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn pos(self) -> OperandPos {
        let pos_field = (self.bits >> 29) & 3;
        match pos_field {
            0 => OperandPos::Before,
            1 => OperandPos::After,
            2 => OperandPos::Both,
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn policy(self) -> OperandPolicy {
        let policy_field = (self.bits >> 26) & 3;
        let preg_field = ((self.bits >> 20) as usize) & PReg::MAX;
        match policy_field {
            0 => OperandPolicy::Any,
            1 => OperandPolicy::Reg,
            2 => OperandPolicy::FixedReg(PReg::new(preg_field, self.class())),
            3 => OperandPolicy::Reuse(preg_field),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn bits(self) -> u32 {
        self.bits
    }

    #[inline(always)]
    pub fn from_bits(bits: u32) -> Self {
        Operand { bits }
    }
}

impl std::fmt::Debug for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Operand(vreg = {:?}, class = {:?}, kind = {:?}, pos = {:?}, policy = {:?})",
            self.vreg().vreg(),
            self.class(),
            self.kind(),
            self.pos(),
            self.policy()
        )
    }
}

impl std::fmt::Display for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{:?}@{:?}: {} {}",
            self.kind(),
            self.pos(),
            self.vreg(),
            self.policy()
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandPolicy {
    /// Any location is fine (register or stack slot).
    Any,
    /// Operand must be in a register. Register is read-only for Uses.
    Reg,
    /// Operand must be in a fixed register.
    FixedReg(PReg),
    /// On defs only: reuse a use's register. Which use is given by `preg` field.
    Reuse(usize),
}

impl std::fmt::Display for OperandPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "any"),
            Self::Reg => write!(f, "reg"),
            Self::FixedReg(preg) => write!(f, "fixed({})", preg),
            Self::Reuse(idx) => write!(f, "reuse({})", idx),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandKind {
    Def = 0,
    Use = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandPos {
    Before = 0,
    After = 1,
    Both = 2,
}

/// An Allocation represents the end result of regalloc for an
/// Operand.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Allocation {
    /// Bit-pack in 31 bits:
    ///
    /// op-or-alloc:1 kind:2  index:29
    bits: u32,
}

impl std::fmt::Debug for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Allocation(kind = {:?}, index = {})",
            self.kind(),
            self.index()
        )
    }
}

impl std::fmt::Display for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.kind() {
            AllocationKind::None => write!(f, "none"),
            AllocationKind::Reg => write!(f, "{}", self.as_reg().unwrap()),
            AllocationKind::Stack => write!(f, "{}", self.as_stack().unwrap()),
        }
    }
}

impl Allocation {
    #[inline(always)]
    pub(crate) fn new(kind: AllocationKind, index: usize) -> Self {
        Self {
            bits: ((kind as u8 as u32) << 29) | (index as u32),
        }
    }

    #[inline(always)]
    pub fn none() -> Allocation {
        Allocation::new(AllocationKind::None, 0)
    }

    #[inline(always)]
    pub fn reg(preg: PReg) -> Allocation {
        Allocation::new(AllocationKind::Reg, preg.index())
    }

    #[inline(always)]
    pub fn stack(slot: SpillSlot) -> Allocation {
        Allocation::new(AllocationKind::Stack, slot.0 as usize)
    }

    #[inline(always)]
    pub fn kind(self) -> AllocationKind {
        match (self.bits >> 29) & 3 {
            0 => AllocationKind::None,
            1 => AllocationKind::Reg,
            2 => AllocationKind::Stack,
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn index(self) -> usize {
        (self.bits & ((1 << 29) - 1)) as usize
    }

    #[inline(always)]
    pub fn as_reg(self) -> Option<PReg> {
        if self.kind() == AllocationKind::Reg {
            Some(PReg::from_index(self.index()))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn as_stack(self) -> Option<SpillSlot> {
        if self.kind() == AllocationKind::Stack {
            Some(SpillSlot(self.index() as u32))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn bits(self) -> u32 {
        self.bits
    }

    #[inline(always)]
    pub fn from_bits(bits: u32) -> Self {
        Self { bits }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum AllocationKind {
    None = 0,
    Reg = 1,
    Stack = 2,
}

impl Allocation {
    #[inline(always)]
    pub fn class(self) -> RegClass {
        match self.kind() {
            AllocationKind::None => panic!("Allocation::None has no class"),
            AllocationKind::Reg => self.as_reg().unwrap().class(),
            AllocationKind::Stack => self.as_stack().unwrap().class(),
        }
    }
}

/// A trait defined by the regalloc client to provide access to its
/// machine-instruction / CFG representation.
pub trait Function {
    // -------------
    // CFG traversal
    // -------------

    /// How many instructions are there?
    fn insts(&self) -> usize;

    /// How many blocks are there?
    fn blocks(&self) -> usize;

    /// Get the index of the entry block.
    fn entry_block(&self) -> Block;

    /// Provide the range of instruction indices contained in each block.
    fn block_insns(&self, block: Block) -> InstRange;

    /// Get CFG successors for a given block.
    fn block_succs(&self, block: Block) -> &[Block];

    /// Get the CFG predecessors for a given block.
    fn block_preds(&self, block: Block) -> &[Block];

    /// Get the block parameters for a given block.
    fn block_params(&self, block: Block) -> &[VReg];

    /// Determine whether an instruction is a call instruction. This is used
    /// only for splitting heuristics.
    fn is_call(&self, insn: Inst) -> bool;

    /// Determine whether an instruction is a return instruction.
    fn is_ret(&self, insn: Inst) -> bool;

    /// Determine whether an instruction is the end-of-block
    /// branch. If so, its operands *must* be the block parameters for
    /// each of its block's `block_succs` successor blocks, in order.
    fn is_branch(&self, insn: Inst) -> bool;

    /// Determine whether an instruction is a safepoint and requires a stackmap.
    fn is_safepoint(&self, insn: Inst) -> bool;

    /// Determine whether an instruction is a move; if so, return the
    /// vregs for (src, dst).
    fn is_move(&self, insn: Inst) -> Option<(VReg, VReg)>;

    // --------------------------
    // Instruction register slots
    // --------------------------

    /// Get the Operands for an instruction.
    fn inst_operands(&self, insn: Inst) -> &[Operand];

    /// Get the clobbers for an instruction.
    fn inst_clobbers(&self, insn: Inst) -> &[PReg];

    /// Get the precise number of `VReg` in use in this function, to allow
    /// preallocating data structures. This number *must* be a correct
    /// lower-bound, otherwise invalid index failures may happen; it is of
    /// course better if it is exact.
    fn num_vregs(&self) -> usize;

    // --------------
    // Spills/reloads
    // --------------

    /// How many logical spill slots does the given regclass require?  E.g., on
    /// a 64-bit machine, spill slots may nominally be 64-bit words, but a
    /// 128-bit vector value will require two slots.  The regalloc will always
    /// align on this size.
    ///
    /// This passes the associated virtual register to the client as well,
    /// because the way in which we spill a real register may depend on the
    /// value that we are using it for. E.g., if a machine has V128 registers
    /// but we also use them for F32 and F64 values, we may use a different
    /// store-slot size and smaller-operand store/load instructions for an F64
    /// than for a true V128.
    fn spillslot_size(&self, regclass: RegClass, for_vreg: VReg) -> usize;

    /// When providing a spillslot number for a multi-slot spillslot,
    /// do we provide the first or the last? This is usually related
    /// to which direction the stack grows and different clients may
    /// have different preferences.
    fn multi_spillslot_named_by_last_slot(&self) -> bool {
        false
    }
}

/// A position before or after an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum InstPosition {
    Before = 0,
    After = 1,
}

/// A program point: a single point before or after a given instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgPoint {
    pub inst: Inst,
    pub pos: InstPosition,
}

impl ProgPoint {
    pub fn before(inst: Inst) -> Self {
        Self {
            inst,
            pos: InstPosition::Before,
        }
    }

    pub fn after(inst: Inst) -> Self {
        Self {
            inst,
            pos: InstPosition::After,
        }
    }

    pub fn next(self) -> ProgPoint {
        match self.pos {
            InstPosition::Before => ProgPoint {
                inst: self.inst,
                pos: InstPosition::After,
            },
            InstPosition::After => ProgPoint {
                inst: self.inst.next(),
                pos: InstPosition::Before,
            },
        }
    }

    pub fn prev(self) -> ProgPoint {
        match self.pos {
            InstPosition::Before => ProgPoint {
                inst: self.inst.prev(),
                pos: InstPosition::After,
            },
            InstPosition::After => ProgPoint {
                inst: self.inst,
                pos: InstPosition::Before,
            },
        }
    }

    pub fn to_index(self) -> u32 {
        debug_assert!(self.inst.index() <= ((1 << 31) - 1));
        ((self.inst.index() as u32) << 1) | (self.pos as u8 as u32)
    }

    pub fn from_index(index: u32) -> Self {
        let inst = Inst::new((index >> 1) as usize);
        let pos = match index & 1 {
            0 => InstPosition::Before,
            1 => InstPosition::After,
            _ => unreachable!(),
        };
        Self { inst, pos }
    }
}

/// An instruction to insert into the program to perform some data movement.
#[derive(Clone, Debug)]
pub enum Edit {
    /// Move one allocation to another. Each allocation may be a
    /// register or a stack slot (spillslot).
    Move { from: Allocation, to: Allocation },
    /// Define blockparams' locations. Note that this is not typically
    /// turned into machine code, but can be useful metadata (e.g. for
    /// the checker).
    BlockParams {
        vregs: Vec<VReg>,
        allocs: Vec<Allocation>,
    },
}

/// A machine envrionment tells the register allocator which registers
/// are available to allocate and what register may be used as a
/// scratch register for each class, and some other miscellaneous info
/// as well.
#[derive(Clone, Debug)]
pub struct MachineEnv {
    regs: Vec<PReg>,
    regs_by_class: Vec<Vec<PReg>>,
    scratch_by_class: Vec<PReg>,
}

/// The output of the register allocator.
#[derive(Clone, Debug)]
pub struct Output {
    /// How many spillslots are needed in the frame?
    pub num_spillslots: usize,
    /// Edits (insertions or removals). Guaranteed to be sorted by
    /// program point.
    pub edits: Vec<(ProgPoint, Edit)>,
    /// Allocations for each operand. Mapping from instruction to
    /// allocations provided by `inst_alloc_offsets` below.
    pub allocs: Vec<Allocation>,
    /// Allocation offset in `allocs` for each instruction.
    pub inst_alloc_offsets: Vec<u32>,

    /// Internal stats from the allocator.
    pub stats: ion::Stats,
}

impl Output {
    pub fn inst_allocs(&self, inst: Inst) -> &[Allocation] {
        let start = self.inst_alloc_offsets[inst.index()] as usize;
        let end = if inst.index() + 1 == self.inst_alloc_offsets.len() {
            self.allocs.len()
        } else {
            self.inst_alloc_offsets[inst.index() + 1] as usize
        };
        &self.allocs[start..end]
    }
}

/// An error that prevents allocation.
#[derive(Clone, Debug)]
pub enum RegAllocError {
    /// Invalid SSA for given vreg at given inst: multiple defs or
    /// illegal use. `inst` may be `Inst::invalid()` if this concerns
    /// a block param.
    SSA(VReg, Inst),
    /// Invalid basic block: does not end in branch/ret, or contains a
    /// branch/ret in the middle.
    BB(Block),
    /// Invalid branch: operand count does not match sum of block
    /// params of successor blocks.
    Branch(Inst),
}

impl std::fmt::Display for RegAllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for RegAllocError {}

pub fn run<F: Function>(func: &F, env: &MachineEnv) -> Result<Output, RegAllocError> {
    ion::run(func, env)
}
