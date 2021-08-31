/*
 * The following license applies to this file, which derives many
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
pub(crate) mod cfg;
pub(crate) mod domtree;
pub(crate) mod ion;
pub(crate) mod moves;
pub(crate) mod postorder;
pub(crate) mod ssa;

#[macro_use]
mod index;
pub use index::{Block, Inst, InstRange, InstRangeIter};

pub mod checker;

#[cfg(feature = "fuzzing")]
pub mod fuzzing;

/// Register classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegClass {
    Int = 0,
    Float = 1,
}

/// A physical register. Contains a physical register number and a class.
///
/// The `hw_enc` field contains the physical register number and is in
/// a logically separate index space per class; in other words, Int
/// register 0 is different than Float register 0.
///
/// Because of bit-packed encodings throughout the implementation,
/// `hw_enc` must fit in 5 bits, i.e., at most 32 registers per class.
///
/// The value returned by `index()`, in contrast, is in a single index
/// space shared by all classes, in order to enable uniform reasoning
/// about physical registers. This is done by putting the class bit at
/// the MSB, or equivalently, declaring that indices 0..31 are the 32
/// integer registers and indices 32..63 are the 32 float registers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PReg {
    hw_enc: u8,
    class: RegClass,
}

impl PReg {
    pub const MAX_BITS: usize = 5;
    pub const MAX: usize = (1 << Self::MAX_BITS) - 1;
    pub const MAX_INDEX: usize = 1 << (Self::MAX_BITS + 1); // including RegClass bit

    /// Create a new PReg. The `hw_enc` range is 6 bits.
    #[inline(always)]
    pub const fn new(hw_enc: usize, class: RegClass) -> Self {
        // We don't have const panics yet (rust-lang/rust#85194) so we
        // need to use a little indexing trick here. We unfortunately
        // can't use the `static-assertions` crate because we need
        // this to work both for const `hw_enc` and for runtime
        // values.
        const HW_ENC_MUST_BE_IN_BOUNDS: &[bool; PReg::MAX + 1] = &[true; PReg::MAX + 1];
        let _ = HW_ENC_MUST_BE_IN_BOUNDS[hw_enc];

        PReg {
            hw_enc: hw_enc as u8,
            class,
        }
    }

    /// The physical register number, as encoded by the ISA for the particular register class.
    #[inline(always)]
    pub fn hw_enc(self) -> usize {
        let hw_enc = self.hw_enc as usize;
        hw_enc
    }

    /// The register class.
    #[inline(always)]
    pub fn class(self) -> RegClass {
        self.class
    }

    /// Get an index into the (not necessarily contiguous) index space of
    /// all physical registers. Allows one to maintain an array of data for
    /// all PRegs and index it efficiently.
    #[inline(always)]
    pub fn index(self) -> usize {
        ((self.class as u8 as usize) << 5) | (self.hw_enc as usize)
    }

    #[inline(always)]
    pub fn from_index(index: usize) -> Self {
        let class = (index >> 5) & 1;
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
pub struct VReg {
    bits: u32,
}

impl VReg {
    pub const MAX_BITS: usize = 20;
    pub const MAX: usize = (1 << Self::MAX_BITS) - 1;

    #[inline(always)]
    pub const fn new(virt_reg: usize, class: RegClass) -> Self {
        // See comment in `PReg::new()`: we are emulating a const
        // assert here until const panics are stable.
        const VIRT_REG_MUST_BE_IN_BOUNDS: &[bool; VReg::MAX + 1] = &[true; VReg::MAX + 1];
        let _ = VIRT_REG_MUST_BE_IN_BOUNDS[virt_reg];

        VReg {
            bits: ((virt_reg as u32) << 1) | (class as u8 as u32),
        }
    }

    #[inline(always)]
    pub fn vreg(self) -> usize {
        let vreg = (self.bits >> 1) as usize;
        vreg
    }

    #[inline(always)]
    pub fn class(self) -> RegClass {
        match self.bits & 1 {
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
pub struct SpillSlot {
    bits: u32,
}

impl SpillSlot {
    #[inline(always)]
    pub fn new(slot: usize, class: RegClass) -> Self {
        assert!(slot < (1 << 24));
        SpillSlot {
            bits: (slot as u32) | (class as u8 as u32) << 24,
        }
    }
    #[inline(always)]
    pub fn index(self) -> usize {
        (self.bits & 0x00ffffff) as usize
    }
    #[inline(always)]
    pub fn class(self) -> RegClass {
        match (self.bits >> 24) as u8 {
            0 => RegClass::Int,
            1 => RegClass::Float,
            _ => unreachable!(),
        }
    }
    #[inline(always)]
    pub fn plus(self, offset: usize) -> Self {
        SpillSlot::new(self.index() + offset, self.class())
    }

    #[inline(always)]
    pub fn invalid() -> Self {
        SpillSlot { bits: 0xffff_ffff }
    }
    #[inline(always)]
    pub fn is_invalid(self) -> bool {
        self == Self::invalid()
    }
    #[inline(always)]
    pub fn is_valid(self) -> bool {
        self != Self::invalid()
    }
}

impl std::fmt::Display for SpillSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "stack{}", self.index())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandConstraint {
    /// Any location is fine (register or stack slot).
    Any,
    /// Operand must be in a register. Register is read-only for Uses.
    Reg,
    /// Operand must be on the stack.
    Stack,
    /// Operand must be in a fixed register.
    FixedReg(PReg),
    /// On defs only: reuse a use's register.
    Reuse(usize),
}

impl std::fmt::Display for OperandConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "any"),
            Self::Reg => write!(f, "reg"),
            Self::Stack => write!(f, "stack"),
            Self::FixedReg(preg) => write!(f, "fixed({})", preg),
            Self::Reuse(idx) => write!(f, "reuse({})", idx),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandKind {
    Def = 0,
    Mod = 1,
    Use = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandPos {
    Before = 0,
    After = 1,
}

/// An `Operand` encodes everything about a mention of a register in
/// an instruction: virtual register number, and any constraint that
/// applies to the register at this program point.
///
/// An Operand may be a use or def (this corresponds to `LUse` and
/// `LAllocation` in Ion).
///
/// Generally, regalloc2 considers operands to have their effects at
/// one of two program points that surround an instruction: "Before"
/// or "After". All operands at a given program-point are assigned
/// non-conflicting locations based on their constraints. Each operand
/// has a "kind", one of use/def/mod, corresponding to
/// read/write/read-write, respectively.
///
/// Usually, an instruction's inputs will be uses-at-Before and
/// outputs will be defs-at-After, though there are valid use-cases
/// for other combinations too. For example, a single "instruction"
/// seen by the regalloc that lowers into multiple machine
/// instructions and reads some of its inputs after it starts to write
/// outputs must either make those input(s) uses-at-After or those
/// output(s) defs-at-Before so that the conflict (overlap) is
/// properly accounted for. See comments on the constructors below for
/// more.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Operand {
    /// Bit-pack into 32 bits.
    ///
    /// constraint:3 kind:2 pos:1 class:1 preg:5 vreg:20
    ///
    /// where `constraint` is an `OperandConstraint`, `kind` is an
    /// `OperandKind`, `pos` is an `OperandPos`, `class` is a
    /// `RegClass`, `preg` is a `PReg` or an index for a reused-input
    /// constraint, and `vreg` is a vreg index.
    bits: u32,
}

impl Operand {
    #[inline(always)]
    pub fn new(
        vreg: VReg,
        constraint: OperandConstraint,
        kind: OperandKind,
        pos: OperandPos,
    ) -> Self {
        let (preg_field, constraint_field): (u32, u32) = match constraint {
            OperandConstraint::Any => (0, 0),
            OperandConstraint::Reg => (0, 1),
            OperandConstraint::Stack => (0, 2),
            OperandConstraint::FixedReg(preg) => {
                assert_eq!(preg.class(), vreg.class());
                (preg.hw_enc() as u32, 3)
            }
            OperandConstraint::Reuse(which) => {
                assert!(which <= PReg::MAX);
                (which as u32, 4)
            }
        };
        let class_field = vreg.class() as u8 as u32;
        let pos_field = pos as u8 as u32;
        let kind_field = kind as u8 as u32;
        Operand {
            bits: vreg.vreg() as u32
                | (preg_field << 20)
                | (class_field << 25)
                | (pos_field << 26)
                | (kind_field << 27)
                | (constraint_field << 29),
        }
    }

    /// Create an `Operand` that designates a use of a VReg that must
    /// be in a register, and that is used at the "before" point,
    /// i.e., can be overwritten by a result.
    #[inline(always)]
    pub fn reg_use(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Reg,
            OperandKind::Use,
            OperandPos::Before,
        )
    }

    /// Create an `Operand` that designates a use of a VReg that must
    /// be in a register, and that is used up until the "after" point,
    /// i.e., must not conflict with any results.
    #[inline(always)]
    pub fn reg_use_at_end(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Reg,
            OperandKind::Use,
            OperandPos::After,
        )
    }

    /// Create an `Operand` that designates a definition of a VReg
    /// that must be in a register, and that occurs at the "after"
    /// point, i.e. may reuse a register that carried a use into this
    /// instruction.
    #[inline(always)]
    pub fn reg_def(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Reg,
            OperandKind::Def,
            OperandPos::After,
        )
    }

    /// Create an `Operand` that designates a definition of a VReg
    /// that must be in a register, and that occurs early at the
    /// "before" point, i.e., must not conflict with any input to the
    /// instruction.
    #[inline(always)]
    pub fn reg_def_at_start(vreg: VReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Reg,
            OperandKind::Def,
            OperandPos::Before,
        )
    }

    /// Create an `Operand` that designates a def (and use) of a
    /// temporary *within* the instruction. This register is assumed
    /// to be written by the instruction, and will not conflict with
    /// any input or output, but should not be used after the
    /// instruction completes.
    #[inline(always)]
    pub fn reg_temp(vreg: VReg) -> Self {
        // For now a temp is equivalent to a def-at-start operand,
        // which gives the desired semantics but does not enforce the
        // "not reused later" constraint.
        Operand::new(
            vreg,
            OperandConstraint::Reg,
            OperandKind::Def,
            OperandPos::Before,
        )
    }

    /// Create an `Operand` that designates a def of a vreg that must
    /// reuse the register assigned to an input to the
    /// instruction. The input is identified by `idx` (is the `idx`th
    /// `Operand` for the instruction) and must be constraint to a
    /// register, i.e., be the result of `Operand::reg_use(vreg)`.
    #[inline(always)]
    pub fn reg_reuse_def(vreg: VReg, idx: usize) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::Reuse(idx),
            OperandKind::Def,
            OperandPos::After,
        )
    }

    /// Create an `Operand` that designates a use of a vreg and
    /// ensures that it is placed in the given, fixed PReg at the
    /// use. It is guaranteed that the `Allocation` resulting for this
    /// operand will be `preg`.
    #[inline(always)]
    pub fn reg_fixed_use(vreg: VReg, preg: PReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::FixedReg(preg),
            OperandKind::Use,
            OperandPos::Before,
        )
    }

    /// Create an `Operand` that designates a def of a vreg and
    /// ensures that it is placed in the given, fixed PReg at the
    /// def. It is guaranteed that the `Allocation` resulting for this
    /// operand will be `preg`.
    #[inline(always)]
    pub fn reg_fixed_def(vreg: VReg, preg: PReg) -> Self {
        Operand::new(
            vreg,
            OperandConstraint::FixedReg(preg),
            OperandKind::Def,
            OperandPos::After,
        )
    }

    /// Get the virtual register designated by an operand. Every
    /// operand must name some virtual register, even if it constrains
    /// the operand to a fixed physical register as well; the vregs
    /// are used to track dataflow.
    #[inline(always)]
    pub fn vreg(self) -> VReg {
        let vreg_idx = ((self.bits as usize) & VReg::MAX) as usize;
        VReg::new(vreg_idx, self.class())
    }

    /// Get the register class used by this operand.
    #[inline(always)]
    pub fn class(self) -> RegClass {
        let class_field = (self.bits >> 25) & 1;
        match class_field {
            0 => RegClass::Int,
            1 => RegClass::Float,
            _ => unreachable!(),
        }
    }

    /// Get the "kind" of this operand: a definition (write), a use
    /// (read), or a "mod" / modify (a read followed by a write).
    #[inline(always)]
    pub fn kind(self) -> OperandKind {
        let kind_field = (self.bits >> 27) & 3;
        match kind_field {
            0 => OperandKind::Def,
            1 => OperandKind::Mod,
            2 => OperandKind::Use,
            _ => unreachable!(),
        }
    }

    /// Get the "position" of this operand, i.e., where its read
    /// and/or write occurs: either before the instruction executes,
    /// or after it does. Ordinarily, uses occur at "before" and defs
    /// at "after", though there are cases where this is not true.
    #[inline(always)]
    pub fn pos(self) -> OperandPos {
        let pos_field = (self.bits >> 26) & 1;
        match pos_field {
            0 => OperandPos::Before,
            1 => OperandPos::After,
            _ => unreachable!(),
        }
    }

    /// Get the "constraint" of this operand, i.e., what requirements
    /// its allocation must fulfill.
    #[inline(always)]
    pub fn constraint(self) -> OperandConstraint {
        let constraint_field = (self.bits >> 29) & 7;
        let preg_field = ((self.bits >> 20) as usize) & PReg::MAX;
        match constraint_field {
            0 => OperandConstraint::Any,
            1 => OperandConstraint::Reg,
            2 => OperandConstraint::Stack,
            3 => OperandConstraint::FixedReg(PReg::new(preg_field, self.class())),
            4 => OperandConstraint::Reuse(preg_field),
            _ => unreachable!(),
        }
    }

    /// Get the raw 32-bit encoding of this operand's fields.
    #[inline(always)]
    pub fn bits(self) -> u32 {
        self.bits
    }

    /// Construct an `Operand` from the raw 32-bit encoding returned
    /// from `bits()`.
    #[inline(always)]
    pub fn from_bits(bits: u32) -> Self {
        debug_assert!(bits >> 29 <= 4);
        Operand { bits }
    }
}

impl std::fmt::Debug for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match (self.kind(), self.pos()) {
            (OperandKind::Def, OperandPos::After)
            | (OperandKind::Mod | OperandKind::Use, OperandPos::Before) => {
                write!(f, "{:?}", self.kind())?;
            }
            _ => {
                write!(f, "{:?}@{:?}", self.kind(), self.pos())?;
            }
        }
        write!(
            f,
            ": {}{} {}",
            self.vreg(),
            match self.class() {
                RegClass::Int => "i",
                RegClass::Float => "f",
            },
            self.constraint()
        )
    }
}

/// An Allocation represents the end result of regalloc for an
/// Operand.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Allocation {
    /// Bit-pack in 32 bits.
    ///
    /// kind:3 unused:1 index:28
    bits: u32,
}

impl std::fmt::Debug for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
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
        assert!(index < (1 << 28));
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
        Allocation::new(AllocationKind::Stack, slot.bits as usize)
    }

    #[inline(always)]
    pub fn kind(self) -> AllocationKind {
        match (self.bits >> 29) & 7 {
            0 => AllocationKind::None,
            1 => AllocationKind::Reg,
            2 => AllocationKind::Stack,
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn is_none(self) -> bool {
        self.kind() == AllocationKind::None
    }

    #[inline(always)]
    pub fn is_reg(self) -> bool {
        self.kind() == AllocationKind::Reg
    }

    #[inline(always)]
    pub fn is_stack(self) -> bool {
        self.kind() == AllocationKind::Stack
    }

    #[inline(always)]
    pub fn index(self) -> usize {
        (self.bits & ((1 << 28) - 1)) as usize
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
            Some(SpillSlot {
                bits: self.index() as u32,
            })
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
        debug_assert!(bits >> 29 >= 5);
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
///
/// (This trait's design is inspired by, and derives heavily from, the
/// trait of the same name in regalloc.rs.)
pub trait Function {
    // -------------
    // CFG traversal
    // -------------

    /// How many instructions are there?
    fn num_insts(&self) -> usize;

    /// How many blocks are there?
    fn num_blocks(&self) -> usize;

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
    /// branch. If so, its operands at the indices given by
    /// `branch_blockparam_arg_offset()` below *must* be the block
    /// parameters for each of its block's `block_succs` successor
    /// blocks, in order.
    fn is_branch(&self, insn: Inst) -> bool;

    /// If `insn` is a branch at the end of `block`, returns the
    /// operand index at which outgoing blockparam arguments are
    /// found. Starting at this index, blockparam arguments for each
    /// successor block's blockparams, in order, must be found.
    ///
    /// It is an error if `self.inst_operands(insn).len() -
    /// self.branch_blockparam_arg_offset(insn)` is not exactly equal
    /// to the sum of blockparam counts for all successor blocks.
    fn branch_blockparam_arg_offset(&self, block: Block, insn: Inst) -> usize;

    /// Determine whether an instruction is a safepoint and requires a stackmap.
    ///
    /// Strictly speaking, these two parts (is a safepoint, requires a
    /// stackmap) are orthogonal. An instruction could want to see a
    /// stackmap of refs on the stack (without forcing them), or it
    /// could want all refs to be on the stack (without knowing where
    /// they are). Only the latter strictly follows from "is a
    /// safepoint". But in practice, both are true at the same time,
    /// so we combine the two notions: for regalloc2, a "safepoint
    /// instruction" is one that both forces refs onto the stack, and
    /// provides a stackmap indicating where they are.
    fn is_safepoint(&self, _: Inst) -> bool {
        false
    }

    /// Determine whether an instruction is a move; if so, return the
    /// Operands for (src, dst).
    fn is_move(&self, insn: Inst) -> Option<(Operand, Operand)>;

    // --------------------------
    // Instruction register slots
    // --------------------------

    /// Get the Operands for an instruction.
    fn inst_operands(&self, insn: Inst) -> &[Operand];

    /// Get the clobbers for an instruction; these are the registers
    /// that the instruction is known to overwrite, separate from its
    /// outputs described by its `Operand`s. This can be used to, for
    /// example, describe ABI-specified registers that are not
    /// preserved by a call instruction, or fixed physical registers
    /// written by an instruction but not used as a vreg output, or
    /// fixed physical registers used as temps within an instruction
    /// out of necessity. Every register written to by an instruction
    /// must either be described by an Operand of kind `Def` or `Mod`,
    /// or else must be a "clobber".
    fn inst_clobbers(&self, insn: Inst) -> &[PReg];

    /// Get the number of `VReg` in use in this function.
    fn num_vregs(&self) -> usize;

    /// Get the VRegs that are pointer/reference types. This has the
    /// following effects for each such vreg:
    ///
    /// - At all safepoint instructions, the vreg will be in a
    ///   SpillSlot, not in a register.
    /// - The vreg *may not* be used as a register operand on
    ///   safepoint instructions: this is because a vreg can only live
    ///   in one place at a time. The client should copy the value to an
    ///   integer-typed vreg and use this to pass a pointer as an input
    ///   to a safepoint instruction (such as a function call).
    /// - At all safepoint instructions, all live vregs' locations
    ///   will be included in a list in the `Output` below, so that
    ///   pointer-inspecting/updating functionality (such as a moving
    ///   garbage collector) may observe and edit their values.
    fn reftype_vregs(&self) -> &[VReg] {
        &[]
    }

    /// Get the VRegs for which we should generate value-location
    /// metadata for debugging purposes. This can be used to generate
    /// e.g. DWARF with valid prgram-point ranges for each value
    /// expression in a way that is more efficient than a post-hoc
    /// analysis of the allocator's output.
    ///
    /// Each tuple is (vreg, inclusive_start, exclusive_end,
    /// label). In the `Output` there will be (label, inclusive_start,
    /// exclusive_end, alloc)` tuples. The ranges may not exactly
    /// match -- specifically, the returned metadata may cover only a
    /// subset of the requested ranges -- if the value is not live for
    /// the entire requested ranges.
    fn debug_value_labels(&self) -> &[(Inst, Inst, VReg, u32)] {
        &[]
    }

    /// Is the given vreg pinned to a preg? If so, every use of the
    /// vreg is automatically assigned to the preg, and live-ranges of
    /// the vreg allocate the preg exclusively (are not spilled
    /// elsewhere). The user must take care not to have too many live
    /// pinned vregs such that allocation is no longer possible;
    /// liverange computation will check that this is the case (that
    /// there are enough remaining allocatable pregs of every class to
    /// hold all Reg-constrained operands).
    fn is_pinned_vreg(&self, _: VReg) -> Option<PReg> {
        None
    }

    /// Return a list of all pinned vregs.
    fn pinned_vregs(&self) -> &[VReg] {
        &[]
    }

    // --------------
    // Spills/reloads
    // --------------

    /// How many logical spill slots does the given regclass require?  E.g., on
    /// a 64-bit machine, spill slots may nominally be 64-bit words, but a
    /// 128-bit vector value will require two slots.  The regalloc will always
    /// align on this size.
    ///
    /// (This trait method's design and doc text derives from
    /// regalloc.rs' trait of the same name.)
    fn spillslot_size(&self, regclass: RegClass) -> usize;

    /// When providing a spillslot number for a multi-slot spillslot,
    /// do we provide the first or the last? This is usually related
    /// to which direction the stack grows and different clients may
    /// have different preferences.
    fn multi_spillslot_named_by_last_slot(&self) -> bool {
        false
    }
}

/// A position before or after an instruction at which we can make an
/// edit.
///
/// Note that this differs from `OperandPos` in that the former
/// describes specifically a constraint on an operand, while this
/// describes a program point. `OperandPos` could grow more options in
/// the future, for example if we decide that an "early write" or
/// "late read" phase makes sense, while `InstPosition` will always
/// describe these two insertion points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum InstPosition {
    Before = 0,
    After = 1,
}

/// A program point: a single point before or after a given instruction.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgPoint {
    bits: u32,
}

impl std::fmt::Debug for ProgPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "progpoint{}{}",
            self.inst().index(),
            match self.pos() {
                InstPosition::Before => "-pre",
                InstPosition::After => "-post",
            }
        )
    }
}

impl ProgPoint {
    #[inline(always)]
    pub fn new(inst: Inst, pos: InstPosition) -> Self {
        let bits = ((inst.0 as u32) << 1) | (pos as u8 as u32);
        Self { bits }
    }
    #[inline(always)]
    pub fn before(inst: Inst) -> Self {
        Self::new(inst, InstPosition::Before)
    }
    #[inline(always)]
    pub fn after(inst: Inst) -> Self {
        Self::new(inst, InstPosition::After)
    }
    #[inline(always)]
    pub fn inst(self) -> Inst {
        // Cast to i32 to do an arithmetic right-shift, which will
        // preserve an `Inst::invalid()` (which is -1, or all-ones).
        Inst::new(((self.bits as i32) >> 1) as usize)
    }
    #[inline(always)]
    pub fn pos(self) -> InstPosition {
        match self.bits & 1 {
            0 => InstPosition::Before,
            1 => InstPosition::After,
            _ => unreachable!(),
        }
    }
    #[inline(always)]
    pub fn next(self) -> ProgPoint {
        Self {
            bits: self.bits + 1,
        }
    }
    #[inline(always)]
    pub fn prev(self) -> ProgPoint {
        Self {
            bits: self.bits - 1,
        }
    }
    #[inline(always)]
    pub fn to_index(self) -> u32 {
        self.bits
    }
    #[inline(always)]
    pub fn from_index(index: u32) -> Self {
        Self { bits: index }
    }
}

/// An instruction to insert into the program to perform some data movement.
#[derive(Clone, Debug)]
pub enum Edit {
    /// Move one allocation to another. Each allocation may be a
    /// register or a stack slot (spillslot). However, stack-to-stack
    /// moves will never be generated.
    ///
    /// `to_vreg`, if defined, is useful as metadata: it indicates
    /// that the moved value is a def of a new vreg.
    ///
    /// `Move` edits will be generated even if src and dst allocation
    /// are the same if the vreg changes; this allows proper metadata
    /// tracking even when moves are elided.
    Move {
        from: Allocation,
        to: Allocation,
        to_vreg: Option<VReg>,
    },

    /// Define a particular Allocation to contain a particular VReg. Useful
    /// for the checker.
    DefAlloc { alloc: Allocation, vreg: VReg },
}

/// A machine envrionment tells the register allocator which registers
/// are available to allocate and what register may be used as a
/// scratch register for each class, and some other miscellaneous info
/// as well.
#[derive(Clone, Debug)]
pub struct MachineEnv {
    /// Physical registers. Every register that might be mentioned in
    /// any constraint must be listed here, even if it is not
    /// allocatable (present in one of
    /// `{preferred,non_preferred}_regs_by_class`).
    pub regs: Vec<PReg>,

    /// Preferred physical registers for each class. These are the
    /// registers that will be allocated first, if free.
    pub preferred_regs_by_class: [Vec<PReg>; 2],

    /// Non-preferred physical registers for each class. These are the
    /// registers that will be allocated if a preferred register is
    /// not available; using one of these is considered suboptimal,
    /// but still better than spilling.
    pub non_preferred_regs_by_class: [Vec<PReg>; 2],

    /// One scratch register per class. This is needed to perform
    /// moves between registers when cyclic move patterns occur. The
    /// register should not be placed in either the preferred or
    /// non-preferred list (i.e., it is not otherwise allocatable).
    ///
    /// Note that the register allocator will freely use this register
    /// between instructions, but *within* the machine code generated
    /// by a single (regalloc-level) instruction, the client is free
    /// to use the scratch register. E.g., if one "instruction" causes
    /// the emission of two machine-code instructions, this lowering
    /// can use the scratch register between them.
    pub scratch_by_class: [PReg; 2],
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

    /// Safepoint records: at a given program point, a reference-typed value lives in the given SpillSlot.
    pub safepoint_slots: Vec<(ProgPoint, SpillSlot)>,

    /// Debug info: a labeled value (as applied to vregs by
    /// `Function::debug_value_labels()` on the input side) is located
    /// in the given allocation from the first program point
    /// (inclusive) to the second (exclusive). Guaranteed to be sorted
    /// by label and program point, and the ranges are guaranteed to
    /// be disjoint.
    pub debug_locations: Vec<(u32, ProgPoint, ProgPoint, Allocation)>,

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
    /// Critical edge is not split between given blocks.
    CritEdge(Block, Block),
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
    /// A VReg is live-in on entry; this is not allowed.
    EntryLivein,
    /// A branch has non-blockparam arg(s) and at least one of the
    /// successor blocks has more than one predecessor, forcing
    /// edge-moves before this branch. This is disallowed because it
    /// places a use after the edge moves occur; insert an edge block
    /// to avoid the situation.
    DisallowedBranchArg(Inst),
    /// Too many pinned VRegs + Reg-constrained Operands are live at
    /// once, making allocation impossible.
    TooManyLiveRegs,
}

impl std::fmt::Display for RegAllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for RegAllocError {}

pub fn run<F: Function>(
    func: &F,
    env: &MachineEnv,
    options: &RegallocOptions,
) -> Result<Output, RegAllocError> {
    ion::run(func, env, options.verbose_log)
}

/// Options for allocation.
#[derive(Clone, Copy, Debug, Default)]
pub struct RegallocOptions {
    /// Add extra verbosity to debug logs.
    pub verbose_log: bool,
}
