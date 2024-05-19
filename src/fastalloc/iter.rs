use crate::{Operand, OperandKind, OperandPos, OperandConstraint};
use std::println;

/// Looking for operands with this particular constraint.
#[derive(Clone, Copy, PartialEq)]
enum LookingFor {
    FixedReg,
    Others
}

/// Iterate over operands in position `pos` and kind
/// `kind` in no particular order.
struct ByKindAndPosOperands<'a> {
    operands: &'a [Operand],
    idx: usize,
    kind: OperandKind,
    pos: OperandPos,
}

impl<'a> ByKindAndPosOperands<'a> {
    fn new(operands: &'a [Operand], kind: OperandKind, pos: OperandPos) -> Self {
        Self { operands, idx: 0, kind, pos }
    }
}

impl<'a> Iterator for ByKindAndPosOperands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        while self.idx < self.operands.len() && (self.operands[self.idx].kind() != self.kind
            || self.operands[self.idx].pos() != self.pos) {
                self.idx += 1;
        }
        if self.idx >= self.operands.len() {
            None
        } else {
            self.idx += 1;
            Some((self.idx - 1, self.operands[self.idx - 1]))
        }
    }
}

/// Iterate over operands with position `pos` starting from the ones with
/// fixed registers, then the rest.
struct ByPosOperands<'a> {
    operands: &'a [Operand],
    idx: usize,
    looking_for: LookingFor,
    pos: OperandPos,
}

impl<'a> ByPosOperands<'a> {
    fn new(operands: &'a [Operand], pos: OperandPos) -> Self {
        Self { operands, idx: 0, looking_for: LookingFor::FixedReg, pos }
    }
}

impl<'a> ByPosOperands<'a> {
    fn next_fixed_reg(&mut self) -> Option<(usize, Operand)> {
        while self.idx < self.operands.len() && (self.operands[self.idx].pos() != self.pos
            || !matches!(self.operands[self.idx].constraint(), OperandConstraint::FixedReg(_))) {
            self.idx += 1;
        }
        if self.idx >= self.operands.len() {
            None
        } else {
            self.idx += 1;
            Some((self.idx - 1, self.operands[self.idx - 1]))
        }
    }

    fn next_others(&mut self) -> Option<(usize, Operand)> {
        while self.idx < self.operands.len() && (self.operands[self.idx].pos() != self.pos
            || matches!(self.operands[self.idx].constraint(), OperandConstraint::FixedReg(_))) {
            self.idx += 1;
        }
        if self.idx >= self.operands.len() {
            None
        } else {
            self.idx += 1;
            Some((self.idx - 1, self.operands[self.idx - 1]))
        }
    }
}

impl<'a> Iterator for ByPosOperands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.operands.len() {
            if self.looking_for == LookingFor::FixedReg {
                self.idx = 0;
                self.looking_for = LookingFor::Others;
            } else {
                return None;
            }
        }
        match self.looking_for {
            LookingFor::FixedReg => {
                let next = self.next_fixed_reg();
                if next.is_none() {
                    self.next()
                } else {
                    next
                }
            },
            LookingFor::Others => self.next_others(),
        }
    }
}

pub struct LateOperands<'a>(ByPosOperands<'a>);

impl<'a> LateOperands<'a> {
    pub fn new(operands: &'a [Operand]) -> Self {
        Self(ByPosOperands::new(operands, OperandPos::Late))
    }
}

impl<'a> Iterator for LateOperands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct EarlyOperands<'a>(ByPosOperands<'a>);

impl<'a> EarlyOperands<'a> {
    pub fn new(operands: &'a [Operand]) -> Self {
        Self(ByPosOperands::new(operands, OperandPos::Early))
    }
}

impl<'a> Iterator for EarlyOperands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct LateDefOperands<'a>(ByKindAndPosOperands<'a>);

impl<'a> LateDefOperands<'a> {
    pub fn new(operands: &'a [Operand]) -> Self {
        Self(ByKindAndPosOperands::new(operands, OperandKind::Def, OperandPos::Late))
    }
}

impl<'a> Iterator for LateDefOperands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct EarlyDefOperands<'a>(ByKindAndPosOperands<'a>);

impl<'a> EarlyDefOperands<'a> {
    pub fn new(operands: &'a [Operand]) -> Self {
        Self(ByKindAndPosOperands::new(operands, OperandKind::Def, OperandPos::Early))
    }
}

impl<'a> Iterator for EarlyDefOperands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
