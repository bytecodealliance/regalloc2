use crate::{Operand, OperandKind, OperandPos, OperandConstraint};

#[derive(Clone, Copy, PartialEq)]
enum OperandConstraintKind {
    Any,
    Reg,
    Stack,
    FixedReg,
    Reuse,
}

impl PartialEq<OperandConstraint> for OperandConstraintKind {
    fn eq(&self, other: &OperandConstraint) -> bool {
        match other {
            OperandConstraint::Any => *self == Self::Any,
            OperandConstraint::Reg => *self == Self::Reg,
            OperandConstraint::Stack => *self == Self::Stack,
            OperandConstraint::FixedReg(_) => *self == Self::FixedReg,
            OperandConstraint::Reuse(_) => *self == Self::Reuse,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct SearchConstraint {
    kind: Option<OperandKind>,
    pos: Option<OperandPos>,
    must_not_have_constraint: Option<OperandConstraintKind>,
    must_have_constraint: Option<OperandConstraintKind>,
}

impl SearchConstraint {
    fn meets_constraint(&self, op: Operand) -> bool {
        match self.pos {
            None => (),
            Some(expected_pos) => if op.pos() != expected_pos {
                return false;
            }
        };
        match self.kind {
            None => (),
            Some(expected_kind) => if op.kind() != expected_kind {
                return false;
            }
        };
        match self.must_not_have_constraint {
            None => (),
            Some(should_not_be_constraint) => if should_not_be_constraint == op.constraint() {
                return false;
            }
        }
        match self.must_have_constraint {
            None => (),
            Some(should_be_constraint) => if should_be_constraint != op.constraint() {
                return false;
            }
        }
        true
    }
}

struct Operands<'a> {
    operands: &'a [Operand],
    idx: usize,
    search_constraint: SearchConstraint,
}

impl<'a> Operands<'a> {
    fn new(operands: &'a [Operand], search_constraint: SearchConstraint) -> Self {
        Self { operands, search_constraint, idx: 0 }
    }
}

impl<'a> Iterator for Operands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        while self.idx < self.operands.len() 
            && !self.search_constraint.meets_constraint(self.operands[self.idx])
        {
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


/*/// Iterate over operands in position `pos` and kind
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
}*/

pub struct NonReuseLateOperands<'a>(Operands<'a>);

impl<'a> NonReuseLateOperands<'a> {
    pub fn new(operands: &'a [Operand]) -> Self {
        Self(Operands::new(operands, SearchConstraint {
            pos: Some(OperandPos::Late),
            kind: None,
            must_not_have_constraint: Some(OperandConstraintKind::Reuse),
            must_have_constraint: None,
        }))
    }
}

impl<'a> Iterator for NonReuseLateOperands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct NonReuseEarlyOperands<'a>(Operands<'a>);

impl<'a> NonReuseEarlyOperands<'a> {
    pub fn new(operands: &'a [Operand]) -> Self {
        Self(Operands::new(operands, SearchConstraint {
            pos: Some(OperandPos::Early),
            kind: None,
            must_not_have_constraint: Some(OperandConstraintKind::Reuse),
            must_have_constraint: None,
        }))
    }
}

impl<'a> Iterator for NonReuseEarlyOperands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct NonReuseLateDefOperands<'a>(Operands<'a>);

impl<'a> NonReuseLateDefOperands<'a> {
    pub fn new(operands: &'a [Operand]) -> Self {
        Self(Operands::new(operands, SearchConstraint {
            kind: Some(OperandKind::Def),
            pos: Some(OperandPos::Late),
            must_not_have_constraint: Some(OperandConstraintKind::Reuse),
            must_have_constraint: None,
        }))
    }
}

impl<'a> Iterator for NonReuseLateDefOperands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct NonReuseEarlyDefOperands<'a>(Operands<'a>);

impl<'a> NonReuseEarlyDefOperands<'a> {
    pub fn new(operands: &'a [Operand]) -> Self {
        Self(Operands::new(operands, SearchConstraint {
            kind: Some(OperandKind::Def),
            pos: Some(OperandPos::Early),
            must_have_constraint: None,
            must_not_have_constraint: Some(OperandConstraintKind::Reuse),
        }))
    }
}

impl<'a> Iterator for NonReuseEarlyDefOperands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

/// Operands that reuse and input allocation.
/// They are all expected to be def operands.
pub struct ReuseOperands<'a>(Operands<'a>);

impl<'a> ReuseOperands<'a> {
    pub fn new(operands: &'a [Operand]) -> Self {
        Self(Operands::new(operands, SearchConstraint {
            kind: None,
            pos: None,
            must_have_constraint: Some(OperandConstraintKind::Reuse),
            must_not_have_constraint: None,
        }))
    }
}

impl<'a> Iterator for ReuseOperands<'a> {
    type Item = (usize, Operand);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::vec;
    use crate::RegClass;
    use super::*;

    // Using a new function because Operand::new isn't a const function
    const fn operand(vreg_no: u32, constraint: OperandConstraint, kind: OperandKind, pos: OperandPos) -> Operand {
        let constraint_field = match constraint {
            OperandConstraint::Any => 0,
            OperandConstraint::Reg => 1,
            OperandConstraint::Stack => 2,
            OperandConstraint::FixedReg(preg) => {
                0b1000000 | preg.hw_enc() as u32
            }
            OperandConstraint::Reuse(which) => {
                0b0100000 | which as u32
            }
        };
        let class_field = RegClass::Int as u8 as u32;
        let pos_field = pos as u8 as u32;
        let kind_field = kind as u8 as u32;
        Operand {
            bits: vreg_no
                | (class_field << 21)
                | (pos_field << 23)
                | (kind_field << 24)
                | (constraint_field << 25),
        }
    }

    const fn late_reuse_def_operand(vreg_no: u32) -> Operand {
        operand(vreg_no, OperandConstraint::Reuse(0), OperandKind::Def, OperandPos::Late)
    }

    const fn early_reuse_def_operand(vreg_no: u32) -> Operand {
        operand(vreg_no, OperandConstraint::Reuse(0), OperandKind::Def, OperandPos::Early)
    }

    const fn early_reuse_use_operand(vreg_no: u32) -> Operand {
        operand(vreg_no, OperandConstraint::Reuse(0), OperandKind::Use, OperandPos::Early)
    }

    const fn late_reuse_use_operand(vreg_no: u32) -> Operand {
        operand(vreg_no, OperandConstraint::Reuse(0), OperandKind::Use, OperandPos::Late)
    }

    const fn late_def_operand(vreg_no: u32) -> Operand {
        operand(vreg_no, OperandConstraint::Any, OperandKind::Def, OperandPos::Late)
    }

    const fn late_use_operand(vreg_no: u32) -> Operand {
        operand(vreg_no, OperandConstraint::Any, OperandKind::Use, OperandPos::Late)
    }

    const fn early_use_operand(vreg_no: u32) -> Operand {
        operand(vreg_no, OperandConstraint::Any, OperandKind::Use, OperandPos::Early)
    }

    const fn early_def_operand(vreg_no: u32) -> Operand {
        operand(vreg_no, OperandConstraint::Any, OperandKind::Def, OperandPos::Early)
    }

    static OPERANDS: [Operand; 10] = [
        late_reuse_def_operand(0),
        late_def_operand(1),
        early_reuse_def_operand(2),
        early_use_operand(3),
        early_def_operand(4),
        late_reuse_def_operand(5),
        late_use_operand(6),
        late_reuse_use_operand(7),
        early_def_operand(8),
        early_use_operand(9),
    ];

    #[test]
    fn late() {
        let late_operands: Vec<Operand> = NonReuseLateOperands::new(&OPERANDS)
            .map(|(_, op)| op)
            .collect();
        assert_eq!(late_operands, vec![
            late_def_operand(1),
            late_use_operand(6),
        ]);
    }

    #[test]
    fn late_def() {
        let late_def_operands: Vec<Operand> = NonReuseLateDefOperands::new(&OPERANDS)
            .map(|(_, op)| op)
            .collect();
        assert_eq!(late_def_operands, vec![ late_def_operand(1) ]);
    }

    #[test]
    fn early() {
        let early_operands: Vec<Operand> = NonReuseEarlyOperands::new(&OPERANDS)
            .map(|(_, op)| op)
            .collect();
        assert_eq!(early_operands, vec![
            early_use_operand(3),
            early_def_operand(4),
            early_def_operand(8),
            early_use_operand(9),
        ]);
    }

    #[test]
    fn early_def() {
        let early_def_operands: Vec<Operand> = NonReuseEarlyDefOperands::new(&OPERANDS)
            .map(|(_, op)| op)
            .collect();
        assert_eq!(early_def_operands, vec![
            early_def_operand(4),
            early_def_operand(8),
        ]);
    }

    #[test]
    fn reuse() {
        let reuse_operands: Vec<Operand> = ReuseOperands::new(&OPERANDS)
            .map(|(_, op)| op)
            .collect();
        assert_eq!(reuse_operands, vec![
            late_reuse_def_operand(0),
            early_reuse_def_operand(2),
            late_reuse_def_operand(5),
            late_reuse_use_operand(7),
        ]);
    }
}
