use crate::{Operand, OperandKind, OperandPos, OperandConstraint};

#[derive(Clone, Copy, PartialEq)]
enum OperandConstraintKind {
    Any,
    Reg,
    Stack,
    FixedReg,
    Reuse,
}

impl From<OperandConstraint> for OperandConstraintKind {
    fn from(constraint: OperandConstraint) -> Self {
        match constraint {
            OperandConstraint::Any => Self::Any,
            OperandConstraint::Reg => Self::Reg,
            OperandConstraint::Stack => Self::Stack,
            OperandConstraint::FixedReg(_) => Self::FixedReg,
            OperandConstraint::Reuse(_) => Self::Reuse,
        }
    }
}

pub struct Operands<'a>(pub &'a [Operand]);

impl<'a> Operands<'a> {
    pub fn new(operands: &'a [Operand]) -> Self {
        Self(operands)
    }

    pub fn matches<F: Fn(Operand) -> bool + 'a>(&self, predicate: F) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.0.iter()
            .cloned()
            .enumerate()
            .filter(move |(_, op)| predicate(*op))
    }

    pub fn non_fixed_non_reuse_late(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op|
            OperandConstraintKind::FixedReg != op.constraint().into()
            && OperandConstraintKind::Reuse != op.constraint().into()
            && op.pos() == OperandPos::Late
        )
    }

    pub fn non_reuse_late_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op|
            OperandConstraintKind::Reuse != op.constraint().into()
            && op.pos() == OperandPos::Late
            && op.kind() == OperandKind::Def
        )
    }

    pub fn non_fixed_non_reuse_early(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op|
            OperandConstraintKind::FixedReg != op.constraint().into()
            && OperandConstraintKind::Reuse != op.constraint().into()
            && op.pos() == OperandPos::Early
        )
    }

    pub fn reuse(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| OperandConstraintKind::Reuse == op.constraint().into())
    }

    pub fn non_reuse_early_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op|
            OperandConstraintKind::Reuse != op.constraint().into()
            && op.pos() == OperandPos::Early
            && op.kind() == OperandKind::Def
        )
    }

    pub fn fixed_early(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op|
            OperandConstraintKind::FixedReg == op.constraint().into()
            && op.pos() == OperandPos::Early
        )
    }

    pub fn fixed_late(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op|
            OperandConstraintKind::FixedReg == op.constraint().into()
            && op.pos() == OperandPos::Late
        )
    }

    pub fn non_reuse_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op|
            OperandConstraintKind::Reuse != op.constraint().into()
            && op.kind() == OperandKind::Def
        )
    }

    pub fn non_fixed_non_reuse_late_use(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op|
            OperandConstraintKind::FixedReg != op.constraint().into()
            && OperandConstraintKind::Reuse != op.constraint().into()
            && op.pos() == OperandPos::Late
            && op.kind() == OperandKind::Use
        )
    }

    pub fn non_fixed_non_reuse_late_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op|
            OperandConstraintKind::FixedReg != op.constraint().into()
            && OperandConstraintKind::Reuse != op.constraint().into()
            && op.pos() == OperandPos::Late
            && op.kind() == OperandKind::Def
        )
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::vec;
    use crate::{PReg, RegClass};
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

    const fn fixed_late_def_operand(vreg_no: u32) -> Operand {
        operand(
            vreg_no,
            OperandConstraint::FixedReg(PReg::new(1, RegClass::Int)),
            OperandKind::Def,
            OperandPos::Late,
        )
    }

    const fn fixed_early_def_operand(vreg_no: u32) -> Operand {
        operand(
            vreg_no,
            OperandConstraint::FixedReg(PReg::new(1, RegClass::Int)),
            OperandKind::Def,
            OperandPos::Early,
        )
    }


    const fn fixed_late_use_operand(vreg_no: u32) -> Operand {
        operand(
            vreg_no,
            OperandConstraint::FixedReg(PReg::new(1, RegClass::Int)),
            OperandKind::Use,
            OperandPos::Late,
        )
    }

    const fn fixed_early_use_operand(vreg_no: u32) -> Operand {
        operand(
            vreg_no,
            OperandConstraint::FixedReg(PReg::new(1, RegClass::Int)),
            OperandKind::Use,
            OperandPos::Early,
        )
    }

    static OPERANDS: [Operand; 14] = [
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
        
        fixed_late_def_operand(10),
        fixed_early_def_operand(11),
        fixed_late_use_operand(12),
        fixed_early_use_operand(13),
    ];

    #[test]
    fn late() {
        let late_operands: Vec<(usize, Operand)> = Operands::new(&OPERANDS).non_fixed_non_reuse_late()
            .collect();
        assert_eq!(late_operands, vec![
            (1, late_def_operand(1)),
            (6, late_use_operand(6)),
        ]);
    }

    #[test]
    fn late_def() {
        let late_def_operands: Vec<(usize, Operand)> = Operands::new(&OPERANDS).non_reuse_late_def()
            .collect();
        assert_eq!(late_def_operands, vec![
            (1, late_def_operand(1)),
            (10, fixed_late_def_operand(10)),
        ]);
    }

    #[test]
    fn early() {
        let early_operands: Vec<(usize, Operand)> = Operands::new(&OPERANDS).non_fixed_non_reuse_early()
            .collect();
        assert_eq!(early_operands, vec![
            (3, early_use_operand(3)),
            (4, early_def_operand(4)),
            (8, early_def_operand(8)),
            (9, early_use_operand(9)),
        ]);
    }

    #[test]
    fn early_def() {
        let early_def_operands: Vec<(usize, Operand)> = Operands::new(&OPERANDS).non_reuse_early_def()
            .collect();
        assert_eq!(early_def_operands, vec![
            (4, early_def_operand(4)),
            (8, early_def_operand(8)),
            (11, fixed_early_def_operand(11)),
        ]);
    }

    #[test]
    fn reuse() {
        let reuse_operands: Vec<(usize, Operand)> = Operands::new(&OPERANDS).reuse()
            .collect();
        assert_eq!(reuse_operands, vec![
            (0, late_reuse_def_operand(0)),
            (2, early_reuse_def_operand(2)),
            (5, late_reuse_def_operand(5)),
            (7, late_reuse_use_operand(7)),
        ]);
    }

    #[test]
    fn fixed_late() {
        let fixed_late_operands: Vec<(usize, Operand)> = Operands::new(&OPERANDS).fixed_late()
            .collect();
        assert_eq!(fixed_late_operands, vec![
            (10, fixed_late_def_operand(10)),
            (12, fixed_late_use_operand(12)),
        ]);
    }

    #[test]
    fn fixed_early() {
        let fixed_early_operands: Vec<(usize, Operand)> = Operands::new(&OPERANDS).fixed_early()
            .collect();
        assert_eq!(fixed_early_operands, vec![
            (11, fixed_early_def_operand(11)),
            (13, fixed_early_use_operand(13)),
        ]);
    }

    #[test]
    fn def() {
        let def_operands: Vec<(usize, Operand)> = Operands::new(&OPERANDS).non_reuse_def()
            .collect();
        assert_eq!(def_operands, vec![
            (1, late_def_operand(1)),
            (4, early_def_operand(4)),
            (8, early_def_operand(8)),
            (10, fixed_late_def_operand(10)),
            (11, fixed_early_def_operand(11)),
        ]);
    }


    #[test]
    fn non_fixed_non_reuse_late_def() {
        let def_operands: Vec<(usize, Operand)> = Operands::new(&OPERANDS).non_fixed_non_reuse_late_def()
            .collect();
        assert_eq!(def_operands, vec![
            (1, late_def_operand(1)),
        ]);
    }

    #[test]
    fn non_fixed_non_reuse_late_use() {
        let late_operands: Vec<(usize, Operand)> = Operands::new(&OPERANDS).non_fixed_non_reuse_late_use()
            .collect();
        assert_eq!(late_operands, vec![
            (6, late_use_operand(6)),
        ]);
    }
}
