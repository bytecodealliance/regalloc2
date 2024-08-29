use crate::{Operand, OperandConstraint, OperandKind, OperandPos};

pub struct Operands<'a>(pub &'a [Operand]);

impl<'a> Operands<'a> {
    pub fn new(operands: &'a [Operand]) -> Self {
        Self(operands)
    }

    pub fn matches<F: Fn(Operand) -> bool + 'a>(
        &self,
        predicate: F,
    ) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.0
            .iter()
            .cloned()
            .enumerate()
            .filter(move |(_, op)| predicate(*op))
    }

    pub fn def_ops(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| op.kind() == OperandKind::Def)
    }

    pub fn use_ops(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| op.kind() == OperandKind::Use)
    }

    pub fn non_fixed_use(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(op.constraint(), OperandConstraint::FixedReg(_))
            && op.kind() == OperandKind::Use
        })
    }

    pub fn non_fixed_non_reuse_late(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(
                op.constraint(),
                OperandConstraint::FixedReg(_) | OperandConstraint::Reuse(_)
            ) && op.pos() == OperandPos::Late
        })
    }

    pub fn non_reuse_late_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(op.constraint(), OperandConstraint::Reuse(_))
                && op.pos() == OperandPos::Late
                && op.kind() == OperandKind::Def
        })
    }

    pub fn non_fixed_non_reuse_early(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(op.constraint(), OperandConstraint::FixedReg(_))
                && !matches!(op.constraint(), OperandConstraint::Reuse(_))
                && op.pos() == OperandPos::Early
        })
    }

    pub fn reuse(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| matches!(op.constraint(), OperandConstraint::Reuse(_)))
    }

    pub fn non_reuse_early_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(op.constraint(), OperandConstraint::Reuse(_))
                && op.pos() == OperandPos::Early
                && op.kind() == OperandKind::Def
        })
    }

    pub fn fixed(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| matches!(op.constraint(), OperandConstraint::FixedReg(_)))
    }

    pub fn fixed_early(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            matches!(op.constraint(), OperandConstraint::FixedReg(_))
                && op.pos() == OperandPos::Early
        })
    }

    pub fn fixed_late(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            matches!(op.constraint(), OperandConstraint::FixedReg(_))
                && op.pos() == OperandPos::Late
        })
    }

    pub fn non_reuse_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(op.constraint(), OperandConstraint::Reuse(_)) && op.kind() == OperandKind::Def
        })
    }

    pub fn non_fixed_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(op.constraint(), OperandConstraint::FixedReg(_))
                && op.kind() == OperandKind::Def
        })
    }

    pub fn non_fixed_non_reuse_late_use(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(
                op.constraint(),
                OperandConstraint::FixedReg(_) | OperandConstraint::Reuse(_)
            ) && op.pos() == OperandPos::Late
                && op.kind() == OperandKind::Use
        })
    }

    pub fn non_fixed_non_reuse_late_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(
                op.constraint(),
                OperandConstraint::FixedReg(_) | OperandConstraint::Reuse(_)
            ) && op.pos() == OperandPos::Late
                && op.kind() == OperandKind::Def
        })
    }

    pub fn non_fixed_late_use(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(op.constraint(), OperandConstraint::FixedReg(_))
                && op.pos() == OperandPos::Late
                && op.kind() == OperandKind::Use
        })
    }

    pub fn non_fixed_late_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(op.constraint(), OperandConstraint::FixedReg(_))
                && op.pos() == OperandPos::Late
                && op.kind() == OperandKind::Def
        })
    }

    pub fn non_fixed_early_use(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(op.constraint(), OperandConstraint::FixedReg(_))
                && op.pos() == OperandPos::Early
                && op.kind() == OperandKind::Use
        })
    }

    pub fn non_fixed_early_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            !matches!(op.constraint(), OperandConstraint::FixedReg(_))
                && op.pos() == OperandPos::Early
                && op.kind() == OperandKind::Def
        })
    }

    pub fn late_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| op.pos() == OperandPos::Late && op.kind() == OperandKind::Def)
    }

    pub fn early_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| op.pos() == OperandPos::Early && op.kind() == OperandKind::Def)
    }

    pub fn fixed_early_use(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            matches!(op.constraint(), OperandConstraint::FixedReg(_))
                && op.pos() == OperandPos::Early
                && op.kind() == OperandKind::Use
        })
    }

    pub fn fixed_late_def(&self) -> impl Iterator<Item = (usize, Operand)> + 'a {
        self.matches(|op| {
            matches!(op.constraint(), OperandConstraint::FixedReg(_))
                && op.pos() == OperandPos::Late
                && op.kind() == OperandKind::Def
        })
    }
}

impl<'a> core::ops::Index<usize> for Operands<'a> {
    type Output = Operand;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
