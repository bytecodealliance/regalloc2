//! Requirements computation.

use super::{Env, LiveBundleIndex};
use crate::{Function, Operand, OperandConstraint, PReg};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Requirement {
    Unknown,
    Fixed(PReg),
    Register,
    Stack,
    Any,
    Conflict,
}
impl Requirement {
    #[inline(always)]
    pub fn merge(self, other: Requirement) -> Requirement {
        match (self, other) {
            (Requirement::Unknown, other) | (other, Requirement::Unknown) => other,
            (Requirement::Conflict, _) | (_, Requirement::Conflict) => Requirement::Conflict,
            (other, Requirement::Any) | (Requirement::Any, other) => other,
            (Requirement::Stack, Requirement::Stack) => self,
            (Requirement::Register, Requirement::Fixed(preg))
            | (Requirement::Fixed(preg), Requirement::Register) => Requirement::Fixed(preg),
            (Requirement::Register, Requirement::Register) => self,
            (Requirement::Fixed(a), Requirement::Fixed(b)) if a == b => self,
            _ => Requirement::Conflict,
        }
    }
    #[inline(always)]
    pub fn from_operand(op: Operand) -> Requirement {
        match op.constraint() {
            OperandConstraint::FixedReg(preg) => Requirement::Fixed(preg),
            OperandConstraint::Reg | OperandConstraint::Reuse(_) => Requirement::Register,
            OperandConstraint::Stack => Requirement::Stack,
            _ => Requirement::Any,
        }
    }
}

impl<'a, F: Function> Env<'a, F> {
    pub fn compute_requirement(&self, bundle: LiveBundleIndex) -> Requirement {
        let mut req = Requirement::Unknown;
        log::trace!("compute_requirement: {:?}", bundle);
        for entry in &self.bundles[bundle.index()].ranges {
            log::trace!(" -> LR {:?}", entry.index);
            for u in &self.ranges[entry.index.index()].uses {
                log::trace!("  -> use {:?}", u);
                let r = Requirement::from_operand(u.operand);
                req = req.merge(r);
                log::trace!("     -> req {:?}", req);
            }
        }
        log::trace!(" -> final: {:?}", req);
        req
    }
}
