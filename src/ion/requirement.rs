//! Requirements computation.

use super::{Env, LiveBundleIndex};
use crate::{Function, Operand, OperandPolicy, PReg, RegClass};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Requirement {
    Unknown,
    Fixed(PReg),
    Register(RegClass),
    Stack(RegClass),
    Any(RegClass),
    Conflict,
}
impl Requirement {
    #[inline(always)]
    pub fn class(self) -> RegClass {
        match self {
            Requirement::Unknown => panic!("No class for unknown Requirement"),
            Requirement::Fixed(preg) => preg.class(),
            Requirement::Register(class) | Requirement::Any(class) | Requirement::Stack(class) => {
                class
            }
            Requirement::Conflict => panic!("No class for conflicted Requirement"),
        }
    }
    #[inline(always)]
    pub fn merge(self, other: Requirement) -> Requirement {
        match (self, other) {
            (Requirement::Unknown, other) | (other, Requirement::Unknown) => other,
            (Requirement::Conflict, _) | (_, Requirement::Conflict) => Requirement::Conflict,
            (other, Requirement::Any(rc)) | (Requirement::Any(rc), other) => {
                if other.class() == rc {
                    other
                } else {
                    Requirement::Conflict
                }
            }
            (Requirement::Stack(rc1), Requirement::Stack(rc2)) => {
                if rc1 == rc2 {
                    self
                } else {
                    Requirement::Conflict
                }
            }
            (Requirement::Register(rc), Requirement::Fixed(preg))
            | (Requirement::Fixed(preg), Requirement::Register(rc)) => {
                if rc == preg.class() {
                    Requirement::Fixed(preg)
                } else {
                    Requirement::Conflict
                }
            }
            (Requirement::Register(rc1), Requirement::Register(rc2)) => {
                if rc1 == rc2 {
                    self
                } else {
                    Requirement::Conflict
                }
            }
            (Requirement::Fixed(a), Requirement::Fixed(b)) if a == b => self,
            _ => Requirement::Conflict,
        }
    }
    #[inline(always)]
    pub fn from_operand(op: Operand) -> Requirement {
        match op.policy() {
            OperandPolicy::FixedReg(preg) => Requirement::Fixed(preg),
            OperandPolicy::Reg | OperandPolicy::Reuse(_) => Requirement::Register(op.class()),
            OperandPolicy::Stack => Requirement::Stack(op.class()),
            _ => Requirement::Any(op.class()),
        }
    }
}

impl<'a, F: Function> Env<'a, F> {
    pub fn compute_requirement(&self, bundle: LiveBundleIndex) -> Requirement {
        let mut req = Requirement::Unknown;
        log::debug!("compute_requirement: {:?}", bundle);
        for entry in &self.bundles[bundle.index()].ranges {
            log::debug!(" -> LR {:?}", entry.index);
            for u in &self.ranges[entry.index.index()].uses {
                log::debug!("  -> use {:?}", u);
                let r = Requirement::from_operand(u.operand);
                req = req.merge(r);
                log::debug!("     -> req {:?}", req);
            }
        }
        log::debug!(" -> final: {:?}", req);
        req
    }
}
