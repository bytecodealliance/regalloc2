/*
 * This file was initially derived from the files
 * `js/src/jit/BacktrackingAllocator.h` and
 * `js/src/jit/BacktrackingAllocator.cpp` in Mozilla Firefox, and was
 * originally licensed under the Mozilla Public License 2.0. We
 * subsequently relicensed it to Apache-2.0 WITH LLVM-exception (see
 * https://github.com/bytecodealliance/regalloc2/issues/7).
 *
 * Since the initial port, the design has been substantially evolved
 * and optimized.
 */

//! Requirements computation.

use super::{Env, LiveBundleIndex};
use crate::{Function, Operand, OperandConstraint, PReg, RegClass};

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
        match op.constraint() {
            OperandConstraint::FixedReg(preg) => Requirement::Fixed(preg),
            OperandConstraint::Reg | OperandConstraint::Reuse(_) => {
                Requirement::Register(op.class())
            }
            OperandConstraint::Stack => Requirement::Stack(op.class()),
            _ => Requirement::Any(op.class()),
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
