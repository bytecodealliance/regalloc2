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
use crate::{Function, Operand, OperandConstraint, PReg, ProgPoint};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Requirement {
    Unknown,
    FixedReg(PReg),
    FixedStack(PReg),
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
            (Requirement::Register, Requirement::Register) => self,
            (Requirement::Stack, Requirement::Stack) => self,
            (Requirement::Register, Requirement::FixedReg(preg))
            | (Requirement::FixedReg(preg), Requirement::Register) => Requirement::FixedReg(preg),
            (Requirement::Stack, Requirement::FixedStack(preg))
            | (Requirement::FixedStack(preg), Requirement::Stack) => Requirement::FixedStack(preg),
            (Requirement::FixedReg(a), Requirement::FixedReg(b)) if a == b => self,
            (Requirement::FixedStack(a), Requirement::FixedStack(b)) if a == b => self,
            _ => Requirement::Conflict,
        }
    }
}

impl<'a, F: Function> Env<'a, F> {
    #[inline(always)]
    pub fn requirement_from_operand(&self, op: Operand) -> Requirement {
        match op.constraint() {
            OperandConstraint::FixedReg(preg) => {
                if self.pregs[preg.index()].is_stack {
                    Requirement::FixedStack(preg)
                } else {
                    Requirement::FixedReg(preg)
                }
            }
            OperandConstraint::Reg | OperandConstraint::Reuse(_) => Requirement::Register,
            OperandConstraint::Stack => Requirement::Stack,
            _ => Requirement::Any,
        }
    }

    pub fn compute_requirement(&self, bundle: LiveBundleIndex) -> (Requirement, ProgPoint) {
        let mut req = Requirement::Unknown;
        log::trace!("compute_requirement: {:?}", bundle);
        let ranges = &self.bundles[bundle.index()].ranges;
        for entry in ranges {
            log::trace!(" -> LR {:?}", entry.index);
            for u in &self.ranges[entry.index.index()].uses {
                log::trace!("  -> use {:?}", u);
                let r = self.requirement_from_operand(u.operand);
                req = req.merge(r);
                log::trace!("     -> req {:?}", req);
                if req == Requirement::Conflict {
                    return (req, u.pos);
                }
            }
        }
        log::trace!(" -> final: {:?}", req);
        (req, ranges.first().unwrap().range.from)
    }
}
