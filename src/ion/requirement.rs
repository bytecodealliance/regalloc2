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

pub struct RequirementConflict;

pub struct RequirementConflictAt(pub ProgPoint);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Requirement {
    FixedReg(PReg),
    FixedStack(PReg),
    Register,
    Stack,
    Any,
}
impl Requirement {
    #[inline(always)]
    pub fn merge(self, other: Requirement) -> Result<Requirement, RequirementConflict> {
        match (self, other) {
            (other, Requirement::Any) | (Requirement::Any, other) => Ok(other),
            (Requirement::Register, Requirement::Register) => Ok(self),
            (Requirement::Stack, Requirement::Stack) => Ok(self),
            (Requirement::Register, Requirement::FixedReg(preg))
            | (Requirement::FixedReg(preg), Requirement::Register) => {
                Ok(Requirement::FixedReg(preg))
            }
            (Requirement::Stack, Requirement::FixedStack(preg))
            | (Requirement::FixedStack(preg), Requirement::Stack) => {
                Ok(Requirement::FixedStack(preg))
            }
            (Requirement::FixedReg(a), Requirement::FixedReg(b)) if a == b => Ok(self),
            (Requirement::FixedStack(a), Requirement::FixedStack(b)) if a == b => Ok(self),
            _ => Err(RequirementConflict),
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
            OperandConstraint::Any => Requirement::Any,
        }
    }

    pub fn compute_requirement(
        &self,
        bundle: LiveBundleIndex,
    ) -> Result<Requirement, RequirementConflictAt> {
        let mut req = Requirement::Any;
        log::trace!("compute_requirement: {:?}", bundle);
        let ranges = &self.bundles[bundle.index()].ranges;
        for entry in ranges {
            log::trace!(" -> LR {:?}", entry.index);
            for u in &self.ranges[entry.index.index()].uses {
                log::trace!("  -> use {:?}", u);
                let r = self.requirement_from_operand(u.operand);
                req = req.merge(r).map_err(|_| {
                    log::trace!("     -> conflict");
                    RequirementConflictAt(u.pos)
                })?;
                log::trace!("     -> req {:?}", req);
            }
        }
        log::trace!(" -> final: {:?}", req);
        Ok(req)
    }

    pub fn merge_bundle_requirements(
        &self,
        a: LiveBundleIndex,
        b: LiveBundleIndex,
    ) -> Result<Requirement, RequirementConflict> {
        let req_a = self
            .compute_requirement(a)
            .map_err(|_| RequirementConflict)?;
        let req_b = self
            .compute_requirement(b)
            .map_err(|_| RequirementConflict)?;
        req_a.merge(req_b)
    }
}
