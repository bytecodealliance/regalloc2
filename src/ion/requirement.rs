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
    #[inline(always)]
    pub fn merge_requirement(&self, a: Requirement, b: Requirement) -> Requirement {
        match (a, b) {
            (Requirement::Unknown, other) | (other, Requirement::Unknown) => other,
            (Requirement::Conflict, _) | (_, Requirement::Conflict) => Requirement::Conflict,
            (other, Requirement::Any) | (Requirement::Any, other) => other,
            (Requirement::Stack, Requirement::Stack) => Requirement::Stack,
            (Requirement::Register, Requirement::Register) => Requirement::Register,
            (Requirement::Register, Requirement::Fixed(preg))
            | (Requirement::Fixed(preg), Requirement::Register) if !self.pregs[preg.index()].is_stack => Requirement::Fixed(preg),
            (Requirement::Stack, Requirement::Fixed(preg))
            | (Requirement::Fixed(preg), Requirement::Stack) if self.pregs[preg.index()].is_stack => Requirement::Fixed(preg),
            (Requirement::Fixed(a), Requirement::Fixed(b)) if a == b => Requirement::Fixed(a),
            _ => Requirement::Conflict,
        }
    }

    pub fn compute_requirement(&self, bundle: LiveBundleIndex) -> Requirement {
        let mut req = Requirement::Unknown;
        log::trace!("compute_requirement: {:?}", bundle);
        for entry in &self.bundles[bundle.index()].ranges {
            log::trace!(" -> LR {:?}", entry.index);
            for u in &self.ranges[entry.index.index()].uses {
                log::trace!("  -> use {:?}", u);
                let r = Requirement::from_operand(u.operand);
                req = self.merge_requirement(req, r);
                log::trace!("     -> req {:?}", req);
            }
        }
        log::trace!(" -> final: {:?}", req);
        req
    }
}
