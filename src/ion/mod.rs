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

//! Backtracking register allocator. See doc/DESIGN.md for details of
//! its design.

use crate::ssa::validate_ssa;
use crate::{Function, MachineEnv, PReg, RegAllocError, RegClass, VecExt};
pub(crate) mod data_structures;
pub use data_structures::Ctx;
pub use data_structures::Stats;
use data_structures::*;
pub(crate) mod reg_traversal;
use reg_traversal::*;
pub(crate) mod requirement;
use requirement::*;
pub(crate) mod redundant_moves;
use redundant_moves::*;
pub(crate) mod liveranges;
use liveranges::*;
pub(crate) mod merge;
pub(crate) mod process;
use process::*;
use smallvec::smallvec;
pub(crate) mod dump;
pub(crate) mod moves;
pub(crate) mod spill;

impl<'a, F: Function> Env<'a, F> {
    pub(crate) fn new(func: &'a F, env: &'a MachineEnv, ctx: &'a mut Ctx) -> Self {
        let ninstrs = func.num_insts();
        let nblocks = func.num_blocks();

        ctx.liveins.prepare(nblocks);
        ctx.liveouts.prepare(nblocks);
        ctx.blockparam_ins.clear();
        ctx.blockparam_outs.clear();
        ctx.ranges.prepare(4 * ninstrs);
        ctx.bundles.prepare(ninstrs);
        ctx.spillsets.prepare(ninstrs);
        ctx.vregs.prepare(ninstrs);
        for preg in ctx.pregs.iter_mut() {
            preg.is_stack = false;
            preg.allocations.btree.clear();
        }
        ctx.allocation_queue.heap.clear();
        ctx.spilled_bundles.clear();
        ctx.scratch_spillset_pool
            .extend(ctx.spillslots.drain(..).map(|mut s| {
                s.ranges.btree.clear();
                s.ranges
            }));
        ctx.slots_by_class = core::array::from_fn(|_| SpillSlotList::default());
        ctx.extra_spillslots_by_class = core::array::from_fn(|_| smallvec![]);
        ctx.preferred_victim_by_class = [PReg::invalid(); 3];
        ctx.multi_fixed_reg_fixups.clear();
        ctx.allocated_bundle_count = 0;
        ctx.debug_annotations.clear();
        ctx.conflict_set.clear();
        ctx.scratch_bump.reset();

        ctx.output.allocs.prepare(4 * ninstrs);
        ctx.output.inst_alloc_offsets.clear();
        ctx.output.num_spillslots = 0;
        ctx.output.debug_locations.clear();
        ctx.output.edits.clear();
        ctx.output.stats = Stats::default();

        Self { func, env, ctx }
    }

    pub(crate) fn init(&mut self) -> Result<(), RegAllocError> {
        self.create_pregs_and_vregs();
        self.compute_liveness()?;
        self.build_liveranges();
        self.fixup_multi_fixed_vregs();
        self.merge_vreg_bundles();
        self.queue_bundles();
        if trace_enabled!() {
            self.dump_state();
        }
        Ok(())
    }

    pub(crate) fn run(&mut self) -> Result<(), RegAllocError> {
        self.process_bundles()?;
        self.try_allocating_regs_for_spilled_bundles();
        self.allocate_spillslots();
        self.apply_allocations_and_insert_moves();
        self.resolve_inserted_moves();
        Ok(())
    }
}

pub fn run<F: Function>(
    func: &F,
    mach_env: &MachineEnv,
    ctx: &mut Ctx,
    enable_annotations: bool,
    enable_ssa_checker: bool,
) -> Result<(), RegAllocError> {
    ctx.cfginfo.init(func)?;

    if enable_ssa_checker {
        validate_ssa(func, &ctx.cfginfo)?;
    }

    ctx.annotations_enabled = enable_annotations;
    let mut env = Env::new(func, mach_env, ctx);
    env.init()?;

    env.run()?;

    if enable_annotations {
        env.dump_results();
    }

    Ok(())
}
