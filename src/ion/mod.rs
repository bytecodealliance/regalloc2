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

use crate::cfg::CFGInfo;
use crate::ssa::validate_ssa;
use crate::{Function, MachineEnv, Output, PReg, ProgPoint, RegAllocError, RegClass};
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;

pub(crate) mod data_structures;
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
pub(crate) mod stackmap;

impl<'a, F: Function> Env<'a, F> {
    pub(crate) fn new(
        func: &'a F,
        env: &'a MachineEnv,
        cfginfo: CFGInfo,
        annotations_enabled: bool,
    ) -> Self {
        let n = func.num_insts();
        Self {
            func,
            env,
            cfginfo,

            liveins: Vec::with_capacity(func.num_blocks()),
            liveouts: Vec::with_capacity(func.num_blocks()),
            blockparam_outs: vec![],
            blockparam_ins: vec![],
            bundles: LiveBundles::with_capacity(n),
            ranges: LiveRanges::with_capacity(4 * n),
            spillsets: SpillSets::with_capacity(n),
            vregs: VRegs::with_capacity(n),
            pregs: vec![],
            allocation_queue: PrioQueue::new(),
            safepoints: vec![],
            safepoints_per_vreg: HashMap::new(),
            spilled_bundles: vec![],
            spillslots: vec![],
            slots_by_class: [
                SpillSlotList::new(),
                SpillSlotList::new(),
                SpillSlotList::new(),
            ],
            allocated_bundle_count: 0,

            extra_spillslots_by_class: [smallvec![], smallvec![], smallvec![]],
            preferred_victim_by_class: [PReg::invalid(), PReg::invalid(), PReg::invalid()],

            multi_fixed_reg_fixups: vec![],
            allocs: Vec::with_capacity(4 * n),
            inst_alloc_offsets: vec![],
            num_spillslots: 0,
            safepoint_slots: vec![],
            debug_locations: vec![],

            stats: Stats::default(),

            debug_annotations: hashbrown::HashMap::new(),
            annotations_enabled,

            conflict_set: Default::default(),
        }
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

    pub(crate) fn run(&mut self) -> Result<Edits, RegAllocError> {
        self.process_bundles()?;
        self.try_allocating_regs_for_spilled_bundles();
        self.allocate_spillslots();
        let moves = self.apply_allocations_and_insert_moves();
        let edits = self.resolve_inserted_moves(moves);
        self.compute_stackmaps();
        Ok(edits)
    }
}

pub fn run<F: Function>(
    func: &F,
    mach_env: &MachineEnv,
    enable_annotations: bool,
    enable_ssa_checker: bool,
) -> Result<Output, RegAllocError> {
    let cfginfo = CFGInfo::new(func)?;

    if enable_ssa_checker {
        validate_ssa(func, &cfginfo)?;
    }

    let mut env = Env::new(func, mach_env, cfginfo, enable_annotations);
    env.init()?;

    let edits = env.run()?;

    if enable_annotations {
        env.dump_results();
    }

    Ok(Output {
        edits: edits.into_edits().collect(),
        allocs: env.allocs,
        inst_alloc_offsets: env.inst_alloc_offsets,
        num_spillslots: env.num_spillslots as usize,
        debug_locations: env.debug_locations,
        safepoint_slots: env.safepoint_slots,
        stats: env.stats,
    })
}
