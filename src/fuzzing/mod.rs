/*
 * Released under the terms of the Apache 2.0 license with LLVM
 * exception. See `LICENSE` for details.
 */

//! Utilities for fuzzing.

pub mod func;

// Re-exports for fuzz targets.

pub mod domtree {
    pub use crate::domtree::*;
}
pub mod postorder {
    pub use crate::postorder::*;
}
pub mod moves {
    pub use crate::moves::*;
}
pub mod cfg {
    pub use crate::cfg::*;
}
pub mod ssa {
    pub use crate::ssa::*;
}
pub mod ion {
    pub use crate::ion::*;
}
pub mod checker {
    pub use crate::checker::*;
}
