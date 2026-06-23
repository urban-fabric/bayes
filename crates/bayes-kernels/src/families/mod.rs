//! Distribution-family implementation modules.
//!
//! The implementation home for distribution-specific kernels is here. Use
//! `families::normal::*`, `families::poisson::*`, and similar modules when you
//! want to understand one distribution end-to-end. The crate root re-exports
//! these same functions as a flat surface (`bayes_kernels::normal_cdf`, …).

mod common;

pub mod continuous;
pub mod discrete;

pub use continuous::*;
pub use discrete::*;
