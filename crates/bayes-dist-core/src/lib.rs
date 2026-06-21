//! Compatibility facade for the split core crates.
//!
//! New lazy-only code should depend on `bayes-core`. Eager distribution
//! fitting, constraints, and probability utilities live in `bayes-core-ext`.

pub use bayes_core_ext::*;
