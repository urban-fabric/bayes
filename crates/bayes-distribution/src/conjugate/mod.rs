//! Shared building blocks for closed-form conjugate updates.
//!
//! This module intentionally contains only dependency-light algebraic pieces.
//! Eager conjugate model wrappers and posterior-predictive distributions remain
//! in `bayes-distribution-ext`.

mod model;
mod natural;

pub use model::Conjugate;
pub use natural::{Counts, NaturalParams, NigNatural, Pair};
