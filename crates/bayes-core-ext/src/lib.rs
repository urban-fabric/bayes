//! Eager distribution extensions and probability utilities.
//!
//! This crate layers host-side statistical surfaces on top of
//! [`bayes_core`]: fitting, constraints, and probability-vector utilities.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod constraint;
pub mod numeric;
mod probability;

pub use bayes_core::{
    Continuous, ContinuousCdf, ContinuousDensityKernel, Discrete, DiscreteCdf, DiscreteMassKernel,
    IntegerDiscrete, IntegerDiscreteCdf, IntegerMassKernel, Moments, Support,
};
pub use constraint::{Constrained, Constraint};
use euclid::LazyReal;
pub use probability::{
    PROBABILITY_TOLERANCE, brier_score, is_probability_vector,
    is_probability_vector_with_tolerance, is_row_stochastic, is_row_stochastic_with_tolerance,
    log_loss, log_score, normalize_probability_rows, normalize_probability_weights,
    probability_sum, ranked_probability_score, spherical_score, validate_probability_vector,
    validate_row_stochastic, validate_row_stochastic_with_tolerance,
};

/// Fit a distribution's parameters from a sample.
pub trait Estimate<R: LazyReal = f64>: Sized {
    fn from_sample(data: &[R]) -> Option<Self>;
}
