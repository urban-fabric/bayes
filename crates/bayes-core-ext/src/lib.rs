//! Eager distribution extensions and probability utilities.
//!
//! This crate layers host-side statistical surfaces on top of
//! [`bayes_core`]: fitting and probability-vector utilities.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

mod error;
pub mod numeric;
mod probability;

pub use bayes_core::{
    Boundary, Constrained, Constraint, Continuous, ContinuousCdf, ContinuousDensityKernel,
    Discrete, DiscreteCdf, DiscreteMassKernel, IntegerConstraint, IntegerDiscrete,
    IntegerDiscreteCdf, IntegerMassKernel, Moments, NaturalConstraint, RealConstraint, Support,
};
pub use error::{BayesError, StatsError};
use euclid::Real;
pub use probability::{
    PROBABILITY_TOLERANCE, brier_score, is_probability_vector,
    is_probability_vector_with_tolerance, is_row_stochastic, is_row_stochastic_with_tolerance,
    log_loss, log_score, normalize_probability_rows, normalize_probability_weights,
    probability_sum, ranked_probability_score, spherical_score, validate_probability_vector,
    validate_row_stochastic, validate_row_stochastic_with_tolerance,
};

/// Fit a distribution's parameters from a sample.
pub trait Estimate<R: Real = f64>: Sized {
    fn from_sample(data: &[R]) -> Option<Self>;
}
