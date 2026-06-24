//! Eager distribution extensions and probability utilities.
//!
//! This crate layers host-side statistical surfaces on top of
//! [`bayes_core`]: fitting and probability-vector utilities.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod numeric;

pub use bayes_core::{
    BayesError, Boundary, Constrained, Constraint, Continuous, ContinuousCdf,
    ContinuousDensityKernel, Discrete, DiscreteCdf, DiscreteMassKernel, IntegerConstraint,
    IntegerDiscrete, IntegerDiscreteCdf, IntegerMassKernel, Moments, NaturalConstraint,
    RealConstraint, StatsError, Support,
};
pub use bayes_core::{
    PROBABILITY_TOLERANCE, brier_score, brier_score_real, is_probability_vector,
    is_probability_vector_with_tolerance, is_row_stochastic, is_row_stochastic_with_tolerance,
    log_loss, log_loss_real, log_score, log_score_real, normalize_probability_rows,
    normalize_probability_rows_real, normalize_probability_weights,
    normalize_probability_weights_real, probability_sum, probability_sum_real,
    ranked_probability_score, ranked_probability_score_real, spherical_score, spherical_score_real,
    validate_probability_vector, validate_row_stochastic, validate_row_stochastic_with_tolerance,
};
use euclid::Real;

/// Fit a distribution's parameters from a sample.
pub trait Estimate<R: Real = f64>: Sized {
    fn from_sample(data: &[R]) -> Option<Self>;
}
