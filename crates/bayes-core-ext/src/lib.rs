//! Eager distribution support traits and probability utilities.
//!
//! This crate layers host-side statistical surfaces on top of
//! [`bayes_core_lazy`]: support metadata, CDF/quantile traits, analytic moments,
//! fitting, constraints, and probability-vector utilities.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod constraint;
pub mod numeric;
mod probability;

pub use bayes_core_lazy::{
    Continuous, ContinuousDensityKernel, Discrete, DiscreteMassKernel, IntegerDiscrete,
    IntegerMassKernel,
};
pub use constraint::{Constrained, Constraint};
use euclid::{LazyInteger, LazyNatural, LazyReal};
pub use probability::{
    PROBABILITY_TOLERANCE, brier_score, is_probability_vector,
    is_probability_vector_with_tolerance, is_row_stochastic, is_row_stochastic_with_tolerance,
    log_loss, log_score, normalize_probability_rows, normalize_probability_weights,
    probability_sum, ranked_probability_score, spherical_score, validate_probability_vector,
    validate_row_stochastic, validate_row_stochastic_with_tolerance,
};

/// The support bounds, the infimum/supremum of the domain.
pub trait Support<X = f64> {
    fn min(&self) -> X;
    fn max(&self) -> X;
}

/// Cumulative distribution, survival, and inverse for a continuous distribution.
pub trait ContinuousCdf<X: LazyReal = f64>: Support<X> {
    /// `P(X <= x)`.
    fn cdf(&self, x: X) -> X;

    /// Survival `P(X > x) = 1 - cdf(x)`.
    fn sf(&self, x: X) -> X
    where
        X: Copy,
    {
        X::one() - self.cdf(x)
    }

    /// The quantile / inverse CDF.
    fn inverse_cdf(&self, p: X) -> X;

    /// `ln P(X <= x)`.
    fn ln_cdf(&self, x: X) -> X {
        self.cdf(x).ln()
    }

    /// `ln P(X > x)`.
    fn ln_sf(&self, x: X) -> X
    where
        X: Copy,
    {
        self.sf(x).ln()
    }

    /// The inverse survival function.
    fn isf(&self, p: X) -> X {
        self.inverse_cdf(X::one() - p)
    }
}

/// Cumulative distribution, survival, and inverse for a natural-count discrete
/// distribution.
pub trait DiscreteCdf<K: LazyNatural = u64>: Support<K> {
    /// `P(X <= k)`.
    fn cdf(&self, k: K) -> K::Real;

    /// Survival `P(X > k) = 1 - cdf(k)`.
    fn sf(&self, k: K) -> K::Real
    where
        K: Copy,
    {
        K::Real::one() - self.cdf(k)
    }

    /// The smallest `k` with `cdf(k) >= p`.
    fn inverse_cdf(&self, p: K::Real) -> K;

    /// `ln P(X <= k)`.
    fn ln_cdf(&self, k: K) -> K::Real {
        self.cdf(k).ln()
    }

    /// `ln P(X > k)`.
    fn ln_sf(&self, k: K) -> K::Real
    where
        K: Copy,
    {
        self.sf(k).ln()
    }

    /// The inverse survival function.
    fn isf(&self, p: K::Real) -> K {
        self.inverse_cdf(K::Real::one() - p)
    }
}

/// Cumulative distribution, survival, and inverse for a signed-integer discrete
/// distribution.
pub trait IntegerDiscreteCdf<K: LazyInteger = i64>: Support<K> {
    /// `P(X <= k)`.
    fn cdf(&self, k: K) -> K::Real;

    /// Survival `P(X > k) = 1 - cdf(k)`.
    fn sf(&self, k: K) -> K::Real
    where
        K: Copy,
    {
        K::Real::one() - self.cdf(k)
    }

    /// The smallest `k` with `cdf(k) >= p`.
    fn inverse_cdf(&self, p: K::Real) -> K;

    /// `ln P(X <= k)`.
    fn ln_cdf(&self, k: K) -> K::Real {
        self.cdf(k).ln()
    }

    /// `ln P(X > k)`.
    fn ln_sf(&self, k: K) -> K::Real
    where
        K: Copy,
    {
        self.sf(k).ln()
    }

    /// The inverse survival function.
    fn isf(&self, p: K::Real) -> K {
        self.inverse_cdf(K::Real::one() - p)
    }
}

/// Analytic moments and summary points.
pub trait Moments<R: LazyReal = f64> {
    fn mean(&self) -> Option<R> {
        None
    }

    fn variance(&self) -> Option<R> {
        None
    }

    /// The raw second moment `E[X^2] = Var(X) + E[X]^2`.
    fn moment2(&self) -> Option<R> {
        let mean = self.mean()?;
        Some(self.variance()? + mean.clone() * mean)
    }

    fn std_dev(&self) -> Option<R> {
        self.variance().map(|variance| variance.sqrt())
    }

    /// Third standardized moment.
    fn skewness(&self) -> Option<R> {
        None
    }

    /// Excess kurtosis.
    fn kurtosis(&self) -> Option<R> {
        None
    }

    /// Differential / Shannon entropy, in nats.
    fn entropy(&self) -> Option<R> {
        None
    }

    fn median(&self) -> Option<R> {
        None
    }

    fn mode(&self) -> Option<R> {
        None
    }
}

/// Fit a distribution's parameters from a sample.
pub trait Estimate<R: LazyReal = f64>: Sized {
    fn from_sample(data: &[R]) -> Option<Self>;
}
