//! Lazy numeric probability traits.
//!
//! This crate is the low-level contract for write-once probability formulas over
//! [`euclid`] lazy numeric substrates. It deliberately excludes eager statistical
//! conveniences such as fitting and sampling.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod constraint;
mod error;
mod probability;

pub use constraint::{
    Boundary, Constrained, Constraint, IntegerConstraint, NaturalConstraint, RealConstraint,
};
pub use error::{BayesError, StatsError};
pub use probability::{
    PROBABILITY_TOLERANCE, brier_score, brier_score_real, is_probability_vector,
    is_probability_vector_with_tolerance, is_row_stochastic, is_row_stochastic_with_tolerance,
    log_loss, log_loss_real, log_score, log_score_real, normalize_probability_rows,
    normalize_probability_rows_real, normalize_probability_weights,
    normalize_probability_weights_real, probability_sum, probability_sum_real,
    ranked_probability_score, ranked_probability_score_real, spherical_score, spherical_score_real,
    validate_probability_vector, validate_row_stochastic, validate_row_stochastic_with_tolerance,
};

use euclid::{Element, Integer, Natural, Real};

/// Continuous probability density, the PDF `f(x)`.
pub trait Continuous<T: Real = f64> {
    /// The density at `x`.
    fn pdf(&self, x: T) -> T;

    /// `ln f(x)`.
    fn ln_pdf(&self, x: T) -> T;
}

/// Lazy-real continuous density formula on the distribution's support.
///
/// This companion to [`Continuous`] intentionally does not perform support
/// checks or host-`bool` branching. It is the closed-form algebra for points
/// already known to lie on the continuous support.
pub trait ContinuousDensityKernel<T: Real = f64> {
    /// `ln f(x)` over a [`Real`] substrate.
    fn ln_pdf_kernel(&self, x: T) -> T;

    /// `f(x) = exp(ln f(x))` over the same [`Real`] substrate.
    fn pdf_kernel(&self, x: T) -> T;
}

/// Analytic moments and summary points over a lazy-real substrate.
///
/// Undefined or unavailable summaries are represented as `NaN`, so formulas can
/// remain value-level and lazy backends can express parameter-dependent existence
/// with `select` rather than host-side `Option` branching.
pub trait Moments<R: Real = f64> {
    fn mean(&self) -> R;

    fn variance(&self) -> R;

    /// The raw second moment `E[X^2] = Var(X) + E[X]^2`.
    fn moment2(&self) -> R {
        let mean = self.mean();
        self.variance() + mean.clone() * mean
    }

    fn std_dev(&self) -> R;

    /// Third standardized moment.
    fn skewness(&self) -> R;

    /// Excess kurtosis.
    fn kurtosis(&self) -> R;

    /// Differential / Shannon entropy, in nats.
    fn entropy(&self) -> R;

    fn median(&self) -> R;

    fn mode(&self) -> R;
}

/// The support bounds, the infimum/supremum of the domain.
pub trait Support<X = f64> {
    fn min(&self) -> X;
    fn max(&self) -> X;
}

/// Cumulative distribution, survival, and inverse for a continuous distribution.
pub trait ContinuousCdf<X: Real = f64>: Support<X> {
    /// `P(X <= x)`.
    fn cdf(&self, x: X) -> X;

    /// Survival `P(X > x) = 1 - cdf(x)`.
    fn sf(&self, x: X) -> X {
        <X as Element>::one() - self.cdf(x)
    }

    /// The quantile / inverse CDF.
    fn inverse_cdf(&self, p: X) -> X;

    /// `ln P(X <= x)`.
    fn ln_cdf(&self, x: X) -> X {
        self.cdf(x).ln()
    }

    /// `ln P(X > x)`.
    fn ln_sf(&self, x: X) -> X {
        self.sf(x).ln()
    }

    /// The inverse survival function.
    fn isf(&self, p: X) -> X {
        self.inverse_cdf(<X as Element>::one() - p)
    }
}

/// Cumulative distribution, survival, and inverse for a natural-count discrete
/// distribution.
pub trait DiscreteCdf<K: Natural = u64>: Support<K> {
    /// `P(X <= k)`.
    fn cdf(&self, k: K) -> K::Real;

    /// Survival `P(X > k) = 1 - cdf(k)`.
    fn sf(&self, k: K) -> K::Real
    where
        K: Copy,
    {
        <K::Real as Element>::one() - self.cdf(k)
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
        self.inverse_cdf(<K::Real as Element>::one() - p)
    }
}

/// Cumulative distribution, survival, and inverse for a signed-integer discrete
/// distribution.
pub trait IntegerDiscreteCdf<K: Integer = i64>: Support<K> {
    /// `P(X <= k)`.
    fn cdf(&self, k: K) -> K::Real;

    /// Survival `P(X > k) = 1 - cdf(k)`.
    fn sf(&self, k: K) -> K::Real
    where
        K: Copy,
    {
        <K::Real as Element>::one() - self.cdf(k)
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
        self.inverse_cdf(<K::Real as Element>::one() - p)
    }
}

/// Discrete probability mass, the PMF `P(X = k)` over natural counts.
pub trait Discrete<K: Natural = u64> {
    /// The mass at `k`.
    fn pmf(&self, k: K) -> K::Real;

    /// `ln P(X = k)`.
    fn ln_pmf(&self, k: K) -> K::Real;
}

/// Signed-integer discrete probability mass for distributions whose support
/// lives on `Z` rather than `N`.
pub trait IntegerDiscrete<K: Integer = i64> {
    /// The mass at `k`.
    fn pmf(&self, k: K) -> K::Real;

    /// `ln P(X = k)`.
    fn ln_pmf(&self, k: K) -> K::Real;
}

/// Lazy-real discrete mass formula on a natural-count support.
///
/// Counts are represented by [`Natural`] so formulas can use exact
/// combinatorial helpers before crossing into the associated [`Real`]
/// analysis surface.
pub trait DiscreteMassKernel<K: Natural = u64> {
    /// `ln P(X = k)` over a lazy natural count substrate.
    fn ln_pmf_kernel(&self, k: K) -> K::Real;

    /// `P(X = k) = exp(ln P(X = k))` over the associated lazy real substrate.
    fn pmf_kernel(&self, k: K) -> K::Real;
}

/// Lazy-real signed-integer mass formula on an integer support.
pub trait IntegerMassKernel<K: Integer = i64> {
    /// `ln P(X = k)` over a lazy integer substrate.
    fn ln_pmf_kernel(&self, k: K) -> K::Real;

    /// `P(X = k) = exp(ln P(X = k))` over the associated lazy real substrate.
    fn pmf_kernel(&self, k: K) -> K::Real;
}
