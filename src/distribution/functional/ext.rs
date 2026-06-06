//! Functional extension traits that compose any distribution implementing the
//! spine: log-likelihood, interval probability, truncation, and bijective
//! transforms. Each is blanket-implemented, so the combinators are available on
//! every distribution without per-type boilerplate.

use crate::distribution::{
    Affine, Bijector, Continuous, ContinuousCdf, Discrete, Exp, Sigmoid, Transformed, Truncated,
};
use crate::error::BayesError;
use crate::function::stable::ln_sub_exp;

/// Functional operations for any continuous distribution.
pub trait ContinuousDistributionExt: Continuous + Sized {
    /// Log-likelihood `Σ ln f(xᵢ)`. Empty samples have log-likelihood `0`.
    fn ln_likelihood(&self, data: &[f64]) -> f64 {
        data.iter().map(|&x| self.ln_pdf(x)).sum()
    }

    /// Likelihood `Π f(xᵢ)`, computed as `exp(log-likelihood)`.
    fn likelihood(&self, data: &[f64]) -> f64 {
        self.ln_likelihood(data).exp()
    }

    /// Push the distribution through a bijector.
    fn transformed<B: Bijector>(self, bijector: B) -> Transformed<Self, B> {
        Transformed::new(self, bijector)
    }

    /// Shift and scale the distribution by `scale * x + shift`.
    fn affine(self, scale: f64, shift: f64) -> Result<Transformed<Self, Affine>, BayesError> {
        Ok(self.transformed(Affine::new(scale, shift)?))
    }

    /// Push through `exp(x)`.
    fn exp(self) -> Transformed<Self, Exp> {
        self.transformed(Exp)
    }

    /// Push through the logistic sigmoid.
    fn sigmoid(self) -> Transformed<Self, Sigmoid> {
        self.transformed(Sigmoid)
    }
}

impl<D> ContinuousDistributionExt for D where D: Continuous + Sized {}

/// Functional operations for any continuous distribution with a CDF.
pub trait ContinuousCdfExt: ContinuousCdf + Sized {
    /// Probability mass in `(lo, hi]`.
    fn probability_between(&self, lo: f64, hi: f64) -> f64 {
        if lo > hi || lo.is_nan() || hi.is_nan() {
            return f64::NAN;
        }
        let p = self.cdf(hi) - self.cdf(lo);
        if p.is_nan() { f64::NAN } else { p.max(0.0) }
    }

    /// Log probability mass in `(lo, hi]`, using CDF or survival logs to reduce
    /// cancellation in tails.
    fn ln_probability_between(&self, lo: f64, hi: f64) -> f64 {
        if lo > hi || lo.is_nan() || hi.is_nan() {
            return f64::NAN;
        }
        let cdf_hi = self.cdf(hi);
        if cdf_hi.is_nan() {
            return f64::NAN;
        }
        if cdf_hi <= 0.5 {
            ln_sub_exp(self.ln_cdf(hi), self.ln_cdf(lo))
        } else {
            ln_sub_exp(self.ln_sf(lo), self.ln_sf(hi))
        }
    }

    /// Truncate and renormalize to `[lo, hi]`.
    fn truncated(self, lo: f64, hi: f64) -> Result<Truncated<Self>, BayesError> {
        Truncated::new(self, lo, hi)
    }
}

impl<D> ContinuousCdfExt for D where D: ContinuousCdf + Sized {}

/// Functional operations for any discrete distribution.
pub trait DiscreteDistributionExt<X>: Discrete<X>
where
    X: Copy,
{
    /// Log-likelihood `Σ ln P(X = xᵢ)`. Empty samples have log-likelihood `0`.
    fn ln_likelihood(&self, data: &[X]) -> f64 {
        data.iter().map(|&x| self.ln_pmf(x)).sum()
    }

    /// Likelihood `Π P(X = xᵢ)`, computed as `exp(log-likelihood)`.
    fn likelihood(&self, data: &[X]) -> f64 {
        self.ln_likelihood(data).exp()
    }
}

impl<D, X> DiscreteDistributionExt<X> for D
where
    D: Discrete<X>,
    X: Copy,
{
}
