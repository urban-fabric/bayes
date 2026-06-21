//! Functional extension traits that compose any distribution implementing the
//! spine: log-likelihood, interval probability, truncation, and bijective
//! transforms. Each is blanket-implemented, so the combinators are available on
//! every distribution without per-type boilerplate.

use crate::{
    Affine, Bijector, Continuous, ContinuousCdf, Discrete, Exp, OrderStatistic, Sigmoid,
    Transformed, Truncated,
};
use bayes_core_ext::BayesError;
use euclid::{Natural, Real};

fn ln_sub_exp_f64(a: f64, b: f64) -> f64 {
    if a < b {
        f64::NAN
    } else if a == b {
        f64::NEG_INFINITY
    } else {
        a.ln_sub_exp(b)
    }
}

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
            ln_sub_exp_f64(self.ln_cdf(hi), self.ln_cdf(lo))
        } else {
            ln_sub_exp_f64(self.ln_sf(lo), self.ln_sf(hi))
        }
    }

    /// Truncate and renormalize to `[lo, hi]`.
    fn truncated(self, lo: f64, hi: f64) -> Result<Truncated<Self>, BayesError> {
        Truncated::new(self, lo, hi)
    }

    /// Distribution of the `rank`-th order statistic from `sample_size`
    /// independent draws.
    fn order_statistic(
        self,
        sample_size: u64,
        rank: u64,
    ) -> Result<OrderStatistic<Self>, BayesError>
    where
        Self: Continuous,
    {
        OrderStatistic::new(self, sample_size, rank)
    }

    /// Distribution of the sample minimum from `sample_size` independent draws.
    fn minimum_of(self, sample_size: u64) -> Result<OrderStatistic<Self>, BayesError>
    where
        Self: Continuous,
    {
        OrderStatistic::minimum(self, sample_size)
    }

    /// Distribution of the sample maximum from `sample_size` independent draws.
    fn maximum_of(self, sample_size: u64) -> Result<OrderStatistic<Self>, BayesError>
    where
        Self: Continuous,
    {
        OrderStatistic::maximum(self, sample_size)
    }

    /// Reliability cumulative hazard `H(x) = -ln S(x)`.
    fn cumulative_hazard(&self, x: f64) -> f64 {
        -self.ln_sf(x)
    }

    /// Log hazard `ln h(x) = ln f(x) - ln S(x)`.
    fn ln_hazard(&self, x: f64) -> f64
    where
        Self: Continuous,
    {
        self.ln_pdf(x) - self.ln_sf(x)
    }

    /// Hazard rate `h(x) = f(x) / S(x)`.
    fn hazard(&self, x: f64) -> f64
    where
        Self: Continuous,
    {
        self.ln_hazard(x).exp()
    }
}

impl<D> ContinuousCdfExt for D where D: ContinuousCdf + Sized {}

/// Functional operations for any discrete distribution.
pub trait DiscreteDistributionExt<X>: Discrete<X>
where
    X: Copy + Natural<Real = f64>,
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
    X: Copy + Natural<Real = f64>,
{
}
