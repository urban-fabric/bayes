//! Extension traits for ext-only distribution combinators.

use crate::{
    Affine, Bijector, Continuous, ContinuousCdf, Discrete, Exp, OrderStatistic, Sigmoid,
    Transformed, Truncated,
};
use bayes_core_ext::BayesError;
use bayes_distribution::functional::{
    ContinuousCdfExt as BaseContinuousCdfExt,
    ContinuousDistributionExt as BaseContinuousDistributionExt,
    DiscreteDistributionExt as BaseDiscreteDistributionExt,
};
use euclid_numeric::Natural;

/// Functional operations for any continuous distribution.
pub trait ContinuousDistributionExt:
    BaseContinuousDistributionExt<f64> + Continuous + Sized
{
    /// Log-likelihood `sum ln f(x_i)`. Empty samples have log-likelihood `0`.
    fn ln_likelihood(&self, data: &[f64]) -> f64 {
        BaseContinuousDistributionExt::ln_likelihood(self, data)
    }

    /// Likelihood `prod f(x_i)`, computed as `exp(log-likelihood)`.
    fn likelihood(&self, data: &[f64]) -> f64 {
        BaseContinuousDistributionExt::likelihood(self, data)
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

impl<D> ContinuousDistributionExt for D where
    D: BaseContinuousDistributionExt<f64> + Continuous + Sized
{
}

/// Functional operations for any continuous distribution with a CDF.
pub trait ContinuousCdfExt: BaseContinuousCdfExt<f64> + ContinuousCdf + Sized {
    /// Probability mass in `(lo, hi]`.
    fn probability_between(&self, lo: f64, hi: f64) -> f64 {
        BaseContinuousCdfExt::probability_between(self, lo, hi)
    }

    /// Log probability mass in `(lo, hi]`.
    fn ln_probability_between(&self, lo: f64, hi: f64) -> f64 {
        BaseContinuousCdfExt::ln_probability_between(self, lo, hi)
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
        BaseContinuousCdfExt::order_statistic(self, sample_size, rank)
    }

    /// Distribution of the sample minimum from `sample_size` independent draws.
    fn minimum_of(self, sample_size: u64) -> Result<OrderStatistic<Self>, BayesError>
    where
        Self: Continuous,
    {
        BaseContinuousCdfExt::minimum_of(self, sample_size)
    }

    /// Distribution of the sample maximum from `sample_size` independent draws.
    fn maximum_of(self, sample_size: u64) -> Result<OrderStatistic<Self>, BayesError>
    where
        Self: Continuous,
    {
        BaseContinuousCdfExt::maximum_of(self, sample_size)
    }

    /// Reliability cumulative hazard `H(x) = -ln S(x)`.
    fn cumulative_hazard(&self, x: f64) -> f64 {
        BaseContinuousCdfExt::cumulative_hazard(self, x)
    }

    /// Log hazard `ln h(x) = ln f(x) - ln S(x)`.
    fn ln_hazard(&self, x: f64) -> f64
    where
        Self: Continuous,
    {
        BaseContinuousCdfExt::ln_hazard(self, x)
    }

    /// Hazard rate `h(x) = f(x) / S(x)`.
    fn hazard(&self, x: f64) -> f64
    where
        Self: Continuous,
    {
        BaseContinuousCdfExt::hazard(self, x)
    }
}

impl<D> ContinuousCdfExt for D where D: BaseContinuousCdfExt<f64> + ContinuousCdf + Sized {}

/// Functional operations for any discrete distribution.
pub trait DiscreteDistributionExt<X>: BaseDiscreteDistributionExt<X> + Discrete<X>
where
    X: Copy + Natural<Real = f64>,
{
    /// Log-likelihood `sum ln P(X = x_i)`. Empty samples have log-likelihood `0`.
    fn ln_likelihood(&self, data: &[X]) -> f64 {
        BaseDiscreteDistributionExt::ln_likelihood(self, data)
    }

    /// Likelihood `prod P(X = x_i)`, computed as `exp(log-likelihood)`.
    fn likelihood(&self, data: &[X]) -> f64 {
        BaseDiscreteDistributionExt::likelihood(self, data)
    }
}

impl<D, X> DiscreteDistributionExt<X> for D
where
    D: BaseDiscreteDistributionExt<X> + Discrete<X>,
    X: Copy + Natural<Real = f64>,
{
}
