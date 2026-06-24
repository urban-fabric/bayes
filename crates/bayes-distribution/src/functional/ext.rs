use crate::{BayesError, Continuous, ContinuousCdf, Discrete, OrderStatistic};
use euclid::{Boolean, Element, Natural, Real};

/// Functional operations for any continuous distribution.
pub trait ContinuousDistributionExt<T: Real = f64>: Continuous<T> + Sized {
    /// Log-likelihood `sum ln f(x_i)`. Empty samples have log-likelihood `0`.
    fn ln_likelihood(&self, data: &[T]) -> T {
        data.iter()
            .cloned()
            .fold(T::zero(), |acc, x| acc + self.ln_pdf(x))
    }

    /// Likelihood `prod f(x_i)`, computed as `exp(log-likelihood)`.
    fn likelihood(&self, data: &[T]) -> T {
        self.ln_likelihood(data).exp()
    }
}

impl<T, D> ContinuousDistributionExt<T> for D
where
    T: Real,
    D: Continuous<T> + Sized,
{
}

/// Functional operations for any continuous distribution with a CDF.
pub trait ContinuousCdfExt<T: Real = f64>: ContinuousCdf<T> + Sized {
    /// Probability mass in `(lo, hi]`.
    fn probability_between(&self, lo: T, hi: T) -> T {
        let invalid = lo.gt(&hi).or(&lo.is_nan()).or(&hi.is_nan());
        let p = self.cdf(hi) - self.cdf(lo);
        T::select(&invalid.or(&p.is_nan()), T::nan(), p.max(T::zero()))
    }

    /// Log probability mass in `(lo, hi]`, using CDF or survival logs to reduce
    /// cancellation in tails.
    fn ln_probability_between(&self, lo: T, hi: T) -> T {
        let invalid = lo.gt(&hi).or(&lo.is_nan()).or(&hi.is_nan());
        let cdf_hi = self.cdf(hi.clone());
        let cdf_part = self.ln_cdf(hi.clone()).ln_sub_exp(self.ln_cdf(lo.clone()));
        let sf_part = self.ln_sf(lo).ln_sub_exp(self.ln_sf(hi));
        let use_cdf = cdf_hi.le(&T::half());
        T::select(
            &invalid.or(&cdf_hi.is_nan()),
            T::nan(),
            T::select(&use_cdf, cdf_part, sf_part),
        )
    }

    /// Distribution of the `rank`-th order statistic from `sample_size`
    /// independent draws.
    fn order_statistic(
        self,
        sample_size: u64,
        rank: u64,
    ) -> Result<OrderStatistic<Self>, BayesError>
    where
        Self: Continuous<T>,
    {
        OrderStatistic::new(self, sample_size, rank)
    }

    /// Distribution of the sample minimum from `sample_size` independent draws.
    fn minimum_of(self, sample_size: u64) -> Result<OrderStatistic<Self>, BayesError>
    where
        Self: Continuous<T>,
    {
        OrderStatistic::minimum(self, sample_size)
    }

    /// Distribution of the sample maximum from `sample_size` independent draws.
    fn maximum_of(self, sample_size: u64) -> Result<OrderStatistic<Self>, BayesError>
    where
        Self: Continuous<T>,
    {
        OrderStatistic::maximum(self, sample_size)
    }

    /// Reliability cumulative hazard `H(x) = -ln S(x)`.
    fn cumulative_hazard(&self, x: T) -> T {
        -self.ln_sf(x)
    }

    /// Log hazard `ln h(x) = ln f(x) - ln S(x)`.
    fn ln_hazard(&self, x: T) -> T
    where
        Self: Continuous<T>,
    {
        self.ln_pdf(x.clone()) - self.ln_sf(x)
    }

    /// Hazard rate `h(x) = f(x) / S(x)`.
    fn hazard(&self, x: T) -> T
    where
        Self: Continuous<T>,
    {
        self.ln_hazard(x).exp()
    }
}

impl<T, D> ContinuousCdfExt<T> for D
where
    T: Real,
    D: ContinuousCdf<T> + Sized,
{
}

/// Functional operations for any discrete distribution.
pub trait DiscreteDistributionExt<X>: Discrete<X>
where
    X: Copy + Natural,
{
    /// Log-likelihood `sum ln P(X = x_i)`. Empty samples have log-likelihood `0`.
    fn ln_likelihood(&self, data: &[X]) -> X::Real {
        data.iter()
            .copied()
            .fold(X::Real::zero(), |acc, x| acc + self.ln_pmf(x))
    }

    /// Likelihood `prod P(X = x_i)`, computed as `exp(log-likelihood)`.
    fn likelihood(&self, data: &[X]) -> X::Real {
        self.ln_likelihood(data).exp()
    }
}

impl<D, X> DiscreteDistributionExt<X> for D
where
    D: Discrete<X>,
    X: Copy + Natural,
{
}
