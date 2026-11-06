use crate::{BayesError, Beta, Continuous, ContinuousCdf, Support};
use euclid_numeric::{Natural, Real};

/// Distribution of the `rank`-th order statistic from `sample_size` independent
/// draws of a continuous base distribution.
///
/// For base CDF/PDF `F, f`, the density is
/// `n! / ((k-1)!(n-k)!) F(x)^(k-1) (1-F(x))^(n-k) f(x)`.
/// Equivalently, `F(X_(k)) ~ Beta(k, n+1-k)`, which supplies the quantile.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderStatistic<D> {
    base: D,
    sample_size: u64,
    rank: u64,
}

impl<D> OrderStatistic<D> {
    /// Build the `rank`-th order statistic from `sample_size` iid draws.
    ///
    /// Uses one-based ranks: `rank = 1` is the minimum and `rank = sample_size`
    /// is the maximum.
    pub fn new(base: D, sample_size: u64, rank: u64) -> Result<Self, BayesError> {
        if sample_size == 0 || rank == 0 || rank > sample_size {
            return Err(BayesError::Malformed {
                what: "order statistic requires 1 <= rank <= sample_size",
            });
        }
        Ok(Self {
            base,
            sample_size,
            rank,
        })
    }

    /// Distribution of the sample minimum from `sample_size` iid draws.
    pub fn minimum(base: D, sample_size: u64) -> Result<Self, BayesError> {
        Self::new(base, sample_size, 1)
    }

    /// Distribution of the sample maximum from `sample_size` iid draws.
    pub fn maximum(base: D, sample_size: u64) -> Result<Self, BayesError> {
        Self::new(base, sample_size, sample_size)
    }

    pub fn base(&self) -> &D {
        &self.base
    }

    pub fn sample_size(&self) -> u64 {
        self.sample_size
    }

    /// One-based rank of the statistic.
    pub fn rank(&self) -> u64 {
        self.rank
    }

    fn cdf_order<T: Real>(&self) -> Beta<T> {
        Beta::new(
            T::scalar(self.rank as f64),
            T::scalar((self.sample_size - self.rank + 1) as f64),
        )
    }

    fn log_density_normalizer<T: Real>(&self) -> T {
        let left = self.rank - 1;
        T::scalar(self.sample_size.ln_binomial(&left) + ((self.sample_size - left) as f64).ln())
    }
}

impl<T, D> Support<T> for OrderStatistic<D>
where
    D: Support<T>,
{
    fn min(&self) -> T {
        self.base.min()
    }

    fn max(&self) -> T {
        self.base.max()
    }
}

impl<T, D> Continuous<T> for OrderStatistic<D>
where
    T: Real,
    D: Continuous<T> + ContinuousCdf<T>,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let base_ln_pdf = self.base.ln_pdf(x.clone());
        let u = self.base.cdf(x).clamp(T::zero(), T::one());
        let left = self.rank - 1;
        let right = self.sample_size - self.rank;
        let body: T = self.log_density_normalizer::<T>()
            + ln_power_probability(u.clone(), left)
            + ln_power_probability(T::one() - u, right)
            + base_ln_pdf;
        body.nan_to_num(T::neg_infinity())
    }
}

impl<T, D> ContinuousCdf<T> for OrderStatistic<D>
where
    T: Real,
    D: ContinuousCdf<T>,
{
    fn cdf(&self, x: T) -> T {
        let u = self.base.cdf(x).clamp(T::zero(), T::one());
        self.cdf_order::<T>().cdf(u).clamp(T::zero(), T::one())
    }

    fn inverse_cdf(&self, p: T) -> T {
        self.base.inverse_cdf(
            self.cdf_order::<T>()
                .inverse_cdf(p.clamp(T::zero(), T::one())),
        )
    }
}

fn ln_power_probability<T: Real>(p: T, exponent: u64) -> T {
    if exponent == 0 {
        T::zero()
    } else {
        T::scalar(exponent as f64) * p.ln()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::Uniform;
    use bayes_core::assert_close;

    #[test]
    fn uniform_order_statistic_matches_beta_formula() {
        let third = OrderStatistic::new(Uniform::new(0.0, 1.0), 5, 3).unwrap();
        assert_close!(third.pdf(0.5), 1.875, 1e-12);
        assert_close!(third.cdf(0.5), 0.5, 1e-12);
        assert_close!(third.inverse_cdf(0.5), 0.5, 1e-9);
    }

    #[test]
    fn min_and_max_have_closed_form_cdfs() {
        let min = OrderStatistic::minimum(Uniform::new(0.0, 1.0), 4).unwrap();
        let max = OrderStatistic::maximum(Uniform::new(0.0, 1.0), 4).unwrap();
        assert_close!(min.cdf(0.25), 1.0 - 0.75_f64.powi(4), 1e-12);
        assert_close!(max.cdf(0.25), 0.25_f64.powi(4), 1e-12);
        assert!(OrderStatistic::new(Uniform::new(0.0, 1.0), 3, 0).is_err());
    }
}
