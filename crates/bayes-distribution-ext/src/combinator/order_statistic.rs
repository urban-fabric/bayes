use crate::{Beta, Continuous, ContinuousCdf, Support};
use bayes_error::BayesError;
use euclid::LazyNatural;

fn log_sum_exp(xs: &[f64]) -> Option<f64> {
    let (&first, rest) = xs.split_first()?;
    let max = rest.iter().fold(first, |acc, &x| acc.max(x));
    let shifted_sum = xs.iter().map(|&x| (x - max).exp()).sum::<f64>();
    Some(max + shifted_sum.ln())
}

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
    cdf_order: Beta,
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
        let cdf_order = Beta::new(rank as f64, (sample_size - rank + 1) as f64)?;
        Ok(Self {
            base,
            sample_size,
            rank,
            cdf_order,
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
}

impl<D: Support> Support for OrderStatistic<D> {
    fn min(&self) -> f64 {
        self.base.min()
    }

    fn max(&self) -> f64 {
        self.base.max()
    }
}

impl<D: Continuous + ContinuousCdf> Continuous for OrderStatistic<D> {
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        let f = self.base.pdf(x);
        if f <= 0.0 || !f.is_finite() {
            return f64::NEG_INFINITY;
        }
        let u = self.base.cdf(x);
        if !u.is_finite() {
            return f64::NEG_INFINITY;
        }
        let left = self.rank - 1;
        let right = self.sample_size - self.rank;
        let Some(ln_u) = ln_power_probability(u, left) else {
            return f64::NEG_INFINITY;
        };
        let Some(ln_1mu) = ln_power_probability(1.0 - u, right) else {
            return f64::NEG_INFINITY;
        };
        self.sample_size.ln_binomial(&left)
            + ((self.sample_size - left) as f64).ln() // n-k+1 factor
            + ln_u
            + ln_1mu
            + self.base.ln_pdf(x)
    }
}

impl<D: Continuous + ContinuousCdf> ContinuousCdf for OrderStatistic<D> {
    fn cdf(&self, x: f64) -> f64 {
        let u = self.base.cdf(x);
        if u <= 0.0 {
            return 0.0;
        }
        if u >= 1.0 {
            return 1.0;
        }
        if !u.is_finite() {
            return f64::NAN;
        }
        let terms: Vec<f64> = (self.rank..=self.sample_size)
            .filter_map(|j| {
                let ln_u = ln_power_probability(u, j)?;
                let ln_1mu = ln_power_probability(1.0 - u, self.sample_size - j)?;
                Some(self.sample_size.ln_binomial(&j) + ln_u + ln_1mu)
            })
            .collect();
        log_sum_exp(&terms)
            .unwrap_or(f64::NEG_INFINITY)
            .exp()
            .clamp(0.0, 1.0)
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        self.base.inverse_cdf(self.cdf_order.inverse_cdf(p))
    }
}

fn ln_power_probability(p: f64, exponent: u64) -> Option<f64> {
    if exponent == 0 {
        Some(0.0)
    } else if p <= 0.0 || !p.is_finite() {
        None
    } else {
        Some(exponent as f64 * p.ln())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::Uniform;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn uniform_order_statistic_matches_beta_formula() {
        let third = OrderStatistic::new(Uniform::new(0.0, 1.0).unwrap(), 5, 3).unwrap();
        close(third.pdf(0.5), 1.875, 1e-12);
        close(third.cdf(0.5), 0.5, 1e-12);
        close(third.inverse_cdf(0.5), 0.5, 1e-9);
    }

    #[test]
    fn min_and_max_have_closed_form_cdfs() {
        let min = OrderStatistic::minimum(Uniform::new(0.0, 1.0).unwrap(), 4).unwrap();
        let max = OrderStatistic::maximum(Uniform::new(0.0, 1.0).unwrap(), 4).unwrap();
        close(min.cdf(0.25), 1.0 - 0.75_f64.powi(4), 1e-12);
        close(max.cdf(0.25), 0.25_f64.powi(4), 1e-12);
        assert!(OrderStatistic::new(Uniform::new(0.0, 1.0).unwrap(), 3, 0).is_err());
    }
}
