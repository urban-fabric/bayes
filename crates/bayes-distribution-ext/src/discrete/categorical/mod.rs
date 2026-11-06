use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::estimation::sample::category_counts;
use crate::{Discrete, DiscreteCdf, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;

/// The categorical distribution over the integer categories `{0, …, k-1}` with
/// (normalized) probabilities `p`. Built from non-negative, not-all-zero weights.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Categorical, Discrete};
///
/// // Weights are normalized, so they need not sum to one.
/// let c = Categorical::new(&[2.0, 3.0, 5.0])?;
/// assert!((c.probabilities()[2] - 0.5).abs() < 1e-12);
/// assert!((c.pmf(0_u64) - 0.2).abs() < 1e-12);
///
/// // Outside the listed categories there is no mass.
/// assert_eq!(c.pmf(3_u64), 0.0);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Categorical {
    /// Normalized category probabilities.
    p: Vec<f64>,
    /// Cumulative probabilities (last entry = 1), for sampling and the CDF.
    cumulative: Vec<f64>,
}

impl Categorical {
    pub fn new(weights: &[f64]) -> Result<Self, BayesError> {
        if weights.is_empty() {
            return Err(BayesError::Malformed {
                what: "categorical needs at least one category",
            });
        }
        let mut sum = 0.0;
        for &w in weights {
            if !w.is_finite() || w < 0.0 {
                return Err(BayesError::Malformed {
                    what: "weights must be finite and non-negative",
                });
            }
            sum += w;
        }
        if sum <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "weight sum",
                value: sum,
            });
        }
        let p: Vec<f64> = weights.iter().map(|&w| w / sum).collect();
        let mut cumulative = Vec::with_capacity(p.len());
        let mut acc = 0.0;
        for &pi in &p {
            acc += pi;
            cumulative.push(acc);
        }
        Ok(Self { p, cumulative })
    }

    pub fn categories(&self) -> usize {
        self.p.len()
    }
    pub fn probabilities(&self) -> &[f64] {
        &self.p
    }
}

impl Distribution<u64> for Categorical {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        let u: f64 = rng.random();
        match self.cumulative.iter().position(|&c| u <= c) {
            Some(i) => i as u64,
            None => (self.p.len() - 1) as u64,
        }
    }
}

impl Discrete<u64> for Categorical {
    fn pmf(&self, k: u64) -> f64 {
        self.p.get(k as usize).copied().unwrap_or(0.0)
    }
    fn ln_pmf(&self, k: u64) -> f64 {
        self.p
            .get(k as usize)
            .copied()
            .map(kernels::categorical_log_pmf)
            .unwrap_or(f64::NEG_INFINITY)
    }
}

impl Support<u64> for Categorical {
    fn min(&self) -> u64 {
        0
    }
    fn max(&self) -> u64 {
        (self.p.len() - 1) as u64
    }
}

impl DiscreteCdf<u64> for Categorical {
    fn cdf(&self, k: u64) -> f64 {
        let i = k as usize;
        if i >= self.cumulative.len() {
            1.0
        } else {
            self.cumulative[i]
        }
    }
    fn inverse_cdf(&self, p: f64) -> u64 {
        match self.cumulative.iter().position(|&c| c >= p) {
            Some(i) => i as u64,
            None => (self.p.len() - 1) as u64,
        }
    }
}

impl Moments for Categorical {
    /// Treating the categories as the integers `0..k`, the mean index `Σ i·pᵢ`.
    fn mean(&self) -> f64 {
        self.p
            .iter()
            .enumerate()
            .map(|(i, &pi)| i as f64 * pi)
            .sum()
    }
    fn variance(&self) -> f64 {
        let m = self.mean();

        self.p
            .iter()
            .enumerate()
            .map(|(i, &pi)| (i as f64 - m).powi(2) * pi)
            .sum()
    }
    fn mode(&self) -> f64 {
        self.p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as f64)
            .unwrap_or(f64::NAN)
    }
    /// Shannon entropy `H = −Σ pᵢ ln pᵢ` (with `0 ln 0 = 0`).
    fn entropy(&self) -> f64 {
        -self
            .p
            .iter()
            .filter(|&&pi| pi > 0.0)
            .map(|&pi| pi * pi.ln())
            .sum::<f64>()
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn skewness(&self) -> f64 {
        f64::NAN
    }

    fn kurtosis(&self) -> f64 {
        f64::NAN
    }

    fn median(&self) -> f64 {
        f64::NAN
    }
}

/// The observed frequencies, which are the maximum likelihood estimates.
///
/// The number of categories is taken as `max + 1`. So a category that was never
/// observed *at the top of the range* is simply not in the fit, while one never
/// observed *in the middle* gets weight zero. That asymmetry cannot be avoided
/// from a bare sample: nothing in the data distinguishes "there is no such
/// category" from "that category did not come up". Pass the width explicitly
/// via [`Categorical::new`] if it is known.
impl crate::Estimate for Categorical {
    fn from_sample(data: &[f64]) -> Option<Self> {
        Categorical::new(&category_counts(data)?).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn normalizes_weights_and_pmf_cdf() {
        let c = Categorical::new(&[1.0, 3.0]).unwrap();
        assert_close!(c.pmf(0), 0.25, 1e-12);
        assert_close!(c.pmf(1), 0.75, 1e-12);
        assert_close!(c.cdf(0), 0.25, 1e-12);
        assert_close!(c.cdf(1), 1.0, 1e-12);
        assert_eq!(c.mode(), 1.0);
        assert_close!(c.mean(), 0.75, 1e-12);
    }

    #[test]
    fn sampled_frequencies_match() {
        let mut rng = ChaCha8Rng::seed_from_u64(6);
        let c = Categorical::new(&[0.2, 0.3, 0.5]).unwrap();
        let n = 50_000;
        let mut counts = [0u64; 3];
        for _ in 0..n {
            counts[c.sample(&mut rng) as usize] += 1;
        }
        assert_close!(counts[0] as f64 / n as f64, 0.2, 0.02);
        assert_close!(counts[2] as f64 / n as f64, 0.5, 0.02);
    }

    #[test]
    fn rejects_degenerate_weights() {
        assert!(Categorical::new(&[]).is_err());
        assert!(Categorical::new(&[0.0, 0.0]).is_err());
        assert!(Categorical::new(&[1.0, -1.0]).is_err());
    }
}
