use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Discrete, DiscreteCdf, Moments, Support};
use crate::error::BayesError;
use crate::lazy;

/// The categorical distribution over the integer categories `{0, …, k-1}` with
/// (normalized) probabilities `p`. Built from non-negative, not-all-zero weights.
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
            .map(lazy::categorical_log_pmf)
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
    fn mean(&self) -> Option<f64> {
        Some(
            self.p
                .iter()
                .enumerate()
                .map(|(i, &pi)| i as f64 * pi)
                .sum(),
        )
    }
    fn variance(&self) -> Option<f64> {
        let m = self.mean()?;
        Some(
            self.p
                .iter()
                .enumerate()
                .map(|(i, &pi)| (i as f64 - m).powi(2) * pi)
                .sum(),
        )
    }
    fn mode(&self) -> Option<f64> {
        self.p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as f64)
    }
    /// Shannon entropy `H = −Σ pᵢ ln pᵢ` (with `0 ln 0 = 0`).
    fn entropy(&self) -> Option<f64> {
        Some(
            -self
                .p
                .iter()
                .filter(|&&pi| pi > 0.0)
                .map(|&pi| pi * pi.ln())
                .sum::<f64>(),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn normalizes_weights_and_pmf_cdf() {
        let c = Categorical::new(&[1.0, 3.0]).unwrap();
        close(c.pmf(0), 0.25, 1e-12);
        close(c.pmf(1), 0.75, 1e-12);
        close(c.cdf(0), 0.25, 1e-12);
        close(c.cdf(1), 1.0, 1e-12);
        assert_eq!(c.mode(), Some(1.0));
        close(c.mean().unwrap(), 0.75, 1e-12);
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
        close(counts[0] as f64 / n as f64, 0.2, 0.02);
        close(counts[2] as f64 / n as f64, 0.5, 0.02);
    }

    #[test]
    fn rejects_degenerate_weights() {
        assert!(Categorical::new(&[]).is_err());
        assert!(Categorical::new(&[0.0, 0.0]).is_err());
        assert!(Categorical::new(&[1.0, -1.0]).is_err());
    }
}
