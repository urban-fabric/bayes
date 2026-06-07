use rand::distr::Distribution;
use rand::{Rng, RngExt};

use super::MultivariateMoments;
use crate::error::BayesError;
use crate::function::factorial::ln_factorial;
use crate::linalg::Matrix;

/// The multinomial distribution: the counts in each of `k` categories after `n`
/// independent trials with category probabilities `p₁, …, p_k` (`Σ pᵢ = 1`).
#[derive(Clone, Debug, PartialEq)]
pub struct Multinomial {
    n: u64,
    p: Vec<f64>,
    cumulative: Vec<f64>,
}

impl Multinomial {
    pub fn new(n: u64, probabilities: &[f64]) -> Result<Self, BayesError> {
        if probabilities.len() < 2 {
            return Err(BayesError::Malformed {
                what: "multinomial needs at least two categories",
            });
        }
        let mut sum = 0.0;
        for &p in probabilities {
            if !p.is_finite() || p < 0.0 {
                return Err(BayesError::Malformed {
                    what: "probabilities must be finite, non-negative",
                });
            }
            sum += p;
        }
        if sum <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "probability sum",
                value: sum,
            });
        }
        let p: Vec<f64> = probabilities.iter().map(|&x| x / sum).collect();
        let mut cumulative = Vec::with_capacity(p.len());
        let mut acc = 0.0;
        for &pi in &p {
            acc += pi;
            cumulative.push(acc);
        }
        Ok(Self { n, p, cumulative })
    }

    pub fn trials(&self) -> u64 {
        self.n
    }
    pub fn probabilities(&self) -> &[f64] {
        &self.p
    }
    pub fn categories(&self) -> usize {
        self.p.len()
    }

    /// `ln P(counts)` for a category-count vector summing to `n`. `−∞` if the
    /// counts are the wrong length or do not sum to `n`.
    pub fn ln_pmf(&self, counts: &[u64]) -> f64 {
        if counts.len() != self.p.len() || counts.iter().sum::<u64>() != self.n {
            return f64::NEG_INFINITY;
        }
        let mut acc = ln_factorial(self.n);
        for (&k, &pi) in counts.iter().zip(&self.p) {
            acc -= ln_factorial(k);
            if k > 0 {
                acc += k as f64 * pi.ln();
            }
        }
        acc
    }

    pub fn pmf(&self, counts: &[u64]) -> f64 {
        self.ln_pmf(counts).exp()
    }
}

impl Distribution<Vec<u64>> for Multinomial {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<u64> {
        let mut counts = vec![0u64; self.p.len()];
        for _ in 0..self.n {
            let u: f64 = rng.random();
            let cat = self
                .cumulative
                .iter()
                .position(|&c| u <= c)
                .unwrap_or(self.p.len() - 1);
            counts[cat] += 1;
        }
        counts
    }
}

impl MultivariateMoments for Multinomial {
    fn mean_vector(&self) -> Vec<f64> {
        self.p.iter().map(|&pi| self.n as f64 * pi).collect()
    }
    fn covariance(&self) -> Matrix {
        let n = self.n as f64;
        let k = self.p.len();
        (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| {
                        if i == j {
                            n * self.p[i] * (1.0 - self.p[i])
                        } else {
                            -n * self.p[i] * self.p[j]
                        }
                    })
                    .collect()
            })
            .collect()
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
    fn pmf_matches_binomial_in_two_categories() {
        let m = Multinomial::new(5, &[0.3, 0.7]).unwrap();
        let expect = 10.0 * 0.3f64.powi(2) * 0.7f64.powi(3);
        close(m.pmf(&[2, 3]), expect, 1e-12);
        assert_eq!(m.pmf(&[2, 2]), 0.0);
    }

    #[test]
    fn sampled_counts_match_the_mean() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let m = Multinomial::new(100, &[0.2, 0.3, 0.5]).unwrap();
        let trials = 4_000;
        let mut avg = [0.0; 3];
        for _ in 0..trials {
            let c = m.sample(&mut rng);
            assert_eq!(c.iter().sum::<u64>(), 100);
            for i in 0..3 {
                avg[i] += c[i] as f64 / trials as f64;
            }
        }
        let mean = m.mean_vector();
        for i in 0..3 {
            close(avg[i], mean[i], 0.5);
        }
    }
}
