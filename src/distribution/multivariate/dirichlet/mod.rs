use rand::Rng;
use rand::distr::Distribution;

use ramanujan::DenseMatrix;

use super::{MultivariateContinuous, MultivariateMoments};
use crate::distribution::continuous::gamma::sample_standard_gamma;
use crate::error::BayesError;
use crate::function::gamma::{digamma, ln_gamma};

/// The Dirichlet distribution over the `k`-simplex (vectors of non-negative
/// components summing to 1), with concentration parameters `α₁, …, α_k > 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct Dirichlet {
    alpha: Vec<f64>,
}

impl Dirichlet {
    pub fn new(alpha: Vec<f64>) -> Result<Self, BayesError> {
        if alpha.len() < 2 {
            return Err(BayesError::Malformed {
                what: "Dirichlet needs at least two components",
            });
        }
        for &a in &alpha {
            if !a.is_finite() || a <= 0.0 {
                return Err(BayesError::NotPositive {
                    name: "alpha",
                    value: a,
                });
            }
        }
        Ok(Self { alpha })
    }

    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// Total concentration `α₀ = Σ αᵢ`.
    pub fn concentration(&self) -> f64 {
        self.alpha.iter().sum()
    }

    /// Alias for [`concentration`](Self::concentration).
    pub fn alpha_sum(&self) -> f64 {
        self.concentration()
    }

    /// Component variances of a Dirichlet random probability vector.
    pub fn variance_vector(&self) -> Vec<f64> {
        let a0 = self.concentration();
        let denom = a0 * a0 * (a0 + 1.0);
        self.alpha.iter().map(|&a| a * (a0 - a) / denom).collect()
    }

    /// Covariance entry `Cov[P_i, P_j]`.
    ///
    /// Returns `NaN` for out-of-range indices.
    pub fn covariance_entry(&self, i: usize, j: usize) -> f64 {
        if i >= self.alpha.len() || j >= self.alpha.len() {
            return f64::NAN;
        }
        self.covariance().get(i, j)
    }

    /// `E[ln P_i] = ψ(α_i) - ψ(α₀)` for each component.
    pub fn expected_log_probabilities(&self) -> Vec<f64> {
        let a0 = self.concentration();
        self.alpha
            .iter()
            .map(|&a| digamma(a) - digamma(a0))
            .collect()
    }

    /// Mean count vector for a future Dirichlet-multinomial draw with `trials`.
    pub fn predictive_count_mean(&self, trials: u64) -> Vec<f64> {
        let n = trials as f64;
        self.mean_vector().into_iter().map(|p| n * p).collect()
    }

    /// Covariance matrix for a future Dirichlet-multinomial count vector with
    /// `trials`, after integrating out the category probabilities.
    pub fn predictive_count_covariance(&self, trials: u64) -> DenseMatrix {
        let n = trials as f64;
        let a0 = self.concentration();
        let p = self.mean_vector();
        let inflation = (n + a0) / (1.0 + a0);
        DenseMatrix::from_fn(self.alpha.len(), self.alpha.len(), |i, j| {
            if i == j {
                n * p[i] * (1.0 - p[i]) * inflation
            } else {
                -n * p[i] * p[j] * inflation
            }
        })
    }
}

impl Distribution<Vec<f64>> for Dirichlet {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let y: Vec<f64> = self
            .alpha
            .iter()
            .map(|&a| sample_standard_gamma(rng, a))
            .collect();
        let sum: f64 = y.iter().sum();
        y.iter().map(|&yi| yi / sum).collect()
    }
}

impl MultivariateContinuous for Dirichlet {
    fn dim(&self) -> usize {
        self.alpha.len()
    }
    fn ln_pdf(&self, x: &[f64]) -> f64 {
        if x.len() != self.alpha.len()
            || x.iter().any(|&xi| !xi.is_finite() || xi <= 0.0)
            || (x.iter().sum::<f64>() - 1.0).abs() > 1e-10
        {
            return f64::NEG_INFINITY;
        }
        let ln_norm =
            ln_gamma(self.alpha_sum()) - self.alpha.iter().map(|&a| ln_gamma(a)).sum::<f64>();
        ln_norm
            + self
                .alpha
                .iter()
                .zip(x)
                .map(|(&a, &xi)| (a - 1.0) * xi.ln())
                .sum::<f64>()
    }
}

impl MultivariateMoments for Dirichlet {
    fn mean_vector(&self) -> Vec<f64> {
        let a0 = self.concentration();
        self.alpha.iter().map(|&a| a / a0).collect()
    }
    fn covariance(&self) -> DenseMatrix {
        let a0 = self.concentration();
        let denom = a0 * a0 * (a0 + 1.0);
        let k = self.alpha.len();
        DenseMatrix::from_fn(k, k, |i, j| {
            if i == j {
                self.alpha[i] * (a0 - self.alpha[i]) / denom
            } else {
                -self.alpha[i] * self.alpha[j] / denom
            }
        })
    }
    /// `H = ln B(α) + (α₀ − K)ψ(α₀) − Σ(αᵢ − 1)ψ(αᵢ)`, with
    /// `ln B(α) = Σ ln Γ(αᵢ) − ln Γ(α₀)`.
    fn entropy(&self) -> Option<f64> {
        let a0 = self.concentration();
        let k = self.alpha.len() as f64;
        let ln_b = self.alpha.iter().map(|&a| ln_gamma(a)).sum::<f64>() - ln_gamma(a0);
        let sum_terms: f64 = self.alpha.iter().map(|&a| (a - 1.0) * digamma(a)).sum();
        Some(ln_b + (a0 - k) * digamma(a0) - sum_terms)
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
    fn mean_and_samples_lie_on_the_simplex() {
        let d = Dirichlet::new(vec![1.0, 2.0, 3.0]).unwrap();
        let m = d.mean_vector();
        close(m[0], 1.0 / 6.0, 1e-12);
        close(m.iter().sum::<f64>(), 1.0, 1e-12);

        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut avg = [0.0; 3];
        let n = 40_000;
        for _ in 0..n {
            let x = d.sample(&mut rng);
            close(x.iter().sum::<f64>(), 1.0, 1e-12);
            for i in 0..3 {
                avg[i] += x[i] / n as f64;
            }
        }
        for i in 0..3 {
            close(avg[i], m[i], 0.02);
        }
    }

    #[test]
    fn uniform_dirichlet_has_constant_density() {
        let d = Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap();
        close(d.pdf(&[0.2, 0.3, 0.5]), 2.0, 1e-10);
        close(d.pdf(&[0.6, 0.1, 0.3]), 2.0, 1e-10);
    }

    #[test]
    fn density_rejects_off_simplex_points() {
        let d = Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap();
        assert_eq!(d.pdf(&[0.2, 0.3, 0.6]), 0.0);
        assert_eq!(d.pdf(&[0.2, f64::NAN, 0.8]), 0.0);
    }

    #[test]
    fn closed_form_log_probability_and_predictive_count_moments() {
        let d = Dirichlet::new(vec![2.0, 3.0, 5.0]).unwrap();
        close(d.concentration(), 10.0, 1e-12);
        let elog = d.expected_log_probabilities();
        close(elog[0], digamma(2.0) - digamma(10.0), 1e-12);
        close(d.variance_vector()[1], 3.0 * 7.0 / (100.0 * 11.0), 1e-12);
        close(d.covariance_entry(0, 2), -2.0 * 5.0 / (100.0 * 11.0), 1e-12);

        let mean = d.predictive_count_mean(4);
        close(mean[0], 0.8, 1e-12);
        close(mean[2], 2.0, 1e-12);
        let cov = d.predictive_count_covariance(4);
        let inflation = 14.0 / 11.0;
        close(cov.get(0, 0), 4.0 * 0.2 * 0.8 * inflation, 1e-12);
        close(cov.get(0, 2), -4.0 * 0.2 * 0.5 * inflation, 1e-12);
    }
}
