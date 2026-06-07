use rand::Rng;
use rand::distr::Distribution;

use super::{MultivariateContinuous, MultivariateMoments};
use crate::distribution::continuous::gamma::sample_standard_gamma;
use crate::error::BayesError;
use crate::function::gamma::ln_gamma;
use crate::linalg::Matrix;

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

    fn alpha_sum(&self) -> f64 {
        self.alpha.iter().sum()
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
        let a0 = self.alpha_sum();
        self.alpha.iter().map(|&a| a / a0).collect()
    }
    fn covariance(&self) -> Matrix {
        let a0 = self.alpha_sum();
        let denom = a0 * a0 * (a0 + 1.0);
        let k = self.alpha.len();
        (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| {
                        if i == j {
                            self.alpha[i] * (a0 - self.alpha[i]) / denom
                        } else {
                            -self.alpha[i] * self.alpha[j] / denom
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
}
