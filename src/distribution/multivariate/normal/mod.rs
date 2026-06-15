use std::f64::consts::TAU;

use rand::Rng;
use rand::distr::Distribution;

use ramanujan::{Cholesky, DenseMatrix};

use super::{MultivariateContinuous, MultivariateMoments};
use crate::distribution::continuous::normal::standard_normal_polar;
use crate::error::BayesError;

const JITTER: f64 = 1e-12;

/// The multivariate normal `N(μ, Σ)` over `ℝ^d`, with mean vector `μ` and a
/// symmetric positive-definite covariance matrix `Σ`. Sampling and density use the
/// ramanujan Cholesky factorization `Σ = L Lᵀ`.
#[derive(Clone, Debug)]
pub struct MultivariateNormal {
    mean: Vec<f64>,
    cov: DenseMatrix,
    chol: Cholesky<f64>,
    /// ln |2πΣ|, cached for the density.
    ln_det_2pi: f64,
}

// `chol` is a deterministic function of `cov`, so the mean and covariance alone
// identify the distribution (ramanujan's `Cholesky` is not itself `PartialEq`).
impl PartialEq for MultivariateNormal {
    fn eq(&self, other: &Self) -> bool {
        self.mean == other.mean && self.cov == other.cov
    }
}

impl MultivariateNormal {
    /// Build from a mean vector and covariance matrix. Errors if the matrix is not
    /// square, mismatched with `mean`, non-finite, non-symmetric, or not
    /// positive-definite.
    pub fn new(mean: Vec<f64>, cov: DenseMatrix) -> Result<Self, BayesError> {
        let d = mean.len();
        if d == 0 || cov.rows() != d || cov.cols() != d {
            return Err(BayesError::Malformed {
                what: "covariance must be square and match the mean",
            });
        }
        if mean.iter().any(|x| !x.is_finite()) || !cov.is_finite() {
            return Err(BayesError::Malformed {
                what: "mean and covariance entries must be finite",
            });
        }
        for i in 0..d {
            for j in (i + 1)..d {
                if (cov.get(i, j) - cov.get(j, i)).abs() > 1e-10 {
                    return Err(BayesError::Malformed {
                        what: "covariance must be symmetric",
                    });
                }
            }
        }
        let chol = DenseMatrix::from_fn(d, d, |i, j| {
            cov.get(i, j) + if i == j { JITTER } else { 0.0 }
        })
        .cholesky()
        .map_err(|_| BayesError::Malformed {
            what: "covariance is not positive-definite",
        })?;
        let ln_det: f64 = (0..d).map(|i| chol.lower().get(i, i).ln()).sum::<f64>() * 2.0;
        let ln_det_2pi = d as f64 * TAU.ln() + ln_det;
        Ok(Self {
            mean,
            cov,
            chol,
            ln_det_2pi,
        })
    }

    pub fn mean(&self) -> &[f64] {
        &self.mean
    }
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }
}

impl Distribution<Vec<f64>> for MultivariateNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let d = self.mean.len();
        let z: Vec<f64> = (0..d).map(|_| standard_normal_polar(rng)).collect();
        let l = self.chol.lower();
        (0..d)
            .map(|i| self.mean[i] + (0..=i).map(|j| l.get(i, j) * z[j]).sum::<f64>())
            .collect()
    }
}

impl MultivariateContinuous for MultivariateNormal {
    fn dim(&self) -> usize {
        self.mean.len()
    }
    fn ln_pdf(&self, x: &[f64]) -> f64 {
        if x.len() != self.mean.len() || x.iter().any(|v| !v.is_finite()) {
            return f64::NEG_INFINITY;
        }
        let diff: Vec<f64> = x.iter().zip(&self.mean).map(|(xi, mi)| xi - mi).collect();
        let solved = self.chol.solve(&diff);
        let quad: f64 = diff.iter().zip(&solved).map(|(a, b)| a * b).sum();
        -0.5 * (quad + self.ln_det_2pi)
    }
}

impl MultivariateMoments for MultivariateNormal {
    fn mean_vector(&self) -> Vec<f64> {
        self.mean.clone()
    }
    fn covariance(&self) -> DenseMatrix {
        self.cov.clone()
    }
    /// `H = ½ ln((2πe)^d |Σ|) = ½(ln|2πΣ| + d)`, straight from the cached
    /// `ln_det_2pi`.
    fn entropy(&self) -> Option<f64> {
        Some(0.5 * (self.ln_det_2pi + self.mean.len() as f64))
    }
}

impl crate::distribution::KlDivergence for MultivariateNormal {
    /// `D_KL(N₀ ‖ N₁) = ½[ ln(|Σ₁|/|Σ₀|) − d + tr(Σ₁⁻¹Σ₀) + (μ₁−μ₀)ᵀΣ₁⁻¹(μ₁−μ₀) ]`,
    /// with the `Σ₁⁻¹` solves done through `other`'s cached Cholesky. `+∞` when
    /// the dimensions differ.
    fn kl_divergence(&self, other: &Self) -> f64 {
        let d = self.mean.len();
        if other.mean.len() != d {
            return f64::INFINITY;
        }
        // ln(|Σ₁|/|Σ₀|): the d·ln(2π) terms in `ln_det_2pi` cancel (same d).
        let ln_det_ratio = other.ln_det_2pi - self.ln_det_2pi;
        // tr(Σ₁⁻¹Σ₀): solve Σ₁ x = (column j of Σ₀), accumulate xⱼ.
        let trace: f64 = (0..d)
            .map(|j| {
                let col: Vec<f64> = (0..d).map(|i| self.cov.get(i, j)).collect();
                other.chol.solve(&col)[j]
            })
            .sum();
        // (μ₁−μ₀)ᵀ Σ₁⁻¹ (μ₁−μ₀).
        let diff: Vec<f64> = (0..d).map(|i| other.mean[i] - self.mean[i]).collect();
        let solved = other.chol.solve(&diff);
        let quad: f64 = diff.iter().zip(&solved).map(|(a, b)| a * b).sum();
        0.5 * (ln_det_ratio - d as f64 + trace + quad)
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
    fn diagonal_matches_independent_normals() {
        let mvn = MultivariateNormal::new(
            vec![1.0, -2.0],
            DenseMatrix::from_rows(&[vec![4.0, 0.0], vec![0.0, 9.0]]),
        )
        .unwrap();
        let x = [2.0, 0.0];
        let marg = |xi: f64, mu: f64, sd: f64| {
            let z = (xi - mu) / sd;
            -0.5 * z * z - (sd * TAU.sqrt()).ln()
        };
        close(
            mvn.ln_pdf(&x),
            marg(2.0, 1.0, 2.0) + marg(0.0, -2.0, 3.0),
            1e-10,
        );
    }

    #[test]
    fn sampled_mean_and_covariance_match() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mean = vec![1.0, -1.0];
        let cov = DenseMatrix::from_rows(&[vec![2.0, 0.8], vec![0.8, 1.0]]);
        let mvn = MultivariateNormal::new(mean.clone(), cov.clone()).unwrap();
        let n = 40_000;
        let samples: Vec<Vec<f64>> = (0..n).map(|_| mvn.sample(&mut rng)).collect();
        let m0 = samples.iter().map(|s| s[0]).sum::<f64>() / n as f64;
        let m1 = samples.iter().map(|s| s[1]).sum::<f64>() / n as f64;
        close(m0, mean[0], 0.05);
        close(m1, mean[1], 0.05);
        let c01 = samples
            .iter()
            .map(|s| (s[0] - m0) * (s[1] - m1))
            .sum::<f64>()
            / n as f64;
        close(c01, cov.get(0, 1), 0.05);
    }

    #[test]
    fn rejects_non_pd_covariance() {
        assert!(
            MultivariateNormal::new(
                vec![0.0, 0.0],
                DenseMatrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 1.0]])
            )
            .is_err()
        );
    }

    #[test]
    fn kl_and_entropy_factor_over_independent_marginals() {
        use crate::distribution::{KlDivergence, Moments, Normal};

        let diag = |a: f64, b: f64| DenseMatrix::from_rows(&[vec![a, 0.0], vec![0.0, b]]);
        let p = MultivariateNormal::new(vec![0.0, 0.0], diag(1.0, 4.0)).unwrap();
        let q = MultivariateNormal::new(vec![1.0, -1.0], diag(2.0, 3.0)).unwrap();

        // A product of independents: joint KL and joint entropy are the sums of
        // the per-axis Normal KLs / entropies.
        let (p0, p1) = (
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(0.0, 2.0).unwrap(),
        );
        let (q0, q1) = (
            Normal::new(1.0, 2.0_f64.sqrt()).unwrap(),
            Normal::new(-1.0, 3.0_f64.sqrt()).unwrap(),
        );
        close(
            p.kl_divergence(&q),
            p0.kl_divergence(&q0) + p1.kl_divergence(&q1),
            1e-10,
        );
        close(
            p.entropy().unwrap(),
            p0.entropy().unwrap() + p1.entropy().unwrap(),
            1e-10,
        );

        // Gibbs: self-KL is zero even for a correlated covariance.
        let r = MultivariateNormal::new(
            vec![2.0, -3.0],
            DenseMatrix::from_rows(&[vec![2.0, 0.8], vec![0.8, 1.0]]),
        )
        .unwrap();
        close(r.kl_divergence(&r), 0.0, 1e-10);
        assert!(p.kl_divergence(&q) > 0.0);
        // Dimension mismatch → +∞.
        let s = MultivariateNormal::new(vec![0.0], DenseMatrix::from_rows(&[vec![1.0]])).unwrap();
        assert!(p.kl_divergence(&s).is_infinite());
    }

    #[test]
    fn rejects_nonfinite_or_nonsymmetric_covariance() {
        assert!(
            MultivariateNormal::new(
                vec![0.0, f64::NAN],
                DenseMatrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]])
            )
            .is_err()
        );
        assert!(
            MultivariateNormal::new(
                vec![0.0, 0.0],
                DenseMatrix::from_rows(&[vec![1.0, 0.2], vec![0.3, 1.0]])
            )
            .is_err()
        );
    }
}
