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

    /// Fit a multivariate normal to data by maximum likelihood: the sample mean
    /// and the (unbiased) sample covariance. Needs at least two finite,
    /// equal-length observations of a positive dimension.
    ///
    /// The sample covariance is a scaled Gram matrix `XᵀX` of the centered data,
    /// so it is **positive semidefinite by construction** — proved in `riemann`
    /// (`RiemannLean/Bayes/Covariance.lean`: `gram_posSemidef`). Together with
    /// [`new`](Self::new)'s tiny diagonal jitter that makes it a valid
    /// positive-definite covariance, so the fit succeeds for any two-or-more
    /// finite, equal-length observations. For rank-deficient data (fewer
    /// independent samples than dimensions, or collinear columns) the result is
    /// near-singular; [`from_samples_ridge`](Self::from_samples_ridge) adds a
    /// chosen ridge for a better-conditioned covariance.
    pub fn from_samples(samples: &[Vec<f64>]) -> Result<Self, BayesError> {
        let cov = crate::statistics::covariance_matrix(samples);
        if cov.rows() == 0 {
            return Err(BayesError::Malformed {
                what: "need at least two finite, equal-length observations",
            });
        }
        Self::new(crate::statistics::mean_vector(samples), cov)
    }

    /// Like [`from_samples`](Self::from_samples) but adds a diagonal **ridge**
    /// `ridge·I` to the sample covariance (Tikhonov / shrinkage toward a
    /// spherical prior) — the control for ill-conditioned, near-singular sample
    /// covariances from rank-deficient data (few samples, duplicates, collinearity).
    ///
    /// `ridge·I` is a positive-(semi)definite prior covariance, and adding it to
    /// the PSD sample Gram stays positive semidefinite — proved in `riemann`
    /// (`RiemannLean/Bayes/Covariance.lean`: `posSemidef_add_gram`). For any
    /// `ridge > 0` the sum is strictly positive-definite (eigenvalues bounded
    /// below by `ridge`), so the resulting covariance is well-conditioned rather
    /// than resting on the construction jitter alone.
    pub fn from_samples_ridge(samples: &[Vec<f64>], ridge: f64) -> Result<Self, BayesError> {
        if !ridge.is_finite() || ridge < 0.0 {
            return Err(BayesError::NotPositive {
                name: "ridge",
                value: ridge,
            });
        }
        let cov = crate::statistics::covariance_matrix(samples);
        let d = cov.rows();
        if d == 0 {
            return Err(BayesError::Malformed {
                what: "need at least two finite, equal-length observations",
            });
        }
        let regularized = DenseMatrix::from_fn(d, d, |i, j| {
            cov.get(i, j) + if i == j { ridge } else { 0.0 }
        });
        Self::new(crate::statistics::mean_vector(samples), regularized)
    }

    pub fn mean(&self) -> &[f64] {
        &self.mean
    }
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// Borrow the covariance matrix.
    pub fn covariance_matrix(&self) -> &DenseMatrix {
        &self.cov
    }

    /// Closed-form marginal distribution over a coordinate subset.
    ///
    /// Coordinates are returned in the order supplied by `indices`. Duplicate or
    /// out-of-range indices are rejected because they would create a singular
    /// coordinate map rather than a true marginal.
    pub fn marginal(&self, indices: &[usize]) -> Result<Self, BayesError> {
        validate_indices(indices, self.dimension(), false)?;
        Self::new(
            select_vector(&self.mean, indices),
            select_matrix(&self.cov, indices, indices),
        )
    }

    /// Closed-form conditional distribution of the unobserved coordinates given
    /// observed coordinates.
    ///
    /// If `observed_indices` is empty this returns `self`. If every coordinate is
    /// observed, there is no non-empty Gaussian left to return and an error is
    /// reported.
    pub fn condition(
        &self,
        observed_indices: &[usize],
        observed_values: &[f64],
    ) -> Result<Self, BayesError> {
        let d = self.dimension();
        if observed_indices.is_empty() {
            if observed_values.is_empty() {
                return Ok(self.clone());
            }
            return Err(BayesError::Malformed {
                what: "observed values must match observed indices",
            });
        }
        validate_indices(observed_indices, d, false)?;
        if observed_values.len() != observed_indices.len()
            || observed_values.iter().any(|x| !x.is_finite())
        {
            return Err(BayesError::Malformed {
                what: "observed values must be finite and match observed indices",
            });
        }
        let unobserved = complement_indices(d, observed_indices);
        if unobserved.is_empty() {
            return Err(BayesError::Malformed {
                what: "conditioning on all coordinates leaves no distribution",
            });
        }

        let sigma_oo = select_matrix(&self.cov, observed_indices, observed_indices);
        let chol_oo = jittered_cholesky(&sigma_oo)?;
        let delta: Vec<f64> = observed_values
            .iter()
            .zip(select_vector(&self.mean, observed_indices))
            .map(|(&x, mu)| x - mu)
            .collect();
        let solved_delta = chol_oo.solve(&delta);

        let sigma_uo = select_matrix(&self.cov, &unobserved, observed_indices);
        let sigma_ou = select_matrix(&self.cov, observed_indices, &unobserved);
        let sigma_uu = select_matrix(&self.cov, &unobserved, &unobserved);

        let mean_u = select_vector(&self.mean, &unobserved);
        let cond_mean: Vec<f64> = mean_u
            .iter()
            .enumerate()
            .map(|(i, &mu)| {
                mu + (0..observed_indices.len())
                    .map(|j| sigma_uo.get(i, j) * solved_delta[j])
                    .sum::<f64>()
            })
            .collect();

        let cond_cov = DenseMatrix::from_fn(unobserved.len(), unobserved.len(), |i, j| {
            let col = sigma_ou.column(j);
            let solved_col = chol_oo.solve(&col);
            sigma_uu.get(i, j)
                - (0..observed_indices.len())
                    .map(|k| sigma_uo.get(i, k) * solved_col[k])
                    .sum::<f64>()
        });
        Self::new(cond_mean, symmetrize(&cond_cov))
    }

    /// Closed-form affine image `Y = A X + b`.
    ///
    /// `A` must have `dimension()` columns and `b` one entry per output row.
    pub fn affine_transform(&self, a: &DenseMatrix, b: &[f64]) -> Result<Self, BayesError> {
        let d = self.dimension();
        if a.rows() == 0 || a.cols() != d || b.len() != a.rows() {
            return Err(BayesError::Malformed {
                what: "affine transform shape mismatch",
            });
        }
        if !a.is_finite() || b.iter().any(|x| !x.is_finite()) {
            return Err(BayesError::Malformed {
                what: "affine transform entries must be finite",
            });
        }
        let m = a.rows();
        let mean = (0..m)
            .map(|i| b[i] + (0..d).map(|j| a.get(i, j) * self.mean[j]).sum::<f64>())
            .collect();
        let cov = DenseMatrix::from_fn(m, m, |r, c| {
            let mut acc = 0.0;
            for i in 0..d {
                for j in 0..d {
                    acc += a.get(r, i) * self.cov.get(i, j) * a.get(c, j);
                }
            }
            acc
        });
        Self::new(mean, symmetrize(&cov))
    }

    /// Batch log likelihood `Σ ln f(xᵢ)`. Empty samples have log likelihood `0`.
    pub fn ln_likelihood(&self, samples: &[Vec<f64>]) -> f64 {
        if samples
            .iter()
            .any(|x| x.len() != self.dimension() || x.iter().any(|v| !v.is_finite()))
        {
            return f64::NEG_INFINITY;
        }
        samples.iter().map(|x| self.ln_pdf(x)).sum()
    }

    /// The integral of the pointwise product of two Gaussian densities:
    ///
    /// `∫ N(x|μ₀,Σ₀) N(x|μ₁,Σ₁) dx = N(μ₀ | μ₁, Σ₀ + Σ₁)`.
    pub fn ln_product_integral(&self, other: &Self) -> f64 {
        let d = self.dimension();
        if other.dimension() != d {
            return f64::NEG_INFINITY;
        }
        let cov_sum = DenseMatrix::from_fn(d, d, |i, j| self.cov.get(i, j) + other.cov.get(i, j));
        let Ok(chol) = jittered_cholesky(&cov_sum) else {
            return f64::NAN;
        };
        let diff: Vec<f64> = (0..d).map(|i| self.mean[i] - other.mean[i]).collect();
        let solved = chol.solve(&diff);
        let quad = diff.iter().zip(solved).map(|(&a, b)| a * b).sum::<f64>();
        let ln_det = 2.0 * (0..d).map(|i| chol.lower().get(i, i).ln()).sum::<f64>();
        -0.5 * (d as f64 * TAU.ln() + ln_det + quad)
    }

    /// Closed-form precision fusion of two Gaussian factors.
    ///
    /// The result is proportional to the product of the two densities; the log
    /// normalizing integral is available from [`ln_product_integral`](Self::ln_product_integral).
    pub fn precision_fusion(&self, other: &Self) -> Result<Self, BayesError> {
        let d = self.dimension();
        if other.dimension() != d {
            return Err(BayesError::Malformed {
                what: "Gaussian dimensions must match",
            });
        }
        let p0 = inverse_spd(&self.cov)?;
        let p1 = inverse_spd(&other.cov)?;
        let precision = DenseMatrix::from_fn(d, d, |i, j| p0.get(i, j) + p1.get(i, j));
        let cov = inverse_spd(&precision)?;
        let h: Vec<f64> = (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| p0.get(i, j) * self.mean[j] + p1.get(i, j) * other.mean[j])
                    .sum()
            })
            .collect();
        let mean: Vec<f64> = (0..d)
            .map(|i| (0..d).map(|j| cov.get(i, j) * h[j]).sum())
            .collect();
        Self::new(mean, symmetrize(&cov))
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

fn validate_indices(
    indices: &[usize],
    dimension: usize,
    allow_empty: bool,
) -> Result<(), BayesError> {
    if indices.is_empty() && !allow_empty {
        return Err(BayesError::Malformed {
            what: "coordinate index set must be non-empty",
        });
    }
    let mut seen = vec![false; dimension];
    for &idx in indices {
        if idx >= dimension || seen[idx] {
            return Err(BayesError::Malformed {
                what: "coordinate indices must be in range and unique",
            });
        }
        seen[idx] = true;
    }
    Ok(())
}

fn complement_indices(dimension: usize, indices: &[usize]) -> Vec<usize> {
    let mut observed = vec![false; dimension];
    for &idx in indices {
        observed[idx] = true;
    }
    (0..dimension).filter(|&idx| !observed[idx]).collect()
}

fn select_vector(values: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&idx| values[idx]).collect()
}

fn select_matrix(matrix: &DenseMatrix, rows: &[usize], cols: &[usize]) -> DenseMatrix {
    DenseMatrix::from_fn(rows.len(), cols.len(), |i, j| matrix.get(rows[i], cols[j]))
}

fn symmetrize(matrix: &DenseMatrix) -> DenseMatrix {
    DenseMatrix::from_fn(matrix.rows(), matrix.cols(), |i, j| {
        0.5 * (matrix.get(i, j) + matrix.get(j, i))
    })
}

fn jittered_cholesky(matrix: &DenseMatrix) -> Result<Cholesky<f64>, BayesError> {
    DenseMatrix::from_fn(matrix.rows(), matrix.cols(), |i, j| {
        matrix.get(i, j) + if i == j { JITTER } else { 0.0 }
    })
    .cholesky()
    .map_err(|_| BayesError::Malformed {
        what: "matrix is not positive-definite",
    })
}

fn inverse_spd(matrix: &DenseMatrix) -> Result<DenseMatrix, BayesError> {
    let n = matrix.rows();
    let chol = jittered_cholesky(matrix)?;
    Ok(DenseMatrix::from_fn(n, n, |i, j| {
        let mut unit = vec![0.0; n];
        unit[j] = 1.0;
        chol.solve(&unit)[i]
    }))
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
    fn marginal_condition_and_affine_transform_match_closed_forms() {
        let mvn = MultivariateNormal::new(
            vec![1.0, 2.0, 3.0],
            DenseMatrix::from_rows(&[
                vec![4.0, 1.0, 0.5],
                vec![1.0, 9.0, 2.0],
                vec![0.5, 2.0, 16.0],
            ]),
        )
        .unwrap();

        let marginal = mvn.marginal(&[2, 0]).unwrap();
        assert_eq!(marginal.mean(), &[3.0, 1.0]);
        close(marginal.covariance_matrix().get(0, 0), 16.0, 1e-12);
        close(marginal.covariance_matrix().get(0, 1), 0.5, 1e-12);
        close(marginal.covariance_matrix().get(1, 1), 4.0, 1e-12);
        assert!(mvn.marginal(&[0, 0]).is_err());

        let bivar = MultivariateNormal::new(
            vec![0.0, 0.0],
            DenseMatrix::from_rows(&[vec![4.0, 2.0], vec![2.0, 9.0]]),
        )
        .unwrap();
        let conditional = bivar.condition(&[0], &[2.0]).unwrap();
        assert_eq!(conditional.mean().len(), 1);
        close(conditional.mean()[0], 1.0, 1e-10);
        close(conditional.covariance_matrix().get(0, 0), 8.0, 1e-10);
        assert!(bivar.condition(&[0, 1], &[0.0, 0.0]).is_err());

        let affine = bivar
            .affine_transform(
                &DenseMatrix::from_rows(&[vec![1.0, 1.0], vec![2.0, 0.0]]),
                &[1.0, -1.0],
            )
            .unwrap();
        assert_eq!(affine.mean(), &[1.0, -1.0]);
        close(affine.covariance_matrix().get(0, 0), 17.0, 1e-12);
        close(affine.covariance_matrix().get(0, 1), 12.0, 1e-12);
        close(affine.covariance_matrix().get(1, 1), 16.0, 1e-12);
    }

    #[test]
    fn batch_likelihood_product_integral_and_precision_fusion_are_closed_form() {
        let p = MultivariateNormal::new(vec![0.0], DenseMatrix::from_rows(&[vec![4.0]])).unwrap();
        let q = MultivariateNormal::new(vec![2.0], DenseMatrix::from_rows(&[vec![6.0]])).unwrap();

        let data = vec![vec![0.0], vec![1.0]];
        close(
            p.ln_likelihood(&data),
            p.ln_pdf(&[0.0]) + p.ln_pdf(&[1.0]),
            1e-12,
        );
        assert_eq!(p.ln_likelihood(&[vec![0.0, 1.0]]), f64::NEG_INFINITY);

        let expected_integral = -0.5 * ((TAU * 10.0).ln() + 4.0 / 10.0);
        close(p.ln_product_integral(&q), expected_integral, 1e-10);

        let fused = p.precision_fusion(&q).unwrap();
        close(fused.mean()[0], 0.8, 1e-9);
        close(fused.covariance_matrix().get(0, 0), 12.0 / 5.0, 1e-9);
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

    #[test]
    fn from_samples_recovers_the_generating_gaussian() {
        // Sample a known correlated Gaussian, then refit by ML; the sample mean
        // and (Gram → PSD, gram_posSemidef) sample covariance recover it.
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mean = vec![1.0, -1.0];
        let cov = DenseMatrix::from_rows(&[vec![2.0, 0.8], vec![0.8, 1.0]]);
        let truth = MultivariateNormal::new(mean.clone(), cov.clone()).unwrap();
        let data: Vec<Vec<f64>> = (0..40_000).map(|_| truth.sample(&mut rng)).collect();

        let fit = MultivariateNormal::from_samples(&data).unwrap();
        close(fit.mean()[0], mean[0], 0.05);
        close(fit.mean()[1], mean[1], 0.05);
        let c = fit.covariance();
        close(c.get(0, 0), cov.get(0, 0), 0.05);
        close(c.get(0, 1), cov.get(0, 1), 0.05);
        close(c.get(1, 1), cov.get(1, 1), 0.05);

        // Too few / ragged observations are rejected.
        assert!(MultivariateNormal::from_samples(&[vec![1.0, 2.0]]).is_err());
        assert!(MultivariateNormal::from_samples(&[]).is_err());
    }

    #[test]
    fn ridge_regularizes_the_sample_covariance() {
        // Three identical points: the sample covariance is the zero matrix
        // (rank 0). gram_posSemidef + the construction jitter still yield a valid
        // (if near-singular) covariance, so the plain fit succeeds…
        let degenerate = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        assert!(MultivariateNormal::from_samples(&degenerate).is_ok());

        // …and a chosen ridge (posSemidef_add_gram) makes it the well-conditioned
        // sample covariance + ridge·I — here 0 + 0.5·I.
        let fit = MultivariateNormal::from_samples_ridge(&degenerate, 0.5).unwrap();
        let c = fit.covariance();
        close(c.get(0, 0), 0.5, 1e-12);
        close(c.get(1, 1), 0.5, 1e-12);
        close(c.get(0, 1), 0.0, 1e-12);
        close(fit.mean()[0], 1.0, 1e-12);

        // On full-rank data the ridge just shifts the diagonal by `ridge`.
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let truth = MultivariateNormal::new(
            vec![0.0, 0.0],
            DenseMatrix::from_rows(&[vec![2.0, 0.5], vec![0.5, 1.0]]),
        )
        .unwrap();
        let data: Vec<Vec<f64>> = (0..5_000).map(|_| truth.sample(&mut rng)).collect();
        let plain = MultivariateNormal::from_samples(&data).unwrap();
        let ridged = MultivariateNormal::from_samples_ridge(&data, 0.25).unwrap();
        close(
            ridged.covariance().get(0, 0),
            plain.covariance().get(0, 0) + 0.25,
            1e-9,
        );
        close(
            ridged.covariance().get(0, 1),
            plain.covariance().get(0, 1),
            1e-9,
        );

        // A negative ridge is rejected.
        assert!(MultivariateNormal::from_samples_ridge(&degenerate, -1.0).is_err());
    }
}
