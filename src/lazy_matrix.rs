//! Lazy multivariate kernels over the `euclid::array` matrix contract.
//!
//! Where `lazy` writes scalar/vector closed forms over [`LazyReal`], this module
//! writes the **matrix** closed forms — the multivariate Gaussian families — over
//! [`LazyMatrix`]/[`LazyVector`]. The kernels never form an inverse or touch a scalar
//! element: they ask the matrix contract for the *mathematical quantities* it owns
//! (`mahalanobis_squared`, `logdet_spd`), so the same source runs on the `f64`
//! reference (`ramanujan`) and on a tensor backend (`lebesgue`/XLA).

use euclid::array::{LazyMatrix, LazyVector};
use euclid::numeric::LazyReal;

/// Multivariate-normal log-density `ln N(x | µ, Σ) = −½·(δᵀΣ⁻¹δ + d·ln(2π) + ln|Σ|)`,
/// `δ = x − µ`. The Mahalanobis term and log-determinant come from the matrix
/// contract (a stable Cholesky on either backend); no inverse is formed.
pub fn multivariate_normal_log_pdf<M: LazyMatrix<Scalar: LazyReal>>(
    x: M::Vector,
    mean: M::Vector,
    covariance: M,
) -> M::Scalar {
    let delta = x.sub(mean);
    let dim = M::Scalar::scalar(delta.len() as f64);
    let mahalanobis = covariance.mahalanobis_squared(&delta);
    let log_det = covariance.logdet_spd();
    -M::Scalar::half() * (mahalanobis + dim * M::Scalar::ln_tau() + log_det)
}

/// Multivariate-normal entropy `½·(d·ln(2π) + ln|Σ| + d)`.
pub fn multivariate_normal_entropy<M: LazyMatrix<Scalar: LazyReal>>(covariance: M) -> M::Scalar {
    let dim = M::Scalar::scalar(covariance.shape().0 as f64);
    let log_det = covariance.logdet_spd();
    M::Scalar::half() * (dim.clone() * M::Scalar::ln_tau() + log_det + dim)
}

/// Gaussian KL `D(N₀ ‖ N₁) = ½[ ln(|Σ₁|/|Σ₀|) − d + tr(Σ₁⁻¹Σ₀) + (µ₁−µ₀)ᵀΣ₁⁻¹(µ₁−µ₀) ]`.
///
/// `tr(Σ₁⁻¹Σ₀)` is `(Σ₁.solve_spd(Σ₀)).trace()`; the quadratic is
/// `Σ₁.mahalanobis_squared(µ₁−µ₀)`. No inverse is formed.
pub fn multivariate_normal_kl<M: LazyMatrix<Scalar: LazyReal>>(
    mean_self: M::Vector,
    cov_self: M,
    mean_other: M::Vector,
    cov_other: M,
) -> M::Scalar {
    let delta = mean_other.sub(mean_self);
    let dim = M::Scalar::scalar(delta.len() as f64);
    let trace = cov_other.clone().solve_spd(cov_self.clone()).trace();
    let mahalanobis = cov_other.mahalanobis_squared(&delta);
    let log_det_ratio = cov_other.logdet_spd() - cov_self.logdet_spd();
    M::Scalar::half() * (log_det_ratio - dim + trace + mahalanobis)
}

/// Wishart log-density at SPD `X`, scale `V`, degrees `ν`:
/// `½(ν−d−1)·ln|X| − ½·tr(V⁻¹X) − [½νd·ln2 + ½ν·ln|V| + ln Γ_d(ν/2)]`.
pub fn wishart_log_pdf<M: LazyMatrix<Scalar: LazyReal>>(
    x: M,
    scale: M,
    freedom: M::Scalar,
) -> M::Scalar {
    let d = x.shape().0;
    let dim = M::Scalar::scalar(d as f64);
    let half = M::Scalar::half();
    let half_freedom = freedom.clone() * half.clone();

    let log_det_x = x.clone().logdet_spd();
    let trace = scale.clone().solve_spd(x).trace();
    let log_det_scale = scale.logdet_spd();
    let ln_mv_gamma = crate::lazy::ln_multivariate_gamma(d, half_freedom.clone());

    let ln_normalizer = half_freedom.clone() * dim.clone() * M::Scalar::ln_2()
        + half_freedom * log_det_scale
        + ln_mv_gamma;
    half.clone() * (freedom - dim - M::Scalar::one()) * log_det_x - half * trace - ln_normalizer
}

/// Inverse-Wishart log-density at SPD `Σ`, scale `V`, degrees `ν`:
/// `[½ν·ln|V| − ½νd·ln2 − ln Γ_d(ν/2)] − ½(ν+d+1)·ln|Σ| − ½·tr(V·Σ⁻¹)`.
pub fn inverse_wishart_log_pdf<M: LazyMatrix<Scalar: LazyReal>>(
    sigma: M,
    scale: M,
    freedom: M::Scalar,
) -> M::Scalar {
    let d = sigma.shape().0;
    let dim = M::Scalar::scalar(d as f64);
    let half = M::Scalar::half();
    let half_freedom = freedom.clone() * half.clone();

    let log_det_sigma = sigma.clone().logdet_spd();
    let trace = sigma.solve_spd(scale.clone()).trace(); // tr(Σ⁻¹V) = tr(VΣ⁻¹)
    let log_det_scale = scale.logdet_spd();
    let ln_mv_gamma = crate::lazy::ln_multivariate_gamma(d, half_freedom.clone());

    let ln_normalizer = half_freedom.clone() * log_det_scale
        - half_freedom * dim.clone() * M::Scalar::ln_2()
        - ln_mv_gamma;
    ln_normalizer - half.clone() * (freedom + dim + M::Scalar::one()) * log_det_sigma - half * trace
}

/// `Σ_{i=1}^{d} ψ((ν + 1 − i)/2)` — the Wishart multivariate digamma sum. A host-side
/// fold over the structural dimension `d` building lazy `digamma` terms, the same
/// shape as [`crate::lazy::ln_multivariate_gamma`].
fn digamma_wishart_sum<S: LazyReal>(dimension: usize, freedom: S) -> S {
    let mut acc = S::zero();
    for i in 1..=dimension {
        acc = acc + ((freedom.clone() + S::scalar(1.0 - i as f64)) * S::half()).digamma();
    }
    acc
}

/// Wishart differential entropy (scale `V`, degrees `ν`):
/// `−½(ν−d−1)·E[ln|X|] + ½νd + lnZ`, with `E[ln|X|] = ln|V| + d·ln2 + Σ ψ((ν+1−i)/2)`
/// and `lnZ = ½νd·ln2 + ½ν·ln|V| + ln Γ_d(ν/2)`. The log-determinant comes from the
/// matrix contract; the digamma sum and `ln Γ_d` are pure scalar analysis.
pub fn wishart_entropy<M: LazyMatrix<Scalar: LazyReal>>(scale: M, freedom: M::Scalar) -> M::Scalar {
    let d = scale.shape().0;
    let dim = M::Scalar::scalar(d as f64);
    let half = M::Scalar::half();
    let log_det_scale = scale.logdet_spd();
    let digamma_sum = digamma_wishart_sum(d, freedom.clone());

    let expected_log_det = log_det_scale.clone() + dim.clone() * M::Scalar::ln_2() + digamma_sum;
    let ln_normalizer = half.clone() * freedom.clone() * dim.clone() * M::Scalar::ln_2()
        + half.clone() * freedom.clone() * log_det_scale
        + crate::lazy::ln_multivariate_gamma(d, freedom.clone() * half.clone());

    -half.clone() * (freedom.clone() - dim.clone() - M::Scalar::one()) * expected_log_det
        + half * freedom * dim
        + ln_normalizer
}

/// Inverse-Wishart differential entropy (scale `V`, degrees `ν`):
/// `−lnZ + ½(ν+d+1)·E[ln|Σ|] + ½νd`, with `E[ln|Σ|] = ln|V| − d·ln2 − Σ ψ((ν+1−i)/2)`
/// and `lnZ = ½ν·ln|V| − ½νd·ln2 − ln Γ_d(ν/2)`.
pub fn inverse_wishart_entropy<M: LazyMatrix<Scalar: LazyReal>>(
    scale: M,
    freedom: M::Scalar,
) -> M::Scalar {
    let d = scale.shape().0;
    let dim = M::Scalar::scalar(d as f64);
    let half = M::Scalar::half();
    let log_det_scale = scale.logdet_spd();
    let digamma_sum = digamma_wishart_sum(d, freedom.clone());

    let expected_log_det = log_det_scale.clone() - dim.clone() * M::Scalar::ln_2() - digamma_sum;
    let ln_normalizer = half.clone() * freedom.clone() * log_det_scale
        - half.clone() * freedom.clone() * dim.clone() * M::Scalar::ln_2()
        - crate::lazy::ln_multivariate_gamma(d, freedom.clone() * half.clone());

    -ln_normalizer
        + half.clone() * (freedom.clone() + dim.clone() + M::Scalar::one()) * expected_log_det
        + half * freedom * dim
}

/// Multivariate Student-t entropy (scale `Σ`, degrees `ν`):
/// `lnΓ(ν/2) − lnΓ((ν+d)/2) + ½(d·ln(νπ) + ln|Σ|) + (ν+d)/2·(ψ((ν+d)/2) − ψ(ν/2))`.
pub fn multivariate_student_t_entropy<M: LazyMatrix<Scalar: LazyReal>>(
    scale: M,
    freedom: M::Scalar,
) -> M::Scalar {
    let dim = M::Scalar::scalar(scale.shape().0 as f64);
    let log_det = scale.logdet_spd();
    let half = M::Scalar::half();
    let half_nu = freedom.clone() * half.clone();
    let half_nu_plus_d = (freedom.clone() + dim.clone()) * half.clone();
    half_nu.clone().ln_gamma() - half_nu_plus_d.clone().ln_gamma()
        + half * (dim * (freedom * M::Scalar::pi()).ln() + log_det)
        + half_nu_plus_d.clone() * (half_nu_plus_d.digamma() - half_nu.digamma())
}

/// Gaussian-process log marginal likelihood `−½(yᵀK⁻¹y + ln|K| + n·ln 2π)` — the
/// log-evidence of observations `y` under a zero-mean GP with kernel matrix `K`.
/// (Structurally the multivariate-normal log-density at zero mean.)
pub fn gaussian_process_log_marginal_likelihood<M: LazyMatrix<Scalar: LazyReal>>(
    y: M::Vector,
    kernel: M,
) -> M::Scalar {
    let n = M::Scalar::scalar(y.len() as f64);
    let mahalanobis = kernel.mahalanobis_squared(&y);
    let log_det = kernel.logdet_spd();
    -M::Scalar::half() * (mahalanobis + n * M::Scalar::ln_tau() + log_det)
}

/// Multivariate Student-t log-density (location `µ`, scale `Σ`, degrees `ν`):
/// `lnΓ((ν+d)/2) − lnΓ(ν/2) − ½(d·ln(νπ) + ln|Σ|) − ½(ν+d)·ln(1 + δᵀΣ⁻¹δ / ν)`.
pub fn multivariate_student_t_log_pdf<M: LazyMatrix<Scalar: LazyReal>>(
    x: M::Vector,
    location: M::Vector,
    scale: M,
    freedom: M::Scalar,
) -> M::Scalar {
    let delta = x.sub(location);
    let dim = M::Scalar::scalar(delta.len() as f64);
    let mahalanobis = scale.mahalanobis_squared(&delta);
    let log_det = scale.logdet_spd();

    let half = M::Scalar::half();
    let nu_plus_d = freedom.clone() + dim.clone();
    (nu_plus_d.clone() * half.clone()).ln_gamma()
        - (freedom.clone() * half.clone()).ln_gamma()
        - half.clone() * (dim * (freedom.clone() * M::Scalar::pi()).ln() + log_det)
        - half * nu_plus_d * (M::Scalar::one() + mahalanobis / freedom).ln()
}

/// Hotelling's `T² = n·(x̄ − µ₀)ᵀ S⁻¹ (x̄ − µ₀)` — the sample mean's squared
/// Mahalanobis distance from `µ₀` under the sample covariance `S`, scaled by `n`.
/// No inverse is formed; the quadratic comes from the matrix contract.
pub fn hotelling_t_squared<M: LazyMatrix<Scalar: LazyReal>>(
    count: M::Scalar,
    mean: M::Vector,
    reference: M::Vector,
    covariance: M,
) -> M::Scalar {
    count * covariance.mahalanobis_squared(&mean.sub(reference))
}

/// Pooled (within-group) covariance `Σᵢ dfᵢ·Sᵢ / Σᵢ dfᵢ`, the degrees-of-freedom
/// weighted average of the per-group covariances `Sᵢ` (typically `dfᵢ = nᵢ − 1`).
/// `None` for fewer than two groups or a length mismatch.
pub fn pooled_covariance<M: LazyMatrix<Scalar: LazyReal>>(
    covariances: &[M],
    degrees: &[M::Scalar],
) -> Option<M> {
    if covariances.len() != degrees.len() || covariances.len() < 2 {
        return None;
    }
    let mut total = degrees[0].clone();
    let mut pooled = covariances[0].clone().scale(degrees[0].clone());
    for i in 1..covariances.len() {
        pooled = pooled.add(covariances[i].clone().scale(degrees[i].clone()));
        total = total + degrees[i].clone();
    }
    Some(pooled.scale(M::Scalar::one() / total))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::{
        InverseWishart, KlDivergence, MultivariateContinuous, MultivariateMoments,
        MultivariateNormal, MultivariateStudentT, Wishart,
    };
    use crate::statistics::mahalanobis_squared;
    use ramanujan::{DenseMatrix, DenseVector};

    fn close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol + tol * b.abs(),
            "observed {a}, expected {b}"
        );
    }

    // --- a minimal `f64` reference implementation of the array contract ---
    //
    // Stands in for `ramanujan::DenseMatrix` (which owns the real impl) so the generic
    // kernels can be exercised end-to-end. Row-major, SPD; naive Cholesky.

    #[derive(Clone)]
    struct RefVector(Vec<f64>);

    impl LazyVector for RefVector {
        type Scalar = f64;
        fn len(&self) -> usize {
            self.0.len()
        }
        fn add(self, other: Self) -> Self {
            RefVector(self.0.iter().zip(&other.0).map(|(a, b)| a + b).collect())
        }
        fn sub(self, other: Self) -> Self {
            RefVector(self.0.iter().zip(&other.0).map(|(a, b)| a - b).collect())
        }
        fn scale(self, factor: f64) -> Self {
            RefVector(self.0.iter().map(|a| a * factor).collect())
        }
        fn hadamard(self, other: Self) -> Self {
            RefVector(self.0.iter().zip(&other.0).map(|(a, b)| a * b).collect())
        }
        fn dot(&self, other: &Self) -> f64 {
            self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum()
        }
        fn sum(&self) -> f64 {
            self.0.iter().sum()
        }
        fn norm_squared(&self) -> f64 {
            self.dot(self)
        }
        fn norm(&self) -> f64 {
            self.norm_squared().sqrt()
        }
    }

    #[derive(Clone)]
    struct RefMatrix {
        n: usize,
        data: Vec<f64>,
    }

    impl RefMatrix {
        fn at(&self, i: usize, j: usize) -> f64 {
            self.data[i * self.n + j]
        }
        /// Lower Cholesky factor `L` (row-major) with `Σ = L Lᵀ`.
        fn chol(&self) -> Vec<f64> {
            let n = self.n;
            let mut l = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..=i {
                    let mut s = self.at(i, j);
                    for k in 0..j {
                        s -= l[i * n + k] * l[j * n + k];
                    }
                    l[i * n + j] = if i == j { s.sqrt() } else { s / l[j * n + j] };
                }
            }
            l
        }
    }

    impl LazyMatrix for RefMatrix {
        type Scalar = f64;
        type Vector = RefVector;

        fn shape(&self) -> (usize, usize) {
            (self.n, self.n)
        }
        fn add(self, other: Self) -> Self {
            RefMatrix {
                n: self.n,
                data: self
                    .data
                    .iter()
                    .zip(&other.data)
                    .map(|(a, b)| a + b)
                    .collect(),
            }
        }
        fn scale(self, factor: f64) -> Self {
            RefMatrix {
                n: self.n,
                data: self.data.iter().map(|a| a * factor).collect(),
            }
        }
        fn transpose(self) -> Self {
            let n = self.n;
            RefMatrix {
                n,
                data: (0..n * n).map(|idx| self.at(idx % n, idx / n)).collect(),
            }
        }
        fn matmul(self, rhs: Self) -> Self {
            let n = self.n;
            let data = (0..n * n)
                .map(|idx| {
                    let (i, j) = (idx / n, idx % n);
                    (0..n).map(|k| self.at(i, k) * rhs.at(k, j)).sum()
                })
                .collect();
            RefMatrix { n, data }
        }
        fn matvec(self, v: RefVector) -> RefVector {
            let n = self.n;
            RefVector(
                (0..n)
                    .map(|i| (0..n).map(|k| self.at(i, k) * v.0[k]).sum())
                    .collect(),
            )
        }
        fn trace(&self) -> f64 {
            (0..self.n).map(|i| self.at(i, i)).sum()
        }
        fn diagonal(&self) -> RefVector {
            RefVector((0..self.n).map(|i| self.at(i, i)).collect())
        }
        fn cholesky(self) -> Self {
            RefMatrix {
                n: self.n,
                data: self.chol(),
            }
        }
        fn solve_spd(self, rhs: Self) -> Self {
            // Σ⁻¹ B, column by column via the Cholesky + forward/back substitution.
            let n = self.n;
            let l = self.chol();
            let mut out = vec![0.0; n * n];
            for col in 0..n {
                // forward solve L y = b
                let mut y = vec![0.0; n];
                for i in 0..n {
                    let mut s = rhs.at(i, col);
                    for k in 0..i {
                        s -= l[i * n + k] * y[k];
                    }
                    y[i] = s / l[i * n + i];
                }
                // back solve Lᵀ x = y
                let mut x = vec![0.0; n];
                for i in (0..n).rev() {
                    let mut s = y[i];
                    for k in (i + 1)..n {
                        s -= l[k * n + i] * x[k];
                    }
                    x[i] = s / l[i * n + i];
                }
                for (i, xi) in x.into_iter().enumerate() {
                    out[i * n + col] = xi;
                }
            }
            RefMatrix { n, data: out }
        }
        fn logdet_spd(self) -> f64 {
            let l = self.chol();
            2.0 * (0..self.n).map(|i| l[i * self.n + i].ln()).sum::<f64>()
        }
        fn mahalanobis_squared(&self, delta: &RefVector) -> f64 {
            // ‖L⁻¹δ‖² = δᵀ(LLᵀ)⁻¹δ via forward substitution.
            let n = self.n;
            let l = self.chol();
            let mut z = vec![0.0; n];
            for i in 0..n {
                let mut s = delta.0[i];
                for k in 0..i {
                    s -= l[i * n + k] * z[k];
                }
                z[i] = s / l[i * n + i];
            }
            z.iter().map(|v| v * v).sum()
        }
    }

    fn ref_matrix(n: usize, rows: &[&[f64]]) -> RefMatrix {
        RefMatrix {
            n,
            data: rows.iter().flat_map(|r| r.iter().copied()).collect(),
        }
    }

    fn dense(n: usize, rows: &[&[f64]]) -> DenseMatrix {
        DenseMatrix::from_fn(n, n, |i, j| rows[i][j])
    }

    #[test]
    fn lazy_matrix_mvn_log_pdf_matches_distribution() {
        // A 3×3 SPD covariance.
        let rows: [&[f64]; 3] = [&[2.0, 0.3, 0.1], &[0.3, 1.5, -0.2], &[0.1, -0.2, 1.0]];
        let mean = [0.5, -1.0, 0.2];
        let x = [1.2, -0.3, 0.7];

        let mvn = MultivariateNormal::new(mean.to_vec(), dense(3, &rows)).unwrap();
        let got = multivariate_normal_log_pdf(
            RefVector(x.to_vec()),
            RefVector(mean.to_vec()),
            ref_matrix(3, &rows),
        );
        close(got, mvn.ln_pdf(&x), 1e-10);
    }

    #[test]
    fn lazy_matrix_mvn_log_pdf_runs_on_ramanujan_densematrix() {
        // The point of the contract: `ramanujan::DenseMatrix` satisfies `LazyMatrix`, so
        // the *same* generic kernel runs on the jacobi-backed f64 reference and agrees
        // with both the naive `RefMatrix` and the concrete `MultivariateNormal`.
        let rows: [&[f64]; 3] = [&[2.0, 0.3, 0.1], &[0.3, 1.5, -0.2], &[0.1, -0.2, 1.0]];
        let mean = [0.5, -1.0, 0.2];
        let x = [1.2, -0.3, 0.7];

        let on_dense = multivariate_normal_log_pdf::<DenseMatrix>(
            DenseVector::new(x.to_vec()),
            DenseVector::new(mean.to_vec()),
            dense(3, &rows),
        );
        let on_ref = multivariate_normal_log_pdf(
            RefVector(x.to_vec()),
            RefVector(mean.to_vec()),
            ref_matrix(3, &rows),
        );
        let mvn = MultivariateNormal::new(mean.to_vec(), dense(3, &rows)).unwrap();
        close(on_dense, on_ref, 1e-10);
        close(on_dense, mvn.ln_pdf(&x), 1e-10);
    }

    #[test]
    fn lazy_matrix_mvn_entropy_and_kl_match_distribution() {
        let rows0: [&[f64]; 2] = [&[2.0, 0.3], &[0.3, 1.5]];
        let rows1: [&[f64]; 2] = [&[1.4, -0.2], &[-0.2, 1.1]];
        let mean0 = [0.5, -1.0];
        let mean1 = [0.2, 0.4];

        let mvn0 = MultivariateNormal::new(mean0.to_vec(), dense(2, &rows0)).unwrap();
        let mvn1 = MultivariateNormal::new(mean1.to_vec(), dense(2, &rows1)).unwrap();

        close(
            multivariate_normal_entropy(ref_matrix(2, &rows0)),
            mvn0.entropy().unwrap(),
            1e-10,
        );
        close(
            multivariate_normal_kl(
                RefVector(mean0.to_vec()),
                ref_matrix(2, &rows0),
                RefVector(mean1.to_vec()),
                ref_matrix(2, &rows1),
            ),
            mvn0.kl_divergence(&mvn1),
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_wishart_log_pdf_matches_distribution() {
        let scale: [&[f64]; 2] = [&[2.0, 0.4], &[0.4, 1.3]];
        let observation: [&[f64]; 2] = [&[3.0, 0.5], &[0.5, 2.2]];
        let freedom = 5.0;

        let wishart = Wishart::new(freedom, dense(2, &scale)).unwrap();
        close(
            wishart_log_pdf(ref_matrix(2, &observation), ref_matrix(2, &scale), freedom),
            wishart.ln_pdf_matrix(&dense(2, &observation)),
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_inverse_wishart_log_pdf_matches_distribution() {
        let scale: [&[f64]; 2] = [&[2.0, 0.4], &[0.4, 1.3]];
        let sigma: [&[f64]; 2] = [&[1.6, 0.2], &[0.2, 0.9]];
        let freedom = 5.0;

        let inv_wishart = InverseWishart::new(freedom, dense(2, &scale)).unwrap();
        close(
            inverse_wishart_log_pdf(ref_matrix(2, &sigma), ref_matrix(2, &scale), freedom),
            inv_wishart.ln_pdf_matrix(&dense(2, &sigma)),
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_wishart_entropy_matches_distribution() {
        let scale: [&[f64]; 2] = [&[2.0, 0.4], &[0.4, 1.3]];
        let freedom = 5.0;

        let wishart = Wishart::new(freedom, dense(2, &scale)).unwrap();
        close(
            wishart_entropy(ref_matrix(2, &scale), freedom),
            wishart.entropy(),
            1e-10,
        );

        let inv_wishart = InverseWishart::new(freedom, dense(2, &scale)).unwrap();
        close(
            inverse_wishart_entropy(ref_matrix(2, &scale), freedom),
            inv_wishart.entropy(),
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_mvt_entropy_matches_distribution() {
        let scale: [&[f64]; 2] = [&[2.0, 0.5], &[0.5, 1.3]];
        let freedom = 6.0;
        let mvt = MultivariateStudentT::new(freedom, vec![0.0, 0.0], dense(2, &scale)).unwrap();
        close(
            multivariate_student_t_entropy(ref_matrix(2, &scale), freedom),
            mvt.entropy().unwrap(),
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_gp_marginal_likelihood_matches_zero_mean_mvn() {
        // The GP log-evidence is the MVN log-density at zero mean.
        let kernel: [&[f64]; 3] = [&[1.2, 0.4, 0.1], &[0.4, 1.0, 0.2], &[0.1, 0.2, 0.8]];
        let y = [0.3, -0.7, 0.5];
        let mvn = MultivariateNormal::new(vec![0.0; 3], dense(3, &kernel)).unwrap();
        close(
            gaussian_process_log_marginal_likelihood(RefVector(y.to_vec()), ref_matrix(3, &kernel)),
            mvn.ln_pdf(&y),
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_mvt_log_pdf_matches_distribution() {
        let rows: [&[f64]; 2] = [&[2.0, 0.5], &[0.5, 1.3]];
        let location = [0.4, -0.6];
        let x = [1.1, 0.2];
        let freedom = 6.0;

        let mvt = MultivariateStudentT::new(freedom, location.to_vec(), dense(2, &rows)).unwrap();
        let got = multivariate_student_t_log_pdf(
            RefVector(x.to_vec()),
            RefVector(location.to_vec()),
            ref_matrix(2, &rows),
            freedom,
        );
        close(got, mvt.ln_pdf(&x), 1e-10);
    }

    #[test]
    fn lazy_matrix_hotelling_t_squared_matches_mahalanobis() {
        let cov: [&[f64]; 2] = [&[2.0, 0.3], &[0.3, 1.5]];
        let mean = [1.2, -0.4];
        let mu0 = [0.5, 0.1];
        let n = 8.0;
        close(
            hotelling_t_squared(
                n,
                RefVector(mean.to_vec()),
                RefVector(mu0.to_vec()),
                ref_matrix(2, &cov),
            ),
            n * mahalanobis_squared(&mean, &mu0, &dense(2, &cov)),
            1e-10,
        );
    }

    #[test]
    fn lazy_matrix_pooled_covariance_is_the_df_weighted_average() {
        let s0: [&[f64]; 2] = [&[2.0, 0.3], &[0.3, 1.5]];
        let s1: [&[f64]; 2] = [&[1.4, -0.2], &[-0.2, 1.1]];
        let (df0, df1) = (4.0, 6.0);
        let pooled =
            pooled_covariance(&[ref_matrix(2, &s0), ref_matrix(2, &s1)], &[df0, df1]).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected = (df0 * s0[i][j] + df1 * s1[i][j]) / (df0 + df1);
                close(pooled.at(i, j), expected, 1e-12);
            }
        }
    }
}
