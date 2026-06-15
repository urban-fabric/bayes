//! Multivariate sample statistics over row-major observations.
#![allow(clippy::needless_range_loop)]

use ramanujan::DenseMatrix;

fn shape(samples: &[Vec<f64>], min_rows: usize) -> Option<(usize, usize)> {
    let n = samples.len();
    let p = samples.first().map_or(0, Vec::len);
    if n < min_rows
        || p == 0
        || samples
            .iter()
            .any(|row| row.len() != p || row.iter().any(|x| !x.is_finite()))
    {
        return None;
    }
    Some((n, p))
}

/// Column means for row-major observations.
pub fn mean_vector(samples: &[Vec<f64>]) -> Vec<f64> {
    let Some((n, p)) = shape(samples, 1) else {
        return Vec::new();
    };
    let mut means = vec![0.0; p];
    for row in samples {
        for j in 0..p {
            means[j] += row[j];
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }
    means
}

/// Unbiased sample covariance matrix for row-major observations.
pub fn covariance_matrix(samples: &[Vec<f64>]) -> DenseMatrix {
    let Some((n, p)) = shape(samples, 2) else {
        return DenseMatrix::zeros(0, 0);
    };
    let means = mean_vector(samples);
    let mut cov = DenseMatrix::zeros(p, p);
    for row in samples {
        for i in 0..p {
            for j in 0..=i {
                cov.set(
                    i,
                    j,
                    cov.get(i, j) + (row[i] - means[i]) * (row[j] - means[j]),
                );
            }
        }
    }
    let denom = n as f64 - 1.0;
    for i in 0..p {
        for j in 0..=i {
            let v = cov.get(i, j) / denom;
            cov.set(i, j, v);
            cov.set(j, i, v);
        }
    }
    cov
}

/// Pearson correlation matrix for row-major observations.
pub fn correlation_matrix(samples: &[Vec<f64>]) -> DenseMatrix {
    let cov = covariance_matrix(samples);
    let p = cov.rows();
    if p == 0 {
        return DenseMatrix::zeros(0, 0);
    }
    DenseMatrix::from_fn(p, p, |i, j| {
        let denom = (cov.get(i, i) * cov.get(j, j)).sqrt();
        if denom > 0.0 {
            cov.get(i, j) / denom
        } else {
            f64::NAN
        }
    })
}

/// Pooled covariance matrix across groups of row-major observations.
pub fn pooled_covariance_matrix(groups: &[&[Vec<f64>]]) -> DenseMatrix {
    if groups.len() < 2 {
        return DenseMatrix::zeros(0, 0);
    }
    let Some((_, p)) = shape(groups[0], 2) else {
        return DenseMatrix::zeros(0, 0);
    };
    let mut pooled = DenseMatrix::zeros(p, p);
    let mut df = 0usize;
    for group in groups {
        let Some((n, gp)) = shape(group, 2) else {
            return DenseMatrix::zeros(0, 0);
        };
        if gp != p {
            return DenseMatrix::zeros(0, 0);
        }
        let cov = covariance_matrix(group);
        for i in 0..p {
            for j in 0..p {
                pooled.set(i, j, pooled.get(i, j) + (n as f64 - 1.0) * cov.get(i, j));
            }
        }
        df += n - 1;
    }
    if df == 0 {
        return DenseMatrix::zeros(0, 0);
    }
    for i in 0..p {
        for j in 0..p {
            pooled.set(i, j, pooled.get(i, j) / df as f64);
        }
    }
    pooled
}

/// Squared Mahalanobis distance `(x-μ)' Σ⁻¹ (x-μ)`.
pub fn mahalanobis_squared(x: &[f64], mean: &[f64], cov: &DenseMatrix) -> f64 {
    let p = x.len();
    if p == 0
        || mean.len() != p
        || cov.rows() != p
        || cov.cols() != p
        || x.iter().chain(mean).any(|v| !v.is_finite())
        || !cov.is_finite()
    {
        return f64::NAN;
    }
    let delta: Vec<f64> = x.iter().zip(mean).map(|(&xi, &mi)| xi - mi).collect();
    let Ok(chol) = DenseMatrix::from_fn(p, p, |i, j| {
        cov.get(i, j) + if i == j { 1e-12 } else { 0.0 }
    })
    .cholesky() else {
        return f64::NAN;
    };
    let solved = chol.solve(&delta);
    delta.iter().zip(solved).map(|(&d, s)| d * s).sum()
}

/// One-sample Hotelling's `T²` statistic against a hypothesized mean vector.
pub fn hotelling_t_squared(samples: &[Vec<f64>], mu0: &[f64]) -> f64 {
    let Some((n, p)) = shape(samples, 2) else {
        return f64::NAN;
    };
    if mu0.len() != p || mu0.iter().any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    let means = mean_vector(samples);
    let cov = covariance_matrix(samples);
    n as f64 * mahalanobis_squared(&means, mu0, &cov)
}

/// **Covariance of two linear combinations**: `Cov(aᵀX, bᵀX) = aᵀ Σ b` for a random
/// vector `X` with covariance `Σ`. The bilinear form of the covariance — the variance
/// of a weighted sum/portfolio, the covariance between two linear functionals, or one
/// step of linear error-propagation. Backed by the covariance-bilinearity proofs in
/// riemann `Bayes/Inequalities` (`covariance_sum_left` — additivity over a finite sum —
/// composed with `covariance_smul_left`). Returns `NaN` on a shape/finiteness mismatch.
pub fn covariance_of_linear_combinations(cov: &DenseMatrix, a: &[f64], b: &[f64]) -> f64 {
    let p = a.len();
    if p == 0
        || b.len() != p
        || cov.rows() != p
        || cov.cols() != p
        || a.iter().chain(b).any(|v| !v.is_finite())
        || !cov.is_finite()
    {
        return f64::NAN;
    }
    let mut acc = 0.0;
    for i in 0..p {
        for j in 0..p {
            acc += a[i] * cov.get(i, j) * b[j];
        }
    }
    acc
}

/// **Variance of a linear combination**: `Var(wᵀX) = wᵀ Σ w` — portfolio variance /
/// linear uncertainty propagation. The `a = b` case of [`covariance_of_linear_combinations`].
/// For an uncorrelated (diagonal) `Σ` it reduces to `Σᵢ wᵢ² σᵢ²` (Bienaymé's identity,
/// riemann `variance_sum_of_uncorrelated`), and the result is `≥ 0` for any genuine
/// (PSD) covariance — a sum of weighted squares (`variance_nonneg`). Returns `NaN` on a
/// shape/finiteness mismatch.
pub fn variance_of_linear_combination(cov: &DenseMatrix, weights: &[f64]) -> f64 {
    covariance_of_linear_combinations(cov, weights, weights)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn covariance_and_correlation_matrices_match_simple_data() {
        let samples = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let cov = covariance_matrix(&samples);
        close(cov.get(0, 0), 1.0, 1e-12);
        close(cov.get(0, 1), 2.0, 1e-12);
        close(cov.get(1, 1), 4.0, 1e-12);
        close(correlation_matrix(&samples).get(0, 1), 1.0, 1e-12);
    }

    #[test]
    fn mahalanobis_uses_covariance_geometry() {
        let cov = DenseMatrix::from_rows(&[vec![4.0, 0.0], vec![0.0, 9.0]]);
        close(
            mahalanobis_squared(&[2.0, 3.0], &[0.0, 0.0], &cov),
            2.0,
            1e-12,
        );
    }

    #[test]
    fn hotelling_is_zero_at_the_sample_mean() {
        let samples = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let m = mean_vector(&samples);
        close(hotelling_t_squared(&samples, &m), 0.0, 1e-9);
    }

    #[test]
    fn linear_combination_variance_is_the_quadratic_form() {
        // Σ = [[4, 1], [1, 9]], w = (1, 2): wᵀΣw = 4 + 2·(2·1) + 4·9 = 44.
        let cov = DenseMatrix::from_rows(&[vec![4.0, 1.0], vec![1.0, 9.0]]);
        close(
            variance_of_linear_combination(&cov, &[1.0, 2.0]),
            44.0,
            1e-12,
        );
        // a unit weight picks out the corresponding variance (Σ's diagonal).
        close(
            variance_of_linear_combination(&cov, &[1.0, 0.0]),
            4.0,
            1e-12,
        );
        // the bilinear form is the variance of the sum X₁+X₂: Var = σ₁²+2σ₁₂+σ₂² = 15.
        close(
            variance_of_linear_combination(&cov, &[1.0, 1.0]),
            15.0,
            1e-12,
        );
    }

    #[test]
    fn bienayme_diagonal_case_is_weighted_variances() {
        // uncorrelated (diagonal Σ): Var(Σ wᵢXᵢ) = Σ wᵢ² σᵢ² (Bienaymé).
        let cov = DenseMatrix::from_rows(&[vec![2.0, 0.0], vec![0.0, 5.0]]);
        close(
            variance_of_linear_combination(&cov, &[3.0, 4.0]),
            9.0 * 2.0 + 16.0 * 5.0,
            1e-12,
        );
    }

    #[test]
    fn covariance_of_combinations_is_symmetric_and_reduces_to_variance() {
        let cov = DenseMatrix::from_rows(&[vec![4.0, 1.0], vec![1.0, 9.0]]);
        let (a, b) = ([1.0, 2.0], [3.0, -1.0]);
        // bilinear + symmetric for symmetric Σ.
        close(
            covariance_of_linear_combinations(&cov, &a, &b),
            covariance_of_linear_combinations(&cov, &b, &a),
            1e-12,
        );
        // a = b recovers the variance.
        close(
            covariance_of_linear_combinations(&cov, &a, &a),
            variance_of_linear_combination(&cov, &a),
            1e-12,
        );
        // shape mismatch ⇒ NaN.
        assert!(covariance_of_linear_combinations(&cov, &a, &[1.0]).is_nan());
    }

    #[test]
    fn portfolio_variance_is_nonnegative_for_a_real_covariance() {
        // wᵀΣw ≥ 0 for any genuine (PSD) covariance — sum of weighted squares.
        let samples = vec![
            vec![1.0, 2.0, 0.5],
            vec![2.0, 1.0, 1.5],
            vec![3.0, 4.0, 0.0],
            vec![0.5, 2.5, 2.0],
        ];
        let cov = covariance_matrix(&samples);
        for w in [[1.0, -1.0, 2.0], [0.3, 0.3, 0.4], [-2.0, 1.0, -0.5]] {
            assert!(variance_of_linear_combination(&cov, &w) >= -1e-12);
        }
    }
}
