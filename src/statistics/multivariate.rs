//! Multivariate sample statistics over row-major observations.
#![allow(clippy::needless_range_loop)]

use crate::linalg::{cholesky, cholesky_solve};

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
pub fn covariance_matrix(samples: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let Some((n, p)) = shape(samples, 2) else {
        return Vec::new();
    };
    let means = mean_vector(samples);
    let mut cov = vec![vec![0.0; p]; p];
    for row in samples {
        for i in 0..p {
            for j in 0..=i {
                cov[i][j] += (row[i] - means[i]) * (row[j] - means[j]);
            }
        }
    }
    let denom = n as f64 - 1.0;
    for i in 0..p {
        for j in 0..=i {
            cov[i][j] /= denom;
            cov[j][i] = cov[i][j];
        }
    }
    cov
}

/// Pearson correlation matrix for row-major observations.
pub fn correlation_matrix(samples: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = covariance_matrix(samples);
    let p = cov.len();
    if p == 0 {
        return Vec::new();
    }
    let mut corr = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            let denom = (cov[i][i] * cov[j][j]).sqrt();
            corr[i][j] = if denom > 0.0 {
                cov[i][j] / denom
            } else {
                f64::NAN
            };
        }
    }
    corr
}

/// Pooled covariance matrix across groups of row-major observations.
pub fn pooled_covariance_matrix(groups: &[&[Vec<f64>]]) -> Vec<Vec<f64>> {
    if groups.len() < 2 {
        return Vec::new();
    }
    let Some((_, p)) = shape(groups[0], 2) else {
        return Vec::new();
    };
    let mut pooled = vec![vec![0.0; p]; p];
    let mut df = 0usize;
    for group in groups {
        let Some((n, gp)) = shape(group, 2) else {
            return Vec::new();
        };
        if gp != p {
            return Vec::new();
        }
        let cov = covariance_matrix(group);
        for i in 0..p {
            for j in 0..p {
                pooled[i][j] += (n as f64 - 1.0) * cov[i][j];
            }
        }
        df += n - 1;
    }
    if df == 0 {
        return Vec::new();
    }
    for row in &mut pooled {
        for x in row {
            *x /= df as f64;
        }
    }
    pooled
}

/// Squared Mahalanobis distance `(x-μ)' Σ⁻¹ (x-μ)`.
pub fn mahalanobis_squared(x: &[f64], mean: &[f64], cov: &[Vec<f64>]) -> f64 {
    let p = x.len();
    if p == 0
        || mean.len() != p
        || cov.len() != p
        || x.iter().chain(mean).any(|v| !v.is_finite())
        || cov
            .iter()
            .any(|row| row.len() != p || row.iter().any(|v| !v.is_finite()))
    {
        return f64::NAN;
    }
    let delta: Vec<f64> = x.iter().zip(mean).map(|(&xi, &mi)| xi - mi).collect();
    let Some(l) = cholesky(cov, 1e-12) else {
        return f64::NAN;
    };
    let solved = cholesky_solve(&l, &delta);
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
        close(cov[0][0], 1.0, 1e-12);
        close(cov[0][1], 2.0, 1e-12);
        close(cov[1][1], 4.0, 1e-12);
        close(correlation_matrix(&samples)[0][1], 1.0, 1e-12);
    }

    #[test]
    fn mahalanobis_uses_covariance_geometry() {
        let cov = vec![vec![4.0, 0.0], vec![0.0, 9.0]];
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
}
