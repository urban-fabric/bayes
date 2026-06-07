//! Multiple ordinary least squares over an explicit **design matrix**:
//! [`fit_least_squares`] solves the ridge-regularised normal equations.

use crate::linalg::gaussian_solve;

/// Ordinary least squares over an explicit **design matrix**: given one basis row
/// per sample (`design[i]` — the predictor values for sample `i`) and the observed
/// responses `y`, return the fitted coefficient for each column.
///
/// Solves the ridge-regularised normal equations `(XᵀX + εI) β = Xᵀy` by Gaussian
/// elimination with partial pivoting; the small ridge `ε`, scaled per column,
/// keeps near-collinear designs stable. There is no implicit intercept — add a
/// constant column to the design if you want one.
///
/// Returns a zero vector for a malformed or underdetermined problem: an empty or
/// ragged design, fewer samples than columns, a length mismatch with `y`, or any
/// non-finite entry.
#[must_use]
pub fn fit_least_squares(design: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let k = design.first().map_or(0, Vec::len);
    if k == 0 || design.len() < k || y.len() != design.len() {
        return vec![0.0; k];
    }
    if design.iter().any(|row| row.len() != k)
        || design.iter().flatten().any(|x| !x.is_finite())
        || y.iter().any(|x| !x.is_finite())
    {
        return vec![0.0; k];
    }
    let mut ata = vec![vec![0.0_f64; k]; k];
    let mut aty = vec![0.0_f64; k];
    for (row, &yi) in design.iter().zip(y) {
        for i in 0..k {
            aty[i] += row[i] * yi;
            for j in 0..k {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    for (i, row) in ata.iter_mut().enumerate() {
        let scale = row[i].abs().max(1.0);
        row[i] += 1e-9 * scale;
    }
    gaussian_solve(ata, aty)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn least_squares_recovers_multiple_coefficients() {
        let (b1, b2, b0) = (2.0, -3.0, 5.0);
        let mut design = Vec::new();
        let mut y = Vec::new();
        for (x1, x2) in [
            (1.0, 0.0),
            (0.0, 1.0),
            (2.0, 1.0),
            (3.0, 2.0),
            (1.0, 4.0),
            (5.0, 2.0),
        ] {
            design.push(vec![x1, x2, 1.0]);
            y.push(b1 * x1 + b2 * x2 + b0);
        }
        let coeffs = fit_least_squares(&design, &y);
        close(coeffs[0], b1, 1e-4);
        close(coeffs[1], b2, 1e-4);
        close(coeffs[2], b0, 1e-4);
    }

    #[test]
    fn underdetermined_system_returns_zeros() {
        assert_eq!(fit_least_squares(&[vec![1.0, 2.0]], &[3.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn malformed_design_returns_zeros() {
        assert_eq!(
            fit_least_squares(&[vec![1.0, 2.0], vec![3.0]], &[4.0, 5.0]),
            vec![0.0, 0.0]
        );
        assert_eq!(
            fit_least_squares(&[vec![1.0, 2.0]], &[3.0, 4.0]),
            vec![0.0, 0.0]
        );
        assert_eq!(
            fit_least_squares(&[vec![1.0, f64::NAN], vec![2.0, 3.0]], &[4.0, 5.0]),
            vec![0.0, 0.0]
        );
    }
}
