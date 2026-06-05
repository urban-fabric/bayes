//! Ordinary least squares: **simple linear regression** ([`LinearRegression`] —
//! fit `y ≈ slope·x + intercept`, with `R²`) and **multiple regression over an
//! explicit design matrix** ([`fit_least_squares`]).

use super::summary::{correlation, covariance, mean, variance};
use crate::linalg::gaussian_solve;

/// A fitted simple (one-predictor) linear regression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
    /// Coefficient of determination `R²` — the fraction of variance explained.
    pub r_squared: f64,
}

impl LinearRegression {
    /// Fit `y` against `x` by ordinary least squares. `None` if the lengths differ,
    /// there are fewer than 2 points, or `x` has zero variance (slope undefined).
    pub fn fit(x: &[f64], y: &[f64]) -> Option<Self> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }
        let var_x = variance(x);
        if var_x <= 0.0 || var_x.is_nan() {
            return None;
        }
        let slope = covariance(x, y) / var_x;
        let intercept = mean(y) - slope * mean(x);
        let r = correlation(x, y);
        Some(Self {
            slope,
            intercept,
            r_squared: r * r,
        })
    }

    /// Predict `y` at a new `x`.
    pub fn predict(&self, x: f64) -> f64 {
        self.intercept + self.slope * x
    }
}

/// Ordinary least squares over an explicit **design matrix**: given one basis row
/// per sample (`design[i]` — the predictor values for sample `i`) and the observed
/// responses `y`, return the fitted coefficient for each column.
///
/// Solves the ridge-regularised normal equations `(XᵀX + εI) β = Xᵀy` by Gaussian
/// elimination with partial pivoting; the small ridge `ε`, scaled per column,
/// keeps near-collinear designs stable. There is no implicit intercept — add a
/// constant column to the design if you want one.
///
/// Returns a zero vector when the system is underdetermined (fewer samples than
/// columns) or the design is empty.
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
    // Ridge, scaled by each column's magnitude, to regularise collinear bases.
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
    fn recovers_an_exact_line() {
        // y = 3x + 2 exactly ⇒ slope 3, intercept 2, R² = 1.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v + 2.0).collect();
        let fit = LinearRegression::fit(&x, &y).unwrap();
        close(fit.slope, 3.0, 1e-12);
        close(fit.intercept, 2.0, 1e-12);
        close(fit.r_squared, 1.0, 1e-12);
        close(fit.predict(10.0), 32.0, 1e-12);
    }

    #[test]
    fn noisy_fit_has_r_squared_below_one() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.1, 3.9, 6.2, 7.8, 10.1]; // ≈ 2x with noise
        let fit = LinearRegression::fit(&x, &y).unwrap();
        assert!(fit.slope > 1.8 && fit.slope < 2.2);
        assert!(fit.r_squared > 0.99 && fit.r_squared < 1.0);
    }

    #[test]
    fn degenerate_x_returns_none() {
        assert!(LinearRegression::fit(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn least_squares_recovers_multiple_coefficients() {
        // Two predictors plus a constant column: y = 2·x₁ - 3·x₂ + 5.
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
        // One sample, two columns ⇒ fewer samples than columns.
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
