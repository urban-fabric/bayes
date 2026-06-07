//! Minimal dense linear algebra: a Cholesky factorization of a symmetric
//! positive-definite matrix with its triangular solves (for Gaussian processes),
//! and a general Gaussian-elimination solver (for ordinary least squares).
//! Hand-rolled (small matrices) so the numerics core needs no `nalgebra` dependency.
#![allow(clippy::needless_range_loop)]

/// A dense row-major square matrix as `Vec<Vec<f64>>`.
pub(crate) type Matrix = Vec<Vec<f64>>;

/// Lower-triangular Cholesky factor `L` with `L Lᵀ = a`, for symmetric
/// positive-definite `a`. A small `jitter` is added to the diagonal for numerical
/// PSD-ness. `None` if the matrix is not positive-definite even with jitter.
pub(crate) fn cholesky(a: &[Vec<f64>], jitter: f64) -> Option<Matrix> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                let d = sum + jitter;
                if d <= 0.0 {
                    return None;
                }
                l[i][j] = d.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L y = b` for lower-triangular `L` (forward substitution).
pub(crate) fn solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    y
}

/// Solve `Lᵀ x = y` for lower-triangular `L` (back substitution).
pub(crate) fn solve_upper_transpose(l: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    x
}

/// Solve the SPD system `K x = b` given `L = chol(K)`: `x = Lᵀ \ (L \ b)`.
pub(crate) fn cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    solve_upper_transpose(l, &solve_lower(l, b))
}

/// Solve a general dense system `A x = b` by Gaussian elimination with partial
/// pivoting (`A` is `k×k`, consumed in place). Returns a zero vector if the system
/// is singular (no usable pivot). Unlike [`cholesky_solve`], this needs no
/// positive-definiteness — used by ordinary least squares on the normal equations.
pub(crate) fn gaussian_solve(mut a: Matrix, mut b: Vec<f64>) -> Vec<f64> {
    let k = b.len();
    for col in 0..k {
        let Some(pivot) = (col..k).max_by(|&r1, &r2| {
            a[r1][col]
                .abs()
                .partial_cmp(&a[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            return vec![0.0; k];
        };
        if a[pivot][col].abs() < 1e-300 {
            return vec![0.0; k];
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in (col + 1)..k {
            let factor = a[row][col] / a[col][col];
            for c in col..k {
                a[row][c] -= factor * a[col][c];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; k];
    for row in (0..k).rev() {
        let mut sum = b[row];
        for c in (row + 1)..k {
            sum -= a[row][c] * x[c];
        }
        x[row] = sum / a[row][row];
    }
    x
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn cholesky_reconstructs_the_matrix() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky(&a, 0.0).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let v: f64 = (0..2).map(|k| l[i][k] * l[j][k]).sum();
                close(v, a[i][j], 1e-12);
            }
        }
    }

    #[test]
    fn cholesky_solve_inverts() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky(&a, 0.0).unwrap();
        let b = [1.0, 2.0];
        let x = cholesky_solve(&l, &b);
        for i in 0..2 {
            let v: f64 = (0..2).map(|j| a[i][j] * x[j]).sum();
            close(v, b[i], 1e-12);
        }
    }

    #[test]
    fn non_pd_returns_none() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(cholesky(&a, 0.0).is_none());
    }
}
