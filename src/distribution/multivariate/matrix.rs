use ramanujan::{Cholesky, DenseMatrix};

use crate::error::BayesError;

const JITTER: f64 = 1e-12;

pub(crate) fn validate_square_finite(
    matrix: &DenseMatrix,
    what: &'static str,
) -> Result<usize, BayesError> {
    if matrix.rows() == 0 || matrix.rows() != matrix.cols() {
        return Err(BayesError::Malformed { what });
    }
    if !matrix.is_finite() {
        return Err(BayesError::Malformed {
            what: "matrix entries must be finite",
        });
    }
    Ok(matrix.rows())
}

pub(crate) fn validate_symmetric(matrix: &DenseMatrix) -> Result<(), BayesError> {
    let n = validate_square_finite(matrix, "matrix must be square")?;
    for i in 0..n {
        for j in (i + 1)..n {
            if (matrix.get(i, j) - matrix.get(j, i)).abs() > 1e-10 {
                return Err(BayesError::Malformed {
                    what: "matrix must be symmetric",
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn cholesky_spd(matrix: &DenseMatrix) -> Result<Cholesky<f64>, BayesError> {
    validate_symmetric(matrix)?;
    DenseMatrix::from_fn(matrix.rows(), matrix.cols(), |i, j| {
        matrix.get(i, j) + if i == j { JITTER } else { 0.0 }
    })
    .cholesky()
    .map_err(|_| BayesError::Malformed {
        what: "matrix is not positive-definite",
    })
}

pub(crate) fn ln_det_spd(matrix: &DenseMatrix) -> Result<f64, BayesError> {
    let chol = cholesky_spd(matrix)?;
    Ok(2.0
        * (0..matrix.rows())
            .map(|i| chol.lower().get(i, i).ln())
            .sum::<f64>())
}

pub(crate) fn inverse_spd(matrix: &DenseMatrix) -> Result<DenseMatrix, BayesError> {
    let n = matrix.rows();
    let chol = cholesky_spd(matrix)?;
    Ok(DenseMatrix::from_fn(n, n, |i, j| {
        let mut unit = vec![0.0; n];
        unit[j] = 1.0;
        chol.solve(&unit)[i]
    }))
}

pub(crate) fn symmetrize(matrix: &DenseMatrix) -> DenseMatrix {
    DenseMatrix::from_fn(matrix.rows(), matrix.cols(), |i, j| {
        0.5 * (matrix.get(i, j) + matrix.get(j, i))
    })
}

pub(crate) fn trace_product(a: &DenseMatrix, b: &DenseMatrix) -> f64 {
    let n = a.rows();
    (0..n)
        .map(|i| (0..n).map(|j| a.get(i, j) * b.get(j, i)).sum::<f64>())
        .sum()
}

pub(crate) fn scaled(matrix: &DenseMatrix, scale: f64) -> DenseMatrix {
    DenseMatrix::from_fn(matrix.rows(), matrix.cols(), |i, j| {
        matrix.get(i, j) * scale
    })
}

pub(crate) fn add(a: &DenseMatrix, b: &DenseMatrix) -> DenseMatrix {
    DenseMatrix::from_fn(a.rows(), a.cols(), |i, j| a.get(i, j) + b.get(i, j))
}

pub(crate) fn outer(x: &[f64], y: &[f64]) -> DenseMatrix {
    DenseMatrix::from_fn(x.len(), y.len(), |i, j| x[i] * y[j])
}

pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

pub(crate) fn norm2(x: &[f64]) -> f64 {
    dot(x, x)
}
