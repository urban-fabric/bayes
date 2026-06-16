use std::f64::consts::PI;

use rand::Rng;
use rand::distr::Distribution;

use ramanujan::{Cholesky, DenseMatrix};

use super::matrix;
use super::{MultivariateContinuous, MultivariateMoments};
use crate::distribution::continuous::gamma::sample_standard_gamma;
use crate::distribution::continuous::normal::standard_normal_polar;
use crate::error::BayesError;
use crate::function::ln_gamma;

/// Multivariate Student-t distribution with degrees of freedom `ν`, location
/// `μ`, and positive-definite scale matrix `Σ`.
///
/// Density:
///
/// `Γ((ν+d)/2) / (Γ(ν/2)(νπ)^(d/2)|Σ|^(1/2))
///  · (1 + δ/ν)^(-(ν+d)/2)`, where
/// `δ = (x-μ)'Σ⁻¹(x-μ)`.
#[derive(Clone, Debug)]
pub struct MultivariateStudentT {
    freedom: f64,
    location: Vec<f64>,
    scale: DenseMatrix,
    chol: Cholesky<f64>,
    ln_det_scale: f64,
}

impl PartialEq for MultivariateStudentT {
    fn eq(&self, other: &Self) -> bool {
        self.freedom == other.freedom
            && self.location == other.location
            && self.scale == other.scale
    }
}

impl MultivariateStudentT {
    /// Build a multivariate Student-t distribution.
    pub fn new(freedom: f64, location: Vec<f64>, scale: DenseMatrix) -> Result<Self, BayesError> {
        if !freedom.is_finite() {
            return Err(BayesError::NonFinite {
                name: "freedom",
                value: freedom,
            });
        }
        if freedom <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "freedom",
                value: freedom,
            });
        }
        let d = location.len();
        if d == 0 || scale.rows() != d || scale.cols() != d {
            return Err(BayesError::Malformed {
                what: "scale must be square and match the location",
            });
        }
        if location.iter().any(|x| !x.is_finite()) {
            return Err(BayesError::Malformed {
                what: "location entries must be finite",
            });
        }
        let chol = matrix::cholesky_spd(&scale)?;
        let ln_det_scale = 2.0 * (0..d).map(|i| chol.lower().get(i, i).ln()).sum::<f64>();
        Ok(Self {
            freedom,
            location,
            scale,
            chol,
            ln_det_scale,
        })
    }

    /// Degrees of freedom `ν`.
    pub fn freedom(&self) -> f64 {
        self.freedom
    }

    /// Location vector `μ`.
    pub fn location(&self) -> &[f64] {
        &self.location
    }

    /// Scale matrix `Σ`.
    pub fn scale_matrix(&self) -> &DenseMatrix {
        &self.scale
    }

    /// Dimension `d`.
    pub fn dimension(&self) -> usize {
        self.location.len()
    }

    /// Closed-form marginal distribution over a coordinate subset.
    ///
    /// Coordinates are returned in the order supplied by `indices`. Duplicate or
    /// out-of-range indices are rejected because they would create a singular
    /// coordinate map rather than a true marginal.
    pub fn marginal(&self, indices: &[usize]) -> Result<Self, BayesError> {
        validate_indices(indices, self.dimension(), false)?;
        Self::new(
            self.freedom,
            select_vector(&self.location, indices),
            select_matrix(&self.scale, indices, indices),
        )
    }

    /// Closed-form conditional distribution of the unobserved coordinates given
    /// observed coordinates.
    ///
    /// If `X = (X_o, X_u)` is multivariate Student-t with degrees `ν`, then
    /// `X_u | X_o = x_o` is Student-t with degrees `ν + dim(X_o)`, the usual
    /// Schur-complement location, and scale inflated by
    /// `(ν + δ_o)/(ν + dim(X_o))`, where `δ_o` is the observed Mahalanobis
    /// distance under the observed block.
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

        let scale_oo = select_matrix(&self.scale, observed_indices, observed_indices);
        let chol_oo = matrix::cholesky_spd(&scale_oo)?;
        let observed_location = select_vector(&self.location, observed_indices);
        let delta: Vec<f64> = observed_values
            .iter()
            .zip(observed_location)
            .map(|(&x, mu)| x - mu)
            .collect();
        let solved_delta = chol_oo.solve(&delta);
        let quad = delta
            .iter()
            .zip(&solved_delta)
            .map(|(&a, b)| a * b)
            .sum::<f64>();

        let scale_uo = select_matrix(&self.scale, &unobserved, observed_indices);
        let scale_ou = select_matrix(&self.scale, observed_indices, &unobserved);
        let scale_uu = select_matrix(&self.scale, &unobserved, &unobserved);

        let location_u = select_vector(&self.location, &unobserved);
        let cond_location: Vec<f64> = location_u
            .iter()
            .enumerate()
            .map(|(i, &mu)| {
                mu + (0..observed_indices.len())
                    .map(|j| scale_uo.get(i, j) * solved_delta[j])
                    .sum::<f64>()
            })
            .collect();

        let schur = DenseMatrix::from_fn(unobserved.len(), unobserved.len(), |i, j| {
            let col = scale_ou.column(j);
            let solved_col = chol_oo.solve(&col);
            scale_uu.get(i, j)
                - (0..observed_indices.len())
                    .map(|k| scale_uo.get(i, k) * solved_col[k])
                    .sum::<f64>()
        });
        let scale_factor = (self.freedom + quad) / (self.freedom + observed_indices.len() as f64);
        Self::new(
            self.freedom + observed_indices.len() as f64,
            cond_location,
            matrix::scaled(&matrix::symmetrize(&schur), scale_factor),
        )
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
        let location = (0..m)
            .map(|i| b[i] + (0..d).map(|j| a.get(i, j) * self.location[j]).sum::<f64>())
            .collect();
        let scale = DenseMatrix::from_fn(m, m, |r, c| {
            let mut acc = 0.0;
            for i in 0..d {
                for j in 0..d {
                    acc += a.get(r, i) * self.scale.get(i, j) * a.get(c, j);
                }
            }
            acc
        });
        Self::new(self.freedom, location, matrix::symmetrize(&scale))
    }
}

impl Distribution<Vec<f64>> for MultivariateStudentT {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let d = self.dimension();
        let z: Vec<f64> = (0..d).map(|_| standard_normal_polar(rng)).collect();
        let chi_square = 2.0 * sample_standard_gamma(rng, self.freedom / 2.0);
        let scale = (self.freedom / chi_square).sqrt();
        let l = self.chol.lower();
        (0..d)
            .map(|i| self.location[i] + scale * (0..=i).map(|j| l.get(i, j) * z[j]).sum::<f64>())
            .collect()
    }
}

impl MultivariateContinuous for MultivariateStudentT {
    fn dim(&self) -> usize {
        self.dimension()
    }

    fn ln_pdf(&self, x: &[f64]) -> f64 {
        let d = self.dimension();
        if x.len() != d || x.iter().any(|v| !v.is_finite()) {
            return f64::NEG_INFINITY;
        }
        let diff: Vec<f64> = x
            .iter()
            .zip(&self.location)
            .map(|(xi, mi)| xi - mi)
            .collect();
        let solved = self.chol.solve(&diff);
        let quad = diff.iter().zip(solved).map(|(&a, b)| a * b).sum::<f64>();
        let nu = self.freedom;
        ln_gamma((nu + d as f64) / 2.0)
            - ln_gamma(nu / 2.0)
            - 0.5 * (d as f64 * (nu * PI).ln() + self.ln_det_scale)
            - 0.5 * (nu + d as f64) * (1.0 + quad / nu).ln()
    }
}

impl MultivariateMoments for MultivariateStudentT {
    fn mean_vector(&self) -> Vec<f64> {
        if self.freedom > 1.0 {
            self.location.clone()
        } else {
            vec![f64::NAN; self.dimension()]
        }
    }

    fn covariance(&self) -> DenseMatrix {
        if self.freedom > 2.0 {
            matrix::scaled(&self.scale, self.freedom / (self.freedom - 2.0))
        } else {
            DenseMatrix::from_fn(self.dimension(), self.dimension(), |_, _| f64::NAN)
        }
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::{Continuous, StudentT};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn univariate_form_matches_scalar_student_t() {
        let mv = MultivariateStudentT::new(5.0, vec![2.0], DenseMatrix::from_rows(&[vec![9.0]]))
            .unwrap();
        let scalar = StudentT::new_location_scale(5.0, 2.0, 3.0).unwrap();
        close(mv.ln_pdf(&[1.0]), scalar.ln_pdf(1.0), 1e-10);
        close(mv.mean_vector()[0], 2.0, 1e-12);
        close(mv.covariance().get(0, 0), 15.0, 1e-12);
    }

    #[test]
    fn rejects_malformed_scale() {
        assert!(
            MultivariateStudentT::new(
                5.0,
                vec![0.0, 0.0],
                DenseMatrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 1.0]])
            )
            .is_err()
        );
    }

    #[test]
    fn marginal_condition_and_affine_transform_match_closed_forms() {
        let t = MultivariateStudentT::new(
            5.0,
            vec![1.0, 2.0, 3.0],
            DenseMatrix::from_rows(&[
                vec![4.0, 1.0, 0.5],
                vec![1.0, 9.0, 2.0],
                vec![0.5, 2.0, 16.0],
            ]),
        )
        .unwrap();

        let marginal = t.marginal(&[2, 0]).unwrap();
        close(marginal.freedom(), 5.0, 1e-12);
        assert_eq!(marginal.location(), &[3.0, 1.0]);
        close(marginal.scale_matrix().get(0, 0), 16.0, 1e-12);
        close(marginal.scale_matrix().get(0, 1), 0.5, 1e-12);
        close(marginal.scale_matrix().get(1, 1), 4.0, 1e-12);
        assert!(t.marginal(&[0, 0]).is_err());

        let bivar = MultivariateStudentT::new(
            5.0,
            vec![0.0, 0.0],
            DenseMatrix::from_rows(&[vec![4.0, 2.0], vec![2.0, 9.0]]),
        )
        .unwrap();
        let conditional = bivar.condition(&[0], &[4.0]).unwrap();
        close(conditional.freedom(), 6.0, 1e-12);
        assert_eq!(conditional.location().len(), 1);
        close(conditional.location()[0], 2.0, 1e-10);
        close(conditional.scale_matrix().get(0, 0), 12.0, 1e-10);
        assert!(bivar.condition(&[0, 1], &[0.0, 0.0]).is_err());

        let affine = bivar
            .affine_transform(
                &DenseMatrix::from_rows(&[vec![1.0, 1.0], vec![2.0, 0.0]]),
                &[1.0, -1.0],
            )
            .unwrap();
        assert_eq!(affine.location(), &[1.0, -1.0]);
        close(affine.scale_matrix().get(0, 0), 17.0, 1e-12);
        close(affine.scale_matrix().get(0, 1), 12.0, 1e-12);
        close(affine.scale_matrix().get(1, 1), 16.0, 1e-12);
        assert!(
            bivar
                .affine_transform(&DenseMatrix::from_rows(&[vec![1.0, 1.0]]), &[0.0])
                .is_ok()
        );
        assert!(
            bivar
                .affine_transform(&DenseMatrix::from_rows(&[vec![1.0, 0.0]]), &[0.0, 0.0])
                .is_err()
        );
    }
}
