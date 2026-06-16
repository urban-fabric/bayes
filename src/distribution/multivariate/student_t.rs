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
}
