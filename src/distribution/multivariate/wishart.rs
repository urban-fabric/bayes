use std::f64::consts::LN_2;

use ramanujan::DenseMatrix;

use super::matrix;
use crate::error::BayesError;
use crate::function::ln_multivariate_gamma;

/// Wishart distribution over symmetric positive-definite matrices.
///
/// For dimension `d`, degrees of freedom `ν > d - 1`, and positive-definite
/// scale matrix `V`, the density is
///
/// `|X|^((ν-d-1)/2) exp(-tr(V⁻¹X)/2)
///  / (2^(νd/2) |V|^(ν/2) Γ_d(ν/2))`.
#[derive(Clone, Debug, PartialEq)]
pub struct Wishart {
    freedom: f64,
    scale: DenseMatrix,
    scale_inv: DenseMatrix,
    ln_det_scale: f64,
}

impl Wishart {
    /// Build a Wishart distribution with `ν > d - 1` and SPD scale matrix.
    pub fn new(freedom: f64, scale: DenseMatrix) -> Result<Self, BayesError> {
        let d = matrix::validate_square_finite(&scale, "scale must be square")?;
        validate_degrees(freedom, d)?;
        let ln_det_scale = matrix::ln_det_spd(&scale)?;
        let scale_inv = matrix::inverse_spd(&scale)?;
        Ok(Self {
            freedom,
            scale,
            scale_inv,
            ln_det_scale,
        })
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }

    pub fn dimension(&self) -> usize {
        self.scale.rows()
    }

    pub fn scale_matrix(&self) -> &DenseMatrix {
        &self.scale
    }

    /// Closed-form log density at an SPD matrix `x`.
    pub fn ln_pdf_matrix(&self, x: &DenseMatrix) -> f64 {
        let d = self.dimension();
        if x.rows() != d || x.cols() != d {
            return f64::NEG_INFINITY;
        }
        let Ok(ln_det_x) = matrix::ln_det_spd(x) else {
            return f64::NEG_INFINITY;
        };
        let trace = matrix::trace_product(&self.scale_inv, x);
        0.5 * (self.freedom - d as f64 - 1.0) * ln_det_x
            - 0.5 * trace
            - 0.5 * self.freedom * d as f64 * LN_2
            - 0.5 * self.freedom * self.ln_det_scale
            - ln_multivariate_gamma(d, self.freedom / 2.0)
    }

    pub fn pdf_matrix(&self, x: &DenseMatrix) -> f64 {
        self.ln_pdf_matrix(x).exp()
    }

    /// `E[X] = νV`.
    pub fn mean_matrix(&self) -> DenseMatrix {
        matrix::scaled(&self.scale, self.freedom)
    }

    /// `mode(X) = (ν - d - 1)V` when `ν >= d + 1`; otherwise the mode is on the
    /// boundary and is not an SPD matrix.
    pub fn mode_matrix(&self) -> Option<DenseMatrix> {
        let factor = self.freedom - self.dimension() as f64 - 1.0;
        (factor >= 0.0).then(|| matrix::scaled(&self.scale, factor))
    }
}

/// Inverse-Wishart distribution over symmetric positive-definite matrices.
///
/// For dimension `d`, degrees of freedom `ν > d - 1`, and positive-definite
/// scale matrix `Ψ`, the density is
///
/// `|Ψ|^(ν/2) |Σ|^(-(ν+d+1)/2) exp(-tr(ΨΣ⁻¹)/2)
///  / (2^(νd/2) Γ_d(ν/2))`.
#[derive(Clone, Debug, PartialEq)]
pub struct InverseWishart {
    freedom: f64,
    scale: DenseMatrix,
    ln_det_scale: f64,
}

impl InverseWishart {
    /// Build an inverse-Wishart distribution with `ν > d - 1` and SPD scale matrix.
    pub fn new(freedom: f64, scale: DenseMatrix) -> Result<Self, BayesError> {
        let d = matrix::validate_square_finite(&scale, "scale must be square")?;
        validate_degrees(freedom, d)?;
        let ln_det_scale = matrix::ln_det_spd(&scale)?;
        Ok(Self {
            freedom,
            scale,
            ln_det_scale,
        })
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }

    pub fn dimension(&self) -> usize {
        self.scale.rows()
    }

    pub fn scale_matrix(&self) -> &DenseMatrix {
        &self.scale
    }

    /// Closed-form log density at an SPD matrix `sigma`.
    pub fn ln_pdf_matrix(&self, sigma: &DenseMatrix) -> f64 {
        let d = self.dimension();
        if sigma.rows() != d || sigma.cols() != d {
            return f64::NEG_INFINITY;
        }
        let Ok(ln_det_sigma) = matrix::ln_det_spd(sigma) else {
            return f64::NEG_INFINITY;
        };
        let Ok(sigma_inv) = matrix::inverse_spd(sigma) else {
            return f64::NEG_INFINITY;
        };
        let trace = matrix::trace_product(&self.scale, &sigma_inv);
        0.5 * self.freedom * self.ln_det_scale
            - 0.5 * (self.freedom + d as f64 + 1.0) * ln_det_sigma
            - 0.5 * trace
            - 0.5 * self.freedom * d as f64 * LN_2
            - ln_multivariate_gamma(d, self.freedom / 2.0)
    }

    pub fn pdf_matrix(&self, sigma: &DenseMatrix) -> f64 {
        self.ln_pdf_matrix(sigma).exp()
    }

    /// `E[Σ] = Ψ / (ν - d - 1)`, available for `ν > d + 1`.
    pub fn mean_matrix(&self) -> Option<DenseMatrix> {
        let denom = self.freedom - self.dimension() as f64 - 1.0;
        (denom > 0.0).then(|| matrix::scaled(&self.scale, 1.0 / denom))
    }

    /// `mode(Σ) = Ψ / (ν + d + 1)`.
    pub fn mode_matrix(&self) -> DenseMatrix {
        matrix::scaled(
            &self.scale,
            1.0 / (self.freedom + self.dimension() as f64 + 1.0),
        )
    }
}

fn validate_degrees(freedom: f64, dimension: usize) -> Result<(), BayesError> {
    if !freedom.is_finite() {
        return Err(BayesError::NonFinite {
            name: "freedom",
            value: freedom,
        });
    }
    let min = dimension as f64 - 1.0;
    if freedom <= min {
        return Err(BayesError::Malformed {
            what: "degrees of freedom must exceed dimension - 1",
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::{Continuous, Gamma, InverseGamma};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn wishart_univariate_reduces_to_gamma() {
        let w = Wishart::new(6.0, DenseMatrix::from_rows(&[vec![2.0]])).unwrap();
        let g = Gamma::new(3.0, 0.25).unwrap();
        close(
            w.ln_pdf_matrix(&DenseMatrix::from_rows(&[vec![4.0]])),
            g.ln_pdf(4.0),
            1e-10,
        );
        close(w.mean_matrix().get(0, 0), 12.0, 1e-12);
    }

    #[test]
    fn inverse_wishart_univariate_reduces_to_inverse_gamma() {
        let iw = InverseWishart::new(6.0, DenseMatrix::from_rows(&[vec![8.0]])).unwrap();
        let ig = InverseGamma::new(3.0, 4.0).unwrap();
        close(
            iw.ln_pdf_matrix(&DenseMatrix::from_rows(&[vec![2.0]])),
            ig.ln_pdf(2.0),
            1e-10,
        );
        close(iw.mean_matrix().unwrap().get(0, 0), 2.0, 1e-12);
        close(iw.mode_matrix().get(0, 0), 1.0, 1e-12);
    }

    #[test]
    fn rejects_non_spd_scale_or_degrees() {
        assert!(Wishart::new(1.0, DenseMatrix::identity(3)).is_err());
        assert!(
            InverseWishart::new(
                4.0,
                DenseMatrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 1.0]])
            )
            .is_err()
        );
    }
}
