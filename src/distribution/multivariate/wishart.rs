use std::f64::consts::LN_2;

use ramanujan::DenseMatrix;

use super::matrix;
use crate::error::BayesError;
use crate::function::{digamma, ln_multivariate_gamma};

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

    /// The natural log of the density normalizer denominator:
    /// `ln(2^(νd/2) |V|^(ν/2) Γ_d(ν/2))`.
    pub fn ln_normalizer(&self) -> f64 {
        let d = self.dimension();
        0.5 * self.freedom * d as f64 * LN_2
            + 0.5 * self.freedom * self.ln_det_scale
            + ln_multivariate_gamma(d, self.freedom / 2.0)
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
        0.5 * (self.freedom - d as f64 - 1.0) * ln_det_x - 0.5 * trace - self.ln_normalizer()
    }

    pub fn pdf_matrix(&self, x: &DenseMatrix) -> f64 {
        self.ln_pdf_matrix(x).exp()
    }

    /// `E[X] = νV`.
    pub fn mean_matrix(&self) -> DenseMatrix {
        matrix::scaled(&self.scale, self.freedom)
    }

    /// `E[log |X|] = Σᵢ ψ((ν + 1 - i)/2) + d log 2 + log |V|`.
    pub fn expected_log_determinant(&self) -> f64 {
        expected_log_det_wishart(self.dimension(), self.freedom, self.ln_det_scale)
    }

    /// `E[X⁻¹] = V⁻¹ / (ν - d - 1)`, available when `ν > d + 1`.
    pub fn inverse_mean_matrix(&self) -> Option<DenseMatrix> {
        let denom = self.freedom - self.dimension() as f64 - 1.0;
        (denom > 0.0).then(|| matrix::scaled(&self.scale_inv, 1.0 / denom))
    }

    /// Covariance between matrix entries:
    /// `Cov(X_ij, X_kl) = ν(V_ik V_jl + V_il V_jk)`.
    ///
    /// Returns `NaN` for out-of-range indices.
    pub fn covariance_entry(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let d = self.dimension();
        if [i, j, k, l].iter().any(|&idx| idx >= d) {
            return f64::NAN;
        }
        self.freedom
            * (self.scale.get(i, k) * self.scale.get(j, l)
                + self.scale.get(i, l) * self.scale.get(j, k))
    }

    /// Variance of matrix entry `X_ij`.
    pub fn variance_entry(&self, i: usize, j: usize) -> f64 {
        self.covariance_entry(i, j, i, j)
    }

    /// Entrywise variance matrix with `(i,j) = Var(X_ij)`.
    pub fn variance_matrix(&self) -> DenseMatrix {
        let d = self.dimension();
        DenseMatrix::from_fn(d, d, |i, j| self.variance_entry(i, j))
    }

    /// Differential entropy in nats.
    pub fn entropy(&self) -> f64 {
        let d = self.dimension();
        -0.5 * (self.freedom - d as f64 - 1.0) * self.expected_log_determinant()
            + 0.5 * self.freedom * d as f64
            + self.ln_normalizer()
    }

    /// Distribution of `X⁻¹` when `X ~ Wishart(ν, V)`:
    /// `InverseWishart(ν, V⁻¹)`.
    pub fn inverse_distribution(&self) -> InverseWishart {
        InverseWishart::new(self.freedom, self.scale_inv.clone()).unwrap_or_else(|_| unreachable!())
    }

    /// `mode(X) = (ν - d - 1)V` when `ν >= d + 1`; otherwise the mode is on the
    /// boundary and is not an SPD matrix.
    pub fn mode_matrix(&self) -> Option<DenseMatrix> {
        let factor = self.freedom - self.dimension() as f64 - 1.0;
        (factor > 0.0).then(|| matrix::scaled(&self.scale, factor))
    }

    /// Congruence image `A X Aᵀ` for nonsingular square `A`.
    ///
    /// If `X ~ Wishart(ν, V)`, then `A X Aᵀ ~ Wishart(ν, A V Aᵀ)`.
    pub fn scale_by_congruence(&self, a: &DenseMatrix) -> Result<Self, BayesError> {
        Self::new(
            self.freedom,
            matrix::symmetrize(&congruence_transform(a, &self.scale)?),
        )
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
    scale_inv: DenseMatrix,
    ln_det_scale: f64,
}

impl InverseWishart {
    /// Build an inverse-Wishart distribution with `ν > d - 1` and SPD scale matrix.
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

    /// The natural log of the leading density constant:
    /// `ln(|Ψ|^(ν/2) / (2^(νd/2) Γ_d(ν/2)))`.
    pub fn ln_normalizer(&self) -> f64 {
        let d = self.dimension();
        0.5 * self.freedom * self.ln_det_scale
            - 0.5 * self.freedom * d as f64 * LN_2
            - ln_multivariate_gamma(d, self.freedom / 2.0)
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
        self.ln_normalizer() - 0.5 * (self.freedom + d as f64 + 1.0) * ln_det_sigma - 0.5 * trace
    }

    pub fn pdf_matrix(&self, sigma: &DenseMatrix) -> f64 {
        self.ln_pdf_matrix(sigma).exp()
    }

    /// `E[Σ] = Ψ / (ν - d - 1)`, available for `ν > d + 1`.
    pub fn mean_matrix(&self) -> Option<DenseMatrix> {
        let denom = self.freedom - self.dimension() as f64 - 1.0;
        (denom > 0.0).then(|| matrix::scaled(&self.scale, 1.0 / denom))
    }

    /// `E[Σ⁻¹] = ν Ψ⁻¹`.
    pub fn expected_inverse_matrix(&self) -> DenseMatrix {
        matrix::scaled(&self.scale_inv, self.freedom)
    }

    /// `E[log |Σ|] = log |Ψ| - d log 2 - Σᵢ ψ((ν + 1 - i)/2)`.
    pub fn expected_log_determinant(&self) -> f64 {
        self.ln_det_scale
            - self.dimension() as f64 * LN_2
            - digamma_wishart_sum(self.dimension(), self.freedom)
    }

    /// Differential entropy in nats.
    pub fn entropy(&self) -> f64 {
        let d = self.dimension();
        -self.ln_normalizer()
            + 0.5 * (self.freedom + d as f64 + 1.0) * self.expected_log_determinant()
            + 0.5 * self.freedom * d as f64
    }

    /// Covariance between matrix entries, available when `ν > d + 3`.
    ///
    /// `Cov(Σ_ij, Σ_kl) =
    /// [2Ψ_ijΨ_kl + (ν-d-1)(Ψ_ikΨ_jl + Ψ_ilΨ_kj)]
    /// / [(ν-d)(ν-d-1)^2(ν-d-3)]`.
    pub fn covariance_entry(&self, i: usize, j: usize, k: usize, l: usize) -> Option<f64> {
        let d = self.dimension();
        if [i, j, k, l].iter().any(|&idx| idx >= d) {
            return None;
        }
        let nu_minus_d = self.freedom - d as f64;
        if nu_minus_d <= 3.0 {
            return None;
        }
        Some(inverse_wishart_covariance_entry_unchecked(
            &self.scale,
            nu_minus_d,
            i,
            j,
            k,
            l,
        ))
    }

    /// Variance of matrix entry `Σ_ij`, available when `ν > d + 3`.
    pub fn variance_entry(&self, i: usize, j: usize) -> Option<f64> {
        self.covariance_entry(i, j, i, j)
    }

    /// Entrywise variance matrix with `(i,j) = Var(Σ_ij)`, available when
    /// `ν > d + 3`.
    pub fn variance_matrix(&self) -> Option<DenseMatrix> {
        let d = self.dimension();
        let nu_minus_d = self.freedom - d as f64;
        (nu_minus_d > 3.0).then(|| {
            DenseMatrix::from_fn(d, d, |i, j| {
                inverse_wishart_covariance_entry_unchecked(&self.scale, nu_minus_d, i, j, i, j)
            })
        })
    }

    /// Distribution of `Σ⁻¹` when `Σ ~ InverseWishart(ν, Ψ)`:
    /// `Wishart(ν, Ψ⁻¹)`.
    pub fn inverse_distribution(&self) -> Wishart {
        Wishart::new(self.freedom, self.scale_inv.clone()).unwrap_or_else(|_| unreachable!())
    }

    /// `mode(Σ) = Ψ / (ν + d + 1)`.
    pub fn mode_matrix(&self) -> DenseMatrix {
        matrix::scaled(
            &self.scale,
            1.0 / (self.freedom + self.dimension() as f64 + 1.0),
        )
    }

    /// Congruence image `A Σ Aᵀ` for nonsingular square `A`.
    ///
    /// If `Σ ~ InverseWishart(ν, Ψ)`, then
    /// `A Σ Aᵀ ~ InverseWishart(ν, A Ψ Aᵀ)`.
    pub fn scale_by_congruence(&self, a: &DenseMatrix) -> Result<Self, BayesError> {
        Self::new(
            self.freedom,
            matrix::symmetrize(&congruence_transform(a, &self.scale)?),
        )
    }
}

impl crate::distribution::KlDivergence for Wishart {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let d = self.dimension();
        if other.dimension() != d {
            return f64::INFINITY;
        }
        let expected_log_det = self.expected_log_determinant();
        let trace = matrix::trace_product(&other.scale_inv, &self.scale);
        0.5 * (self.freedom - other.freedom) * expected_log_det
            + 0.5 * self.freedom * (trace - d as f64)
            + other.ln_normalizer()
            - self.ln_normalizer()
    }
}

impl crate::distribution::KlDivergence for InverseWishart {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let d = self.dimension();
        if other.dimension() != d {
            return f64::INFINITY;
        }
        let expected_log_det = self.expected_log_determinant();
        let trace = matrix::trace_product(&other.scale, &self.scale_inv);
        self.ln_normalizer()
            - other.ln_normalizer()
            - 0.5 * (self.freedom - other.freedom) * expected_log_det
            - 0.5 * self.freedom * d as f64
            + 0.5 * self.freedom * trace
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

fn digamma_wishart_sum(dimension: usize, freedom: f64) -> f64 {
    (1..=dimension)
        .map(|i| digamma((freedom + 1.0 - i as f64) / 2.0))
        .sum()
}

fn expected_log_det_wishart(dimension: usize, freedom: f64, ln_det_scale: f64) -> f64 {
    digamma_wishart_sum(dimension, freedom) + dimension as f64 * LN_2 + ln_det_scale
}

fn inverse_wishart_covariance_entry_unchecked(
    scale: &DenseMatrix,
    nu_minus_d: f64,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
) -> f64 {
    let denom = nu_minus_d * (nu_minus_d - 1.0).powi(2) * (nu_minus_d - 3.0);
    let numerator = 2.0 * scale.get(i, j) * scale.get(k, l)
        + (nu_minus_d - 1.0)
            * (scale.get(i, k) * scale.get(j, l) + scale.get(i, l) * scale.get(k, j));
    numerator / denom
}

fn congruence_transform(a: &DenseMatrix, matrix: &DenseMatrix) -> Result<DenseMatrix, BayesError> {
    let d = matrix.rows();
    if a.rows() != d || a.cols() != d {
        return Err(BayesError::Malformed {
            what: "congruence transform must be square and match the distribution dimension",
        });
    }
    if !a.is_finite() {
        return Err(BayesError::Malformed {
            what: "congruence transform entries must be finite",
        });
    }
    if a.try_inverse().is_err() {
        return Err(BayesError::Malformed {
            what: "congruence transform must be nonsingular",
        });
    }
    Ok(DenseMatrix::from_fn(d, d, |i, j| {
        let mut acc = 0.0;
        for r in 0..d {
            for c in 0..d {
                acc += a.get(i, r) * matrix.get(r, c) * a.get(j, c);
            }
        }
        acc
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::{Continuous, Gamma, InverseGamma, KlDivergence, Moments};

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
        close(
            w.expected_log_determinant(),
            digamma(3.0) - 0.25_f64.ln(),
            1e-12,
        );
        close(w.entropy(), g.entropy().unwrap(), 1e-10);
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
        close(
            iw.expected_log_determinant(),
            4.0_f64.ln() - digamma(3.0),
            1e-12,
        );
        close(iw.entropy(), ig.entropy().unwrap(), 1e-10);
    }

    #[test]
    fn inverse_transforms_swap_wishart_families() {
        let w = Wishart::new(6.0, DenseMatrix::from_rows(&[vec![2.0]])).unwrap();
        let iw = w.inverse_distribution();
        close(iw.freedom(), 6.0, 1e-12);
        close(iw.scale_matrix().get(0, 0), 0.5, 1e-12);

        let back = iw.inverse_distribution();
        close(back.freedom(), 6.0, 1e-12);
        close(back.scale_matrix().get(0, 0), 2.0, 1e-10);
    }

    #[test]
    fn wishart_matrix_entry_covariances_are_closed_form() {
        let w = Wishart::new(
            5.0,
            DenseMatrix::from_rows(&[vec![2.0, 0.5], vec![0.5, 1.0]]),
        )
        .unwrap();
        close(w.covariance_entry(0, 0, 1, 1), 2.5, 1e-12);
        close(w.variance_entry(0, 1), 11.25, 1e-12);
        close(w.variance_matrix().get(0, 1), 11.25, 1e-12);
        assert!(w.covariance_entry(0, 0, 2, 0).is_nan());
        let inverse_mean = w.inverse_mean_matrix().unwrap();
        close(
            inverse_mean.get(0, 0),
            1.0 / 1.75 / (5.0 - 2.0 - 1.0),
            1e-10,
        );
    }

    #[test]
    fn inverse_wishart_entry_variances_are_closed_form_when_available() {
        let iw = InverseWishart::new(
            8.0,
            DenseMatrix::from_rows(&[vec![2.0, 0.5], vec![0.5, 1.0]]),
        )
        .unwrap();
        let diagonal = iw.variance_entry(0, 0).unwrap();
        close(diagonal, 8.0 / 75.0, 1e-12);
        close(iw.variance_matrix().unwrap().get(0, 0), diagonal, 1e-12);
        assert!(iw.covariance_entry(0, 0, 2, 0).is_none());

        let low_freedom = InverseWishart::new(5.0, DenseMatrix::identity(2)).unwrap();
        assert!(low_freedom.variance_entry(0, 0).is_none());
        assert!(low_freedom.variance_matrix().is_none());
    }

    #[test]
    fn congruence_transforms_update_scale_matrices() {
        let transform = DenseMatrix::from_rows(&[vec![2.0, 0.0], vec![0.0, 3.0]]);
        let scale = DenseMatrix::from_rows(&[vec![2.0, 0.5], vec![0.5, 1.0]]);
        let expected = DenseMatrix::from_rows(&[vec![8.0, 3.0], vec![3.0, 9.0]]);

        let w = Wishart::new(6.0, scale.clone()).unwrap();
        let mapped = w.scale_by_congruence(&transform).unwrap();
        close(mapped.scale_matrix().get(0, 0), expected.get(0, 0), 1e-12);
        close(mapped.scale_matrix().get(0, 1), expected.get(0, 1), 1e-12);
        close(mapped.scale_matrix().get(1, 1), expected.get(1, 1), 1e-12);

        let iw = InverseWishart::new(8.0, scale).unwrap();
        let mapped_inverse = iw.scale_by_congruence(&transform).unwrap();
        close(
            mapped_inverse.scale_matrix().get(0, 0),
            expected.get(0, 0),
            1e-12,
        );
        close(
            mapped_inverse.scale_matrix().get(0, 1),
            expected.get(0, 1),
            1e-12,
        );
        close(
            mapped_inverse.scale_matrix().get(1, 1),
            expected.get(1, 1),
            1e-12,
        );

        let singular = DenseMatrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 0.0]]);
        assert!(w.scale_by_congruence(&singular).is_err());
        assert!(iw.scale_by_congruence(&singular).is_err());
    }

    #[test]
    fn wishart_mode_exists_only_inside_the_spd_cone() {
        let boundary = Wishart::new(3.0, DenseMatrix::identity(2)).unwrap();
        assert!(boundary.mode_matrix().is_none());
        let interior = Wishart::new(4.0, DenseMatrix::identity(2)).unwrap();
        close(interior.mode_matrix().unwrap().get(0, 0), 1.0, 1e-12);
    }

    #[test]
    fn wishart_kl_reduces_to_gamma_kl_in_one_dimension() {
        let p = Wishart::new(6.0, DenseMatrix::from_rows(&[vec![2.0]])).unwrap();
        let q = Wishart::new(8.0, DenseMatrix::from_rows(&[vec![3.0]])).unwrap();
        let pg = Gamma::new(3.0, 0.25).unwrap();
        let qg = Gamma::new(4.0, 1.0 / 6.0).unwrap();
        close(p.kl_divergence(&q), pg.kl_divergence(&qg), 1e-10);
        close(p.kl_divergence(&p), 0.0, 1e-10);
    }

    #[test]
    fn inverse_wishart_kl_reduces_to_inverse_gamma_kl_in_one_dimension() {
        let p = InverseWishart::new(6.0, DenseMatrix::from_rows(&[vec![8.0]])).unwrap();
        let q = InverseWishart::new(8.0, DenseMatrix::from_rows(&[vec![12.0]])).unwrap();
        let pig = InverseGamma::new(3.0, 4.0).unwrap();
        let qig = InverseGamma::new(4.0, 6.0).unwrap();
        close(p.kl_divergence(&q), pig.kl_divergence(&qig), 1e-10);
        close(p.kl_divergence(&p), 0.0, 1e-10);
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
