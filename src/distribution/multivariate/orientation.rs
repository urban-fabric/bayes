use ramanujan::DenseMatrix;

use super::matrix;
use crate::error::BayesError;

const UNIT_TOLERANCE: f64 = 1e-10;

/// Closed-form Kent/Fisher-Bingham 5-parameter **potential** on `S²`.
///
/// The unnormalized log density is
/// `κ γ₁·x + β[(γ₂·x)² - (γ₃·x)²]`, with an orthonormal frame
/// `(γ₁, γ₂, γ₃)`. Generic Kent normalization is a special-function series, so
/// this type intentionally exposes only unnormalized energies and ratios.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KentPotential {
    mean_axis: [f64; 3],
    major_axis: [f64; 3],
    minor_axis: [f64; 3],
    kappa: f64,
    beta: f64,
}

impl KentPotential {
    /// Build a Kent potential from an orthonormal frame. Requires `κ >= 0` and
    /// `2|β| <= κ`; the equality case is accepted as a limiting potential.
    pub fn new(
        mean_axis: [f64; 3],
        major_axis: [f64; 3],
        minor_axis: [f64; 3],
        kappa: f64,
        beta: f64,
    ) -> Result<Self, BayesError> {
        if !kappa.is_finite() {
            return Err(BayesError::NonFinite {
                name: "kappa",
                value: kappa,
            });
        }
        if kappa < 0.0 {
            return Err(BayesError::NotPositive {
                name: "kappa",
                value: kappa,
            });
        }
        if !beta.is_finite() {
            return Err(BayesError::NonFinite {
                name: "beta",
                value: beta,
            });
        }
        if 2.0 * beta.abs() > kappa + UNIT_TOLERANCE {
            return Err(BayesError::Malformed {
                what: "Kent potential requires 2|beta| <= kappa",
            });
        }
        validate_unit3(&mean_axis)?;
        validate_unit3(&major_axis)?;
        validate_unit3(&minor_axis)?;
        if dot3(&mean_axis, &major_axis).abs() > UNIT_TOLERANCE
            || dot3(&mean_axis, &minor_axis).abs() > UNIT_TOLERANCE
            || dot3(&major_axis, &minor_axis).abs() > UNIT_TOLERANCE
        {
            return Err(BayesError::Malformed {
                what: "Kent axes must be orthonormal",
            });
        }
        Ok(Self {
            mean_axis,
            major_axis,
            minor_axis,
            kappa,
            beta,
        })
    }

    pub fn mean_axis(&self) -> [f64; 3] {
        self.mean_axis
    }

    pub fn major_axis(&self) -> [f64; 3] {
        self.major_axis
    }

    pub fn minor_axis(&self) -> [f64; 3] {
        self.minor_axis
    }

    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Unnormalized log density at a unit vector on `S²`.
    pub fn ln_unnormalized_density(&self, x: &[f64; 3]) -> f64 {
        if validate_unit3(x).is_err() {
            return f64::NEG_INFINITY;
        }
        let g1 = dot3(&self.mean_axis, x);
        let g2 = dot3(&self.major_axis, x);
        let g3 = dot3(&self.minor_axis, x);
        self.kappa * g1 + self.beta * (g2 * g2 - g3 * g3)
    }

    /// Density ratio `q(x)/q(y)`; the unknown normalizer cancels.
    pub fn density_ratio(&self, x: &[f64; 3], y: &[f64; 3]) -> f64 {
        (self.ln_unnormalized_density(x) - self.ln_unnormalized_density(y)).exp()
    }
}

/// Closed-form Bingham **potential** on `S^{d-1}`.
///
/// The unnormalized log density is `xᵀ A x` for a symmetric matrix `A`.
/// The normalized Bingham constant is a hypergeometric function of matrix
/// argument, so this type exposes energies and ratios but does not implement a
/// normalized density trait.
#[derive(Clone, Debug, PartialEq)]
pub struct BinghamPotential {
    concentration: DenseMatrix,
}

impl BinghamPotential {
    /// Build from a finite symmetric concentration matrix.
    pub fn new(concentration: DenseMatrix) -> Result<Self, BayesError> {
        if concentration.rows() < 2 {
            return Err(BayesError::Malformed {
                what: "Bingham potential needs dimension at least two",
            });
        }
        matrix::validate_symmetric(&concentration)?;
        Ok(Self { concentration })
    }

    pub fn dimension(&self) -> usize {
        self.concentration.rows()
    }

    pub fn concentration_matrix(&self) -> &DenseMatrix {
        &self.concentration
    }

    /// Bingham potentials are antipodally symmetric: `q(x) = q(-x)`.
    pub fn is_antipodally_symmetric(&self) -> bool {
        true
    }

    /// Unnormalized log density at a unit vector.
    pub fn ln_unnormalized_density(&self, x: &[f64]) -> f64 {
        let d = self.dimension();
        if x.len() != d || x.iter().any(|v| !v.is_finite()) {
            return f64::NEG_INFINITY;
        }
        if (matrix::norm2(x) - 1.0).abs() > UNIT_TOLERANCE {
            return f64::NEG_INFINITY;
        }
        (0..d)
            .map(|i| {
                x[i] * (0..d)
                    .map(|j| self.concentration.get(i, j) * x[j])
                    .sum::<f64>()
            })
            .sum()
    }

    /// Density ratio `q(x)/q(y)`; the unknown normalizer cancels.
    pub fn density_ratio(&self, x: &[f64], y: &[f64]) -> f64 {
        (self.ln_unnormalized_density(x) - self.ln_unnormalized_density(y)).exp()
    }
}

fn validate_unit3(x: &[f64; 3]) -> Result<(), BayesError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(BayesError::Malformed {
            what: "axis entries must be finite",
        });
    }
    let norm = dot3(x, x);
    if (norm - 1.0).abs() > UNIT_TOLERANCE {
        return Err(BayesError::Malformed {
            what: "axis must be unit length",
        });
    }
    Ok(())
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn kent_potential_scores_and_ratios_are_closed_form() {
        let kent = KentPotential::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 4.0, 1.0)
            .unwrap();
        close(kent.ln_unnormalized_density(&[1.0, 0.0, 0.0]), 4.0, 1e-12);
        close(kent.ln_unnormalized_density(&[0.0, 1.0, 0.0]), 1.0, 1e-12);
        close(kent.ln_unnormalized_density(&[0.0, 0.0, 1.0]), -1.0, 1e-12);
        close(
            kent.density_ratio(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]),
            3.0_f64.exp(),
            1e-12,
        );
        assert!(
            KentPotential::new([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 4.0, 1.0)
                .is_err()
        );
    }

    #[test]
    fn bingham_potential_is_antipodally_symmetric() {
        let bingham = BinghamPotential::new(DenseMatrix::from_rows(&[
            vec![2.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0],
            vec![0.0, 0.0, 0.5],
        ]))
        .unwrap();
        let x = [1.0, 0.0, 0.0];
        let anti = [-1.0, 0.0, 0.0];
        close(bingham.ln_unnormalized_density(&x), 2.0, 1e-12);
        close(
            bingham.ln_unnormalized_density(&x),
            bingham.ln_unnormalized_density(&anti),
            1e-12,
        );
        assert!(bingham.is_antipodally_symmetric());
        assert_eq!(
            bingham.ln_unnormalized_density(&[2.0, 0.0, 0.0]),
            f64::NEG_INFINITY
        );
    }
}
