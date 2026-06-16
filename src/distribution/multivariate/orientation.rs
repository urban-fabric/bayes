use std::f64::consts::{PI, TAU};

use hausdorff::space::sphere::Sphere;
use ramanujan::DenseMatrix;

use super::matrix;
use crate::distribution::VonMisesFisher;
use crate::error::BayesError;
use crate::function::bessel::ln_i0;
use crate::function::gamma::ln_gamma;

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

    /// The orthonormal Kent frame as a `3×3` matrix whose rows are
    /// `(mean_axis, major_axis, minor_axis)`.
    pub fn frame_matrix(&self) -> DenseMatrix {
        DenseMatrix::from_rows(&[
            self.mean_axis.to_vec(),
            self.major_axis.to_vec(),
            self.minor_axis.to_vec(),
        ])
    }

    /// One global mode axis. Under the standard Kent constraint `2|β| <= κ`,
    /// the mean axis is a mode of the potential.
    pub fn mode_axis(&self) -> [f64; 3] {
        self.mean_axis
    }

    /// The antipodal direction to [`mode_axis`](Self::mode_axis).
    pub fn anti_mode_axis(&self) -> [f64; 3] {
        neg3(&self.mean_axis)
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

    /// Log density ratio `ln q(x) - ln q(y)`; the unknown normalizer cancels.
    pub fn ln_density_difference(&self, x: &[f64; 3], y: &[f64; 3]) -> f64 {
        self.ln_unnormalized_density(x) - self.ln_unnormalized_density(y)
    }

    /// Density ratio `q(x)/q(y)`; the unknown normalizer cancels.
    pub fn density_ratio(&self, x: &[f64; 3], y: &[f64; 3]) -> f64 {
        self.ln_density_difference(x, y).exp()
    }

    /// Euclidean gradient of the unnormalized log density before projecting to
    /// the sphere tangent space.
    pub fn euclidean_gradient(&self, x: &[f64; 3]) -> Result<[f64; 3], BayesError> {
        validate_unit3(x)?;
        let g2x = dot3(&self.major_axis, x);
        let g3x = dot3(&self.minor_axis, x);
        Ok(add3(
            &scale3(&self.mean_axis, self.kappa),
            &add3(
                &scale3(&self.major_axis, 2.0 * self.beta * g2x),
                &scale3(&self.minor_axis, -2.0 * self.beta * g3x),
            ),
        ))
    }

    /// Riemannian gradient on `S²`: the Euclidean gradient projected onto the
    /// tangent plane at `x`.
    pub fn tangent_gradient(&self, x: &[f64; 3]) -> Result<[f64; 3], BayesError> {
        let grad = self.euclidean_gradient(x)?;
        Ok(project_tangent3(x, &grad))
    }

    /// Whether the Kent potential is exactly the von Mises-Fisher special case
    /// (`β = 0`) on `S²`.
    pub fn is_von_mises_fisher(&self) -> bool {
        self.beta == 0.0
    }

    /// Convert the `β = 0` special case into a normalized von Mises-Fisher
    /// distribution over any 3-dimensional [`Sphere`] representation.
    pub fn as_von_mises_fisher<S: Sphere>(&self) -> Option<VonMisesFisher<S>> {
        self.is_von_mises_fisher()
            .then(|| VonMisesFisher::new(S::from_ambient(&self.mean_axis), self.kappa).ok())
            .flatten()
    }

    /// Closed-form log partition function when this potential is in a normalized
    /// special case. Generic Kent normalization is intentionally not exposed.
    pub fn ln_partition_closed_form(&self) -> Option<f64> {
        self.is_von_mises_fisher()
            .then(|| ln_vmf_s2_partition(self.kappa))
    }

    /// Normalized log density where the closed-form normalizer is available.
    pub fn ln_pdf_closed_form(&self, x: &[f64; 3]) -> Option<f64> {
        self.ln_partition_closed_form()
            .map(|ln_z| self.ln_unnormalized_density(x) - ln_z)
    }

    /// Normalized density where the closed-form normalizer is available.
    pub fn pdf_closed_form(&self, x: &[f64; 3]) -> Option<f64> {
        self.ln_pdf_closed_form(x).map(f64::exp)
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

    /// Trace of the concentration matrix.
    pub fn trace(&self) -> f64 {
        (0..self.dimension())
            .map(|i| self.concentration.get(i, i))
            .sum()
    }

    /// Add `shift · I` to the concentration matrix.
    ///
    /// On the unit sphere this adds a constant `shift` to every log potential,
    /// so density ratios and normalized densities are unchanged.
    pub fn shifted_by_identity(&self, shift: f64) -> Result<Self, BayesError> {
        if !shift.is_finite() {
            return Err(BayesError::NonFinite {
                name: "shift",
                value: shift,
            });
        }
        let d = self.dimension();
        Self::new(DenseMatrix::from_fn(d, d, |i, j| {
            self.concentration.get(i, j) + if i == j { shift } else { 0.0 }
        }))
    }

    /// Remove the trace component `(tr A / d)I`. This canonical representative
    /// has the same density ratios as the original potential.
    pub fn trace_centered(&self) -> Self {
        let shift = -self.trace() / self.dimension() as f64;
        self.shifted_by_identity(shift)
            .unwrap_or_else(|_| unreachable!())
    }

    /// Whether `A` is numerically an identity shift `cI`, making the normalized
    /// law uniform on the sphere.
    pub fn is_isotropic(&self) -> bool {
        let d = self.dimension();
        let diag = self.concentration.get(0, 0);
        (0..d).all(|i| {
            (0..d).all(|j| {
                let expected = if i == j { diag } else { 0.0 };
                (self.concentration.get(i, j) - expected).abs() <= UNIT_TOLERANCE
            })
        })
    }

    /// Bingham potentials are antipodally symmetric: `q(x) = q(-x)`.
    pub fn is_antipodally_symmetric(&self) -> bool {
        true
    }

    /// Quadratic form `xᵀ A x` for a unit vector.
    pub fn quadratic_form(&self, x: &[f64]) -> f64 {
        self.ln_unnormalized_density(x)
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

    /// Log density ratio `ln q(x) - ln q(y)`; the unknown normalizer cancels.
    pub fn ln_density_difference(&self, x: &[f64], y: &[f64]) -> f64 {
        self.ln_unnormalized_density(x) - self.ln_unnormalized_density(y)
    }

    /// Density ratio `q(x)/q(y)`; the unknown normalizer cancels.
    pub fn density_ratio(&self, x: &[f64], y: &[f64]) -> f64 {
        self.ln_density_difference(x, y).exp()
    }

    /// Euclidean gradient of `xᵀ A x`, equal to `2Ax` because `A` is symmetric.
    pub fn euclidean_gradient(&self, x: &[f64]) -> Result<Vec<f64>, BayesError> {
        self.validate_unit_vector(x)?;
        let d = self.dimension();
        Ok((0..d)
            .map(|i| {
                2.0 * (0..d)
                    .map(|j| self.concentration.get(i, j) * x[j])
                    .sum::<f64>()
            })
            .collect())
    }

    /// Riemannian gradient on `S^{d-1}`: the Euclidean gradient projected onto
    /// the tangent plane at `x`.
    pub fn tangent_gradient(&self, x: &[f64]) -> Result<Vec<f64>, BayesError> {
        let grad = self.euclidean_gradient(x)?;
        let radial = matrix::dot(x, &grad);
        Ok(grad
            .iter()
            .zip(x)
            .map(|(&g, &xi)| g - radial * xi)
            .collect())
    }

    /// Closed-form log partition function where available.
    ///
    /// Available cases:
    /// - isotropic `A = cI` in any dimension: `c + log area(S^{d-1})`;
    /// - any two-dimensional Bingham on `S¹`:
    ///   `tr(A)/2 + log(2π I₀(r))`, with
    ///   `r = sqrt(((a-d)/2)^2 + b^2)`.
    pub fn ln_partition_closed_form(&self) -> Option<f64> {
        if self.is_isotropic() {
            return Some(self.concentration.get(0, 0) + ln_sphere_area(self.dimension()));
        }
        if self.dimension() == 2 {
            let a = self.concentration.get(0, 0);
            let b = self.concentration.get(0, 1);
            let d = self.concentration.get(1, 1);
            let center = 0.5 * (a + d);
            let radius = (0.25 * (a - d) * (a - d) + b * b).sqrt();
            return Some(center + TAU.ln() + ln_i0(radius));
        }
        None
    }

    /// Normalized log density where the closed-form normalizer is available.
    pub fn ln_pdf_closed_form(&self, x: &[f64]) -> Option<f64> {
        self.ln_partition_closed_form()
            .map(|ln_z| self.ln_unnormalized_density(x) - ln_z)
    }

    /// Normalized density where the closed-form normalizer is available.
    pub fn pdf_closed_form(&self, x: &[f64]) -> Option<f64> {
        self.ln_pdf_closed_form(x).map(f64::exp)
    }

    fn validate_unit_vector(&self, x: &[f64]) -> Result<(), BayesError> {
        let d = self.dimension();
        if x.len() != d || x.iter().any(|v| !v.is_finite()) {
            return Err(BayesError::Malformed {
                what: "point must be finite and match Bingham dimension",
            });
        }
        if (matrix::norm2(x) - 1.0).abs() > UNIT_TOLERANCE {
            return Err(BayesError::Malformed {
                what: "point must be unit length",
            });
        }
        Ok(())
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

fn neg3(x: &[f64; 3]) -> [f64; 3] {
    [-x[0], -x[1], -x[2]]
}

fn scale3(x: &[f64; 3], scale: f64) -> [f64; 3] {
    [scale * x[0], scale * x[1], scale * x[2]]
}

fn add3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn project_tangent3(x: &[f64; 3], v: &[f64; 3]) -> [f64; 3] {
    let radial = dot3(x, v);
    [
        v[0] - radial * x[0],
        v[1] - radial * x[1],
        v[2] - radial * x[2],
    ]
}

fn ln_vmf_s2_partition(kappa: f64) -> f64 {
    if kappa == 0.0 {
        (4.0 * PI).ln()
    } else if kappa > 50.0 {
        // ln(4π sinh κ / κ) = ln(2π/κ) + κ + ln(1 - exp(-2κ)).
        (2.0 * PI / kappa).ln() + kappa + (-(-2.0 * kappa).exp()).ln_1p()
    } else {
        (4.0 * PI * kappa.sinh() / kappa).ln()
    }
}

fn ln_sphere_area(ambient_dimension: usize) -> f64 {
    2.0_f64.ln() + 0.5 * ambient_dimension as f64 * PI.ln()
        - ln_gamma(0.5 * ambient_dimension as f64)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use hausdorff::space::sphere::UnitVector;

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
        close(kent.frame_matrix().get(1, 1), 1.0, 1e-12);
        assert_eq!(kent.mode_axis(), [1.0, 0.0, 0.0]);
        assert_eq!(kent.anti_mode_axis(), [-1.0, 0.0, 0.0]);
        close(
            kent.ln_density_difference(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]),
            3.0,
            1e-12,
        );
        close(
            kent.density_ratio(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]),
            3.0_f64.exp(),
            1e-12,
        );
        let grad = kent.euclidean_gradient(&[0.0, 1.0, 0.0]).unwrap();
        close(grad[0], 4.0, 1e-12);
        close(grad[1], 2.0, 1e-12);
        let tangent = kent.tangent_gradient(&[0.0, 1.0, 0.0]).unwrap();
        close(tangent[0], 4.0, 1e-12);
        close(tangent[1], 0.0, 1e-12);
        assert!(
            KentPotential::new([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 4.0, 1.0)
                .is_err()
        );
    }

    #[test]
    fn kent_vmf_special_case_has_closed_form_normalizer() {
        let kent = KentPotential::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 4.0, 0.0)
            .unwrap();
        assert!(kent.is_von_mises_fisher());
        let vmf = kent.as_von_mises_fisher::<UnitVector>().unwrap();
        let x = UnitVector::from_coords(&[0.0, 1.0, 0.0]);
        close(
            kent.ln_pdf_closed_form(&[0.0, 1.0, 0.0]).unwrap(),
            vmf.ln_pdf_point(&x),
            1e-12,
        );
        close(
            kent.ln_partition_closed_form().unwrap(),
            ln_vmf_s2_partition(4.0),
            1e-12,
        );

        let uniform =
            KentPotential::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 0.0, 0.0)
                .unwrap();
        close(
            uniform.pdf_closed_form(&[0.0, 1.0, 0.0]).unwrap(),
            1.0 / (4.0 * PI),
            1e-12,
        );
        let generic =
            KentPotential::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 4.0, 1.0)
                .unwrap();
        assert!(generic.ln_partition_closed_form().is_none());
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
        close(bingham.quadratic_form(&x), 2.0, 1e-12);
        close(
            bingham.ln_unnormalized_density(&x),
            bingham.ln_unnormalized_density(&anti),
            1e-12,
        );
        close(
            bingham.ln_density_difference(&x, &[0.0, 1.0, 0.0]),
            3.0,
            1e-12,
        );
        let shifted = bingham.shifted_by_identity(3.0).unwrap();
        close(
            bingham.ln_density_difference(&x, &[0.0, 1.0, 0.0]),
            shifted.ln_density_difference(&x, &[0.0, 1.0, 0.0]),
            1e-12,
        );
        close(bingham.trace_centered().trace(), 0.0, 1e-12);
        let grad = bingham.euclidean_gradient(&x).unwrap();
        close(grad[0], 4.0, 1e-12);
        close(grad[1], 0.0, 1e-12);
        let tangent = bingham.tangent_gradient(&x).unwrap();
        close(tangent[0], 0.0, 1e-12);
        close(tangent[1], 0.0, 1e-12);
        assert!(bingham.is_antipodally_symmetric());
        assert_eq!(
            bingham.ln_unnormalized_density(&[2.0, 0.0, 0.0]),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn bingham_closed_form_normalizers_cover_isotropic_and_circle_cases() {
        let isotropic = BinghamPotential::new(DenseMatrix::from_rows(&[
            vec![2.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ]))
        .unwrap();
        assert!(isotropic.is_isotropic());
        close(
            isotropic.ln_partition_closed_form().unwrap(),
            2.0 + (4.0 * PI).ln(),
            1e-12,
        );
        close(
            isotropic.pdf_closed_form(&[1.0, 0.0, 0.0]).unwrap(),
            1.0 / (4.0 * PI),
            1e-12,
        );

        let circle =
            BinghamPotential::new(DenseMatrix::from_rows(&[vec![2.0, 0.5], vec![0.5, -1.0]]))
                .unwrap();
        let radius = (1.5_f64 * 1.5 + 0.25).sqrt();
        close(
            circle.ln_partition_closed_form().unwrap(),
            0.5 + TAU.ln() + ln_i0(radius),
            1e-12,
        );
        assert!(circle.pdf_closed_form(&[1.0, 0.0]).unwrap().is_finite());

        let generic = BinghamPotential::new(DenseMatrix::from_rows(&[
            vec![2.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0],
            vec![0.0, 0.0, 0.5],
        ]))
        .unwrap();
        assert!(generic.ln_partition_closed_form().is_none());
    }
}
