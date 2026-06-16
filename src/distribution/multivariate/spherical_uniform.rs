//! Uniform distribution on a unit sphere.
//!
//! This is the normalized base measure for directional distributions. It is generic over
//! any [`Sphere`](hausdorff::space::sphere::Sphere) representation because the closed-form
//! density depends only on the ambient dimension:
//! `p(x) = 1 / area(S^{d-1})`, `area(S^{d-1}) = 2π^{d/2}/Γ(d/2)`.

use hausdorff::space::sphere::Sphere;
use ramanujan::DenseMatrix;
use rand::Rng;
use rand::distr::Distribution;

use crate::distribution::continuous::normal::standard_normal_polar;
use crate::distribution::multivariate::{MultivariateContinuous, MultivariateMoments};
use crate::error::BayesError;
use crate::function::gamma::ln_gamma;

use std::f64::consts::PI;

/// Uniform law on a unit sphere `S^{d-1}`.
///
/// A representative point carries the concrete sphere type and ambient dimension; the
/// density and moments are otherwise parameter-free.
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalUniform<S> {
    representative: S,
}

impl<S: Sphere> SphericalUniform<S> {
    /// Build the uniform law on the same sphere as `representative`.
    ///
    /// # Errors
    /// [`Malformed`](BayesError::Malformed) if the ambient dimension is less than two.
    pub fn new(representative: S) -> Result<Self, BayesError> {
        if representative.ambient_dim() < 2 {
            return Err(BayesError::Malformed {
                what: "spherical uniform needs ambient dimension at least two",
            });
        }
        Ok(Self { representative })
    }

    /// A representative point on the sphere.
    #[must_use]
    pub fn representative(&self) -> &S {
        &self.representative
    }

    /// Ambient dimension `d`; the manifold is `S^{d-1}`.
    #[must_use]
    pub fn ambient_dim(&self) -> usize {
        self.representative.ambient_dim()
    }

    /// `ln area(S^{d-1})`.
    #[must_use]
    pub fn ln_surface_area(&self) -> f64 {
        ln_sphere_area(self.ambient_dim())
    }

    /// Surface area of `S^{d-1}`.
    #[must_use]
    pub fn surface_area(&self) -> f64 {
        self.ln_surface_area().exp()
    }

    /// Log density at a sphere point.
    #[must_use]
    pub fn ln_pdf_point(&self, x: &S) -> f64 {
        if x.ambient_dim() != self.ambient_dim() {
            return f64::NEG_INFINITY;
        }
        -self.ln_surface_area()
    }

    /// Density at a sphere point.
    #[must_use]
    pub fn pdf_point(&self, x: &S) -> f64 {
        self.ln_pdf_point(x).exp()
    }
}

impl<S: Sphere> Distribution<S> for SphericalUniform<S> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> S {
        let d = self.ambient_dim();
        loop {
            let coords: Vec<f64> = (0..d).map(|_| standard_normal_polar(rng)).collect();
            if coords.iter().any(|v| *v != 0.0) {
                return S::from_ambient(&coords);
            }
        }
    }
}

impl<S: Sphere> MultivariateContinuous for SphericalUniform<S> {
    fn dim(&self) -> usize {
        self.ambient_dim()
    }

    /// Density of the direction of a finite non-zero ambient vector.
    fn ln_pdf(&self, x: &[f64]) -> f64 {
        if x.len() != self.ambient_dim()
            || x.iter().any(|v| !v.is_finite())
            || x.iter().all(|v| *v == 0.0)
        {
            return f64::NEG_INFINITY;
        }
        -self.ln_surface_area()
    }
}

impl<S: Sphere> MultivariateMoments for SphericalUniform<S> {
    fn mean_vector(&self) -> Vec<f64> {
        vec![0.0; self.ambient_dim()]
    }

    fn covariance(&self) -> DenseMatrix {
        let d = self.ambient_dim();
        DenseMatrix::from_fn(d, d, |i, j| if i == j { 1.0 / d as f64 } else { 0.0 })
    }

    fn entropy(&self) -> Option<f64> {
        Some(self.ln_surface_area())
    }
}

pub(crate) fn ln_sphere_area(ambient_dimension: usize) -> f64 {
    2.0_f64.ln() + 0.5 * ambient_dimension as f64 * PI.ln()
        - ln_gamma(0.5 * ambient_dimension as f64)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use hausdorff::space::sphere::UnitVector;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn s2_density_moments_and_entropy_are_closed_form() {
        let uniform = SphericalUniform::new(UnitVector::from_coords(&[1.0, 0.0, 0.0])).unwrap();
        close(uniform.surface_area(), 4.0 * PI, 1e-12);
        close(
            uniform.pdf_point(&UnitVector::from_coords(&[0.0, 1.0, 0.0])),
            1.0 / (4.0 * PI),
            1e-12,
        );
        close(uniform.entropy().unwrap(), (4.0 * PI).ln(), 1e-12);
        close(uniform.covariance().get(0, 0), 1.0 / 3.0, 1e-12);
        close(uniform.covariance().get(0, 1), 0.0, 1e-12);
    }

    #[test]
    fn sampling_returns_unit_directions() {
        let uniform =
            SphericalUniform::new(UnitVector::from_coords(&[1.0, 0.0, 0.0, 0.0])).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let x = uniform.sample(&mut rng);
        close(x.dot(&x), 1.0, 1e-12);
        assert_eq!(x.dim(), 4);
    }
}
