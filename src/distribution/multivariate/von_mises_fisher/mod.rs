//! The **von Mises–Fisher** distribution on a unit sphere — the directional analogue of
//! the Gaussian, the maximum-entropy distribution on the sphere for a fixed mean
//! direction. A point `x ∈ Sᵈ⁻¹` has density
//!
//! ```text
//! p(x) = C_d(κ) · exp(κ · μ·x),   C_d(κ) = κ^{d/2−1} / ((2π)^{d/2} I_{d/2−1}(κ)),
//! ```
//!
//! concentrated about the **mean direction** `μ` with **concentration** `κ ≥ 0` (`κ = 0`
//! is uniform on the sphere). It is the spherical counterpart of [`VonMises`] on the
//! circle.
//!
//! The distribution is generic over the **space it lives on** — any `hausdorff`
//! [`Sphere`]: the fixed `S2Point` (geographic S²) *and* the dimension-general
//! `UnitVector` (the hypersphere `Sᵈ⁻¹` that L2-normalized embeddings live on). The
//! normaliser uses the general-order Bessel `ln I_{d/2−1}` (closed-form `sinh` at
//! `d = 3`); sampling is Wood's algorithm in any dimension.
//!
//! [`VonMises`]: crate::VonMises
//! [`Sphere`]: hausdorff::space::sphere::Sphere

use std::f64::consts::{PI, TAU};

use hausdorff::space::sphere::Sphere;
use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::Beta;
use crate::distribution::continuous::normal::standard_normal_polar;
use crate::distribution::multivariate::MultivariateContinuous;
use crate::error::BayesError;
use crate::function::bessel::ln_iv;
use crate::function::gamma::ln_gamma;

/// Largest mean-resultant length admitted by [`VonMisesFisher::fit_weighted`] before the
/// concentration estimate is treated as effectively infinite — guards the `1 − R̄²`
/// denominator when every sample points the same way.
const MAX_RESULTANT: f64 = 1.0 - 1e-10;

/// A von Mises–Fisher distribution over a [`Sphere`] `S`: mean direction `μ` and
/// concentration `κ ≥ 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct VonMisesFisher<S> {
    mu: S,
    kappa: f64,
}

/// `ln C_d(κ)` — the log normaliser of the `Sᵈ⁻¹` vMF density, `C_d(κ) = κ^{d/2−1} /
/// ((2π)^{d/2} I_{d/2−1}(κ))`. `κ = 0` is the uniform density `1/area(Sᵈ⁻¹) = Γ(d/2) /
/// (2 π^{d/2})`. At `d = 3` this equals the closed form `κ/(4π sinh κ)`.
fn ln_norm_const(d: usize, kappa: f64) -> f64 {
    let df = d as f64;
    let nu = df / 2.0 - 1.0;
    if kappa == 0.0 {
        ln_gamma(df / 2.0) - 2.0_f64.ln() - (df / 2.0) * PI.ln()
    } else {
        nu * kappa.ln() - (df / 2.0) * TAU.ln() - ln_iv(nu, kappa)
    }
}

impl<S: Sphere> VonMisesFisher<S> {
    /// A von Mises–Fisher distribution with mean direction `mu` and concentration
    /// `kappa ≥ 0` (`0` is uniform on the sphere).
    ///
    /// # Errors
    /// [`NonFinite`](BayesError::NonFinite) if `kappa` is not finite and
    /// [`NotPositive`](BayesError::NotPositive) if `kappa < 0`.
    pub fn new(mu: S, kappa: f64) -> Result<Self, BayesError> {
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
        Ok(Self { mu, kappa })
    }

    /// The mean direction `μ`.
    #[must_use]
    pub fn mean_direction(&self) -> &S {
        &self.mu
    }

    /// The concentration `κ`.
    #[must_use]
    pub fn concentration(&self) -> f64 {
        self.kappa
    }

    /// The ambient dimension `d` (the sphere is `Sᵈ⁻¹`).
    #[must_use]
    pub fn ambient_dim(&self) -> usize {
        self.mu.ambient_dim()
    }

    /// The **mean resultant length** `R̄ = E[μ·X] = A_d(κ) = I_{d/2}(κ)/I_{d/2−1}(κ) ∈
    /// [0, 1)` — the population analogue of the sample resultant [`fit`](Self::fit)
    /// inverts (at `d = 3`, `coth κ − 1/κ`).
    #[must_use]
    pub fn mean_resultant_length(&self) -> f64 {
        if self.kappa == 0.0 {
            return 0.0;
        }
        let df = self.mu.ambient_dim() as f64;
        (ln_iv(df / 2.0, self.kappa) - ln_iv(df / 2.0 - 1.0, self.kappa)).exp()
    }

    /// The log density at a sphere point `x`: `ln C_d(κ) + κ·μ·x`.
    #[must_use]
    pub fn ln_pdf_point(&self, x: &S) -> f64 {
        ln_norm_const(self.mu.ambient_dim(), self.kappa) + self.kappa * self.mu.dot(x)
    }

    /// The density at a sphere point `x`.
    #[must_use]
    pub fn pdf_point(&self, x: &S) -> f64 {
        self.ln_pdf_point(x).exp()
    }

    /// **Maximum-likelihood fit** to observed directions: `μ̂ = (Σ xᵢ)/‖Σ xᵢ‖`,
    /// `κ̂ = R̄(d − R̄²)/(1 − R̄²)` (Banerjee et al.), `R̄ = ‖Σ xᵢ‖/n`.
    ///
    /// # Errors
    /// [`Malformed`](BayesError::Malformed) if `samples` is empty.
    pub fn fit(samples: &[S]) -> Result<Self, BayesError> {
        if samples.is_empty() {
            return Err(BayesError::Malformed {
                what: "von Mises–Fisher fit needs at least one sample",
            });
        }
        Self::fit_weighted(samples, &vec![1.0; samples.len()])
    }

    /// **Weighted** maximum-likelihood fit — the responsibility-weighted MLE a mixture
    /// M-step needs. Equal weights recover [`fit`](Self::fit).
    ///
    /// # Errors
    /// [`Malformed`](BayesError::Malformed) if the lengths differ or the total weight is
    /// not positive.
    pub fn fit_weighted(samples: &[S], weights: &[f64]) -> Result<Self, BayesError> {
        if samples.is_empty() || samples.len() != weights.len() {
            return Err(BayesError::Malformed {
                what: "von Mises–Fisher weighted fit needs matching non-empty samples and weights",
            });
        }
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(BayesError::Malformed {
                what: "von Mises–Fisher weighted fit needs positive total weight",
            });
        }
        let d = samples[0].ambient_dim();
        let mut resultant = vec![0.0; d];
        let mut buf = vec![0.0; d];
        for (s, &w) in samples.iter().zip(weights) {
            s.write_ambient(&mut buf);
            for (acc, &c) in resultant.iter_mut().zip(buf.iter()) {
                *acc += w * c;
            }
        }
        let r = resultant.iter().map(|v| v * v).sum::<f64>().sqrt();
        // A vanishing resultant carries no direction; fall back to the first sample
        // (κ̂ = 0 makes the choice immaterial — the fit is the uniform distribution).
        let mu = if r > 0.0 {
            S::from_ambient(&resultant)
        } else {
            samples[0].clone()
        };
        let r_bar = (r / total).clamp(0.0, MAX_RESULTANT);
        let kappa = r_bar * (d as f64 - r_bar * r_bar) / (1.0 - r_bar * r_bar);
        Self::new(mu, kappa)
    }

    /// Draw a direction on the sphere by **Wood's algorithm**: sample the cosine `w = μ·x`
    /// from its marginal `∝ (1−w²)^{(d−3)/2} e^{κw}` by rejection, a uniform direction in
    /// the tangent space (a Gaussian projected off `μ`), and combine `x = w·μ +
    /// √(1−w²)·tangent`.
    fn sample_point<R: Rng + ?Sized>(&self, rng: &mut R) -> S {
        let d = self.mu.ambient_dim();
        let mut mu = vec![0.0; d];
        self.mu.write_ambient(&mut mu);
        let w = sample_colatitude(d as f64, self.kappa, rng);
        // Uniform tangent unit vector ⊥ μ: a Gaussian with its μ-component removed.
        let mut g: Vec<f64> = (0..d).map(|_| standard_normal_polar(rng)).collect();
        let g_dot_mu: f64 = g.iter().zip(&mu).map(|(a, b)| a * b).sum();
        for (gi, mi) in g.iter_mut().zip(&mu) {
            *gi -= g_dot_mu * mi;
        }
        let t_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        let s = (1.0 - w * w).max(0.0).sqrt();
        let x: Vec<f64> = (0..d).map(|i| w * mu[i] + s * g[i] / t_norm).collect();
        S::from_ambient(&x)
    }
}

/// Sample `w = μ·x` from the vMF marginal on `Sᵈ⁻¹` (Wood 1994). Handles `κ = 0` (uniform)
/// via `b = 1`.
fn sample_colatitude<R: Rng + ?Sized>(d: f64, kappa: f64, rng: &mut R) -> f64 {
    let dm1 = d - 1.0;
    let b = (-2.0 * kappa + (4.0 * kappa * kappa + dm1 * dm1).sqrt()) / dm1;
    let x0 = (1.0 - b) / (1.0 + b);
    let c = kappa * x0 + dm1 * (1.0 - x0 * x0).ln();
    // Z ~ Beta((d−1)/2, (d−1)/2); a ≥ ½ for any real sphere (d ≥ 2).
    let Ok(beta) = Beta::new(dm1 / 2.0, dm1 / 2.0) else {
        return 1.0; // degenerate S⁰ — a point mass
    };
    loop {
        let z: f64 = beta.sample(rng);
        let w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z);
        let u: f64 = rng.random();
        if kappa * w + dm1 * (1.0 - x0 * w).ln() - c >= u.ln() {
            return w;
        }
    }
}

impl<S: Sphere> Distribution<S> for VonMisesFisher<S> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> S {
        self.sample_point(rng)
    }
}

impl<S: Sphere> MultivariateContinuous for VonMisesFisher<S> {
    fn dim(&self) -> usize {
        self.mu.ambient_dim()
    }

    /// The density of the **direction** of `x` (a length-`d` ambient vector, projected onto
    /// the sphere). Non-finite, zero, or wrong-length input is `−∞`.
    fn ln_pdf(&self, x: &[f64]) -> f64 {
        if x.len() != self.mu.ambient_dim()
            || x.iter().any(|v| !v.is_finite())
            || x.iter().all(|v| *v == 0.0)
        {
            return f64::NEG_INFINITY;
        }
        self.ln_pdf_point(&S::from_ambient(x))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use hausdorff::GeoPoint;
    use hausdorff::space::s2::S2Point;
    use hausdorff::space::sphere::UnitVector;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn pt(lat: f64, lon: f64) -> S2Point {
        S2Point::from_geo_point(GeoPoint::from_lat_lon(lat, lon))
    }

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn density_integrates_to_one_on_s2() {
        let vmf = VonMisesFisher::new(pt(10.0, 20.0), 3.0).unwrap();
        let uniform = VonMisesFisher::new(pt(0.0, 0.0), 0.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let n = 200_000;
        let mean_p: f64 = (0..n)
            .map(|_| {
                let x: S2Point = uniform.sample(&mut rng);
                vmf.pdf_point(&x)
            })
            .sum::<f64>()
            / f64::from(n);
        close(mean_p * 2.0 * TAU, 1.0, 0.02); // ×4π (area of S²)
    }

    #[test]
    fn density_peaks_at_the_mean_direction() {
        let mu = pt(35.0, -80.0);
        let vmf = VonMisesFisher::new(mu, 5.0).unwrap();
        assert!(vmf.ln_pdf_point(&mu) > vmf.ln_pdf_point(&pt(-35.0, 100.0)));
        // Uniform (κ = 0) on S² is flat at −ln(4π).
        let uni = VonMisesFisher::new(mu, 0.0).unwrap();
        close(uni.ln_pdf_point(&pt(12.0, 34.0)), -(2.0 * TAU).ln(), 1e-12);
    }

    #[test]
    fn fit_recovers_mean_direction_and_concentration_s2() {
        let mu = pt(40.0, -100.0);
        let vmf = VonMisesFisher::new(mu, 8.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let samples: Vec<S2Point> = (0..5_000).map(|_| vmf.sample(&mut rng)).collect();
        let fitted = VonMisesFisher::fit(&samples).unwrap();
        assert!(fitted.mean_direction().dot(&mu) > 0.999, "μ̂ off");
        close(fitted.concentration(), 8.0, 0.15);
    }

    #[test]
    fn high_dimensional_hypersphere_fit_and_density() {
        // A vMF on S⁹ (ambient d = 10): sample, fit, and recover μ and κ — the path that
        // serves embedding clustering, exercising the general-order Bessel normaliser.
        let d = 10;
        let mut centre = vec![0.0; d];
        centre[0] = 1.0;
        centre[3] = 1.0; // an arbitrary direction
        let mu = UnitVector::from_coords(&centre);
        let vmf = VonMisesFisher::new(mu.clone(), 25.0).unwrap();
        assert_eq!(vmf.ambient_dim(), d);

        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let samples: Vec<UnitVector> = (0..8_000).map(|_| vmf.sample(&mut rng)).collect();
        // Drawn points are unit vectors of the right dimension.
        assert!(samples[0].dim() == d && (samples[0].dot(&samples[0]) - 1.0).abs() < 1e-9);

        let fitted = VonMisesFisher::fit(&samples).unwrap();
        assert!(fitted.mean_direction().dot(&mu) > 0.99, "μ̂ off in high-d");
        close(fitted.concentration(), 25.0, 0.2);
        // Density is higher at the mean than away from it.
        let away = UnitVector::from_coords(&vec![1.0; d]);
        assert!(vmf.ln_pdf_point(&mu) > vmf.ln_pdf_point(&away));
    }

    #[test]
    fn rejects_negative_or_nonfinite_kappa() {
        assert!(VonMisesFisher::new(pt(0.0, 0.0), -1.0).is_err());
        assert!(VonMisesFisher::new(pt(0.0, 0.0), f64::NAN).is_err());
        assert!(VonMisesFisher::<S2Point>::fit(&[]).is_err());
    }
}
