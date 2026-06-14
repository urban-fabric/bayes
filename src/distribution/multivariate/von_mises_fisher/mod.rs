//! The **von Mises–Fisher** distribution on the 2-sphere S² — the directional
//! analogue of the Gaussian, the maximum-entropy distribution on the sphere for a
//! fixed mean direction. A point `x ∈ S²` has density
//!
//! ```text
//! p(x) = C₃(κ) · exp(κ · μ·x),   C₃(κ) = κ / (4π · sinh κ),
//! ```
//!
//! concentrated about the **mean direction** `μ ∈ S²` with **concentration** `κ ≥ 0`
//! (`κ = 0` is uniform on the sphere; larger `κ` is tighter). It is the spherical
//! counterpart of [`VonMises`](crate::VonMises) on the circle.
//!
//! The distribution is written over the sphere **as a space**: it speaks
//! `hausdorff`'s [`S2Point`] directly (its support is the unit sphere, its sufficient
//! statistic is the ambient resultant vector, sampling uses a tangent frame). This is
//! the concrete S² case of the manifold-distribution pattern the higher-dimensional
//! spheres will later share.

use std::f64::consts::TAU;

use hausdorff::space::r3::Vector;
use hausdorff::space::s2::S2Point;
use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::multivariate::MultivariateContinuous;
use crate::error::BayesError;

/// Largest mean-resultant length admitted by [`VonMisesFisher::fit`] before the
/// concentration estimate is treated as effectively infinite — guards the
/// `1 − R̄²` denominator when every sample points the same way.
const MAX_RESULTANT: f64 = 1.0 - 1e-10;

/// A von Mises–Fisher distribution on S²: mean direction `μ` and concentration `κ ≥ 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VonMisesFisher {
    mu: S2Point,
    kappa: f64,
}

/// `ln C₃(κ)` — the log normalising constant of the S² von Mises–Fisher density,
/// `C₃(κ) = κ / (4π sinh κ)`. Written as `ln κ − ln(2π) − κ − ln(1 − e^{−2κ})`, which
/// is stable for all `κ > 0` (large `κ`: the `e^{−2κ}` term vanishes; small `κ`:
/// `ln(1 − e^{−2κ}) → ln(2κ)`, recovering the uniform `−ln(4π)`). `κ = 0` is the
/// uniform density `1/(4π)`.
fn ln_c3(kappa: f64) -> f64 {
    if kappa == 0.0 {
        return -(2.0 * TAU).ln(); // ln(1 / 4π)
    }
    let m = (-2.0 * kappa).exp();
    kappa.ln() - TAU.ln() - kappa - (-m).ln_1p()
}

impl VonMisesFisher {
    /// A von Mises–Fisher distribution with mean direction `mu` and concentration
    /// `kappa ≥ 0` (`0` is uniform on the sphere).
    ///
    /// # Errors
    /// [`NonFinite`](BayesError::NonFinite) if `kappa` is not finite and
    /// [`NotPositive`](BayesError::NotPositive) if `kappa < 0`.
    pub fn new(mu: S2Point, kappa: f64) -> Result<Self, BayesError> {
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

    /// The mean direction `μ ∈ S²`.
    #[must_use]
    pub fn mean_direction(&self) -> S2Point {
        self.mu
    }

    /// The concentration `κ`.
    #[must_use]
    pub fn concentration(&self) -> f64 {
        self.kappa
    }

    /// The **mean resultant length** `R̄ = E[μ·X] = A₃(κ) = coth κ − 1/κ ∈ [0, 1)` —
    /// the population analogue of the sample resultant [`fit`](Self::fit) inverts.
    #[must_use]
    pub fn mean_resultant_length(&self) -> f64 {
        if self.kappa == 0.0 {
            0.0
        } else {
            1.0 / self.kappa.tanh() - 1.0 / self.kappa
        }
    }

    /// The log density at a sphere point `x`: `ln C₃(κ) + κ·μ·x`.
    #[must_use]
    pub fn ln_pdf_point(&self, x: S2Point) -> f64 {
        ln_c3(self.kappa) + self.kappa * self.mu.dot(x)
    }

    /// The density at a sphere point `x`.
    #[must_use]
    pub fn pdf_point(&self, x: S2Point) -> f64 {
        self.ln_pdf_point(x).exp()
    }

    /// **Maximum-likelihood fit** to observed directions. The mean direction is the
    /// normalised resultant `μ̂ = (Σ xᵢ)/‖Σ xᵢ‖` and the concentration uses Banerjee
    /// et al.'s closed-form estimate `κ̂ = R̄(3 − R̄²)/(1 − R̄²)` (`R̄ = ‖Σ xᵢ‖/n`), the
    /// standard approximation for the d = 3 sphere.
    ///
    /// # Errors
    /// [`Malformed`](BayesError::Malformed) if `samples` is empty.
    pub fn fit(samples: &[S2Point]) -> Result<Self, BayesError> {
        if samples.is_empty() {
            return Err(BayesError::Malformed {
                what: "von Mises–Fisher fit needs at least one sample",
            });
        }
        Self::fit_weighted(samples, &vec![1.0; samples.len()])
    }

    /// **Weighted** maximum-likelihood fit — the responsibility-weighted MLE the
    /// M-step of a mixture model needs. `μ̂ = (Σ wᵢxᵢ)/‖Σ wᵢxᵢ‖`, `R̄ = ‖Σ wᵢxᵢ‖/Σwᵢ`,
    /// `κ̂` by the same Banerjee estimate as [`fit`](Self::fit). Equal weights recover
    /// the unweighted fit.
    ///
    /// # Errors
    /// [`Malformed`](BayesError::Malformed) if the lengths differ or the total weight
    /// is not positive.
    pub fn fit_weighted(samples: &[S2Point], weights: &[f64]) -> Result<Self, BayesError> {
        if samples.is_empty() || samples.len() != weights.len() {
            return Err(BayesError::Malformed {
                what: "von Mises–Fisher weighted fit needs matching non-empty samples and weights",
            });
        }
        let total: f64 = weights.iter().sum();
        if !(total > 0.0) {
            return Err(BayesError::Malformed {
                what: "von Mises–Fisher weighted fit needs positive total weight",
            });
        }
        let resultant = samples
            .iter()
            .zip(weights)
            .fold(Vector::new(0.0, 0.0, 0.0), |acc, (s, &w)| {
                acc.add(s.as_vector().mul(w))
            });
        let r = resultant.norm();
        // A vanishing resultant carries no direction; fall back to the first sample
        // (κ̂ = 0 makes the choice immaterial — the fit is the uniform distribution).
        let mu = if r > 0.0 {
            S2Point::from_direction(resultant)
        } else {
            samples[0]
        };
        let r_bar = (r / total).clamp(0.0, MAX_RESULTANT);
        let kappa = r_bar * (3.0 - r_bar * r_bar) / (1.0 - r_bar * r_bar);
        Self::new(mu, kappa)
    }

    /// Draw a direction on the sphere (Ulrich's S² algorithm): sample the cosine
    /// `w = μ·x` from its `∝ e^{κw}` marginal by inverse-CDF, a uniform azimuth in the
    /// tangent plane, and combine `x = w·μ + √(1−w²)·(tangent)`.
    fn sample_point<R: Rng + ?Sized>(&self, rng: &mut R) -> S2Point {
        let k = self.kappa;
        let w = if k == 0.0 {
            2.0 * rng.random::<f64>() - 1.0
        } else {
            let u: f64 = rng.random();
            1.0 + (u + (1.0 - u) * (-2.0 * k).exp()).ln() / k
        };
        let theta = TAU * rng.random::<f64>();
        let [e1, e2] = self.mu.tangent_basis();
        let tangent = e1.mul(theta.cos()).add(e2.mul(theta.sin()));
        let s = (1.0 - w * w).max(0.0).sqrt();
        S2Point::from_direction(self.mu.as_vector().mul(w).add(tangent.mul(s)))
    }
}

impl Distribution<S2Point> for VonMisesFisher {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> S2Point {
        self.sample_point(rng)
    }
}

impl Distribution<Vec<f64>> for VonMisesFisher {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let v = self.sample_point(rng).as_vector();
        vec![v.x, v.y, v.z]
    }
}

impl MultivariateContinuous for VonMisesFisher {
    fn dim(&self) -> usize {
        3
    }

    /// The density of the **direction** of `x` (a length-3 ambient vector, projected
    /// onto the sphere); non-unit `x` is read as its direction. Non-finite or
    /// wrong-length input is `−∞`.
    fn ln_pdf(&self, x: &[f64]) -> f64 {
        if x.len() != 3 || x.iter().any(|v| !v.is_finite()) {
            return f64::NEG_INFINITY;
        }
        let v = Vector::new(x[0], x[1], x[2]);
        if v.norm() == 0.0 {
            return f64::NEG_INFINITY;
        }
        self.ln_pdf_point(S2Point::from_direction(v))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use hausdorff::GeoPoint;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn pt(lat: f64, lon: f64) -> S2Point {
        S2Point::from_geo_point(GeoPoint::from_lat_lon(lat, lon))
    }

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn density_integrates_to_one_on_the_sphere() {
        // Monte-Carlo ∫ p dΩ over the sphere = E_uniform[p]·(4π) ≈ 1.
        let vmf = VonMisesFisher::new(pt(10.0, 20.0), 3.0).unwrap();
        let uniform = VonMisesFisher::new(pt(0.0, 0.0), 0.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let n = 200_000;
        let mean_p: f64 = (0..n)
            .map(|_| {
                let x: S2Point = uniform.sample(&mut rng);
                vmf.pdf_point(x)
            })
            .sum::<f64>()
            / f64::from(n);
        close(mean_p * 2.0 * TAU, 1.0, 0.02); // ×4π
    }

    #[test]
    fn density_peaks_at_the_mean_direction() {
        let mu = pt(35.0, -80.0);
        let vmf = VonMisesFisher::new(mu, 5.0).unwrap();
        let here = vmf.ln_pdf_point(mu);
        let away = vmf.ln_pdf_point(pt(-35.0, 100.0)); // ~antipodal
        assert!(here > away);
        // Uniform (κ = 0) is flat: ln p = −ln(4π) everywhere.
        let uni = VonMisesFisher::new(mu, 0.0).unwrap();
        close(uni.ln_pdf_point(pt(12.0, 34.0)), -(2.0 * TAU).ln(), 1e-12);
    }

    #[test]
    fn fit_recovers_mean_direction_and_concentration() {
        let mu = pt(40.0, -100.0);
        let vmf = VonMisesFisher::new(mu, 8.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let samples: Vec<S2Point> = (0..5_000).map(|_| vmf.sample(&mut rng)).collect();
        let fitted = VonMisesFisher::fit(&samples).unwrap();
        // Mean direction within ~2° of the truth.
        assert!(fitted.mean_direction().dot(mu) > 0.999, "μ̂ off");
        // Concentration in the right neighbourhood (Banerjee approximation).
        close(fitted.concentration(), 8.0, 0.15);
    }

    #[test]
    fn samples_concentrate_with_kappa() {
        let mu = pt(0.0, 0.0);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mean_dot = |kappa: f64, rng: &mut ChaCha8Rng| {
            let vmf = VonMisesFisher::new(mu, kappa).unwrap();
            let n = 3_000;
            (0..n)
                .map(|_| {
                    let x: S2Point = vmf.sample(rng);
                    mu.dot(x)
                })
                .sum::<f64>()
                / f64::from(n)
        };
        let loose = mean_dot(1.0, &mut rng);
        let tight = mean_dot(20.0, &mut rng);
        assert!(tight > loose && tight > 0.9, "loose {loose}, tight {tight}");
        // The empirical resultant tracks A₃(κ).
        close(
            tight,
            VonMisesFisher::new(mu, 20.0)
                .unwrap()
                .mean_resultant_length(),
            0.03,
        );
    }

    #[test]
    fn rejects_negative_or_nonfinite_kappa() {
        assert!(VonMisesFisher::new(pt(0.0, 0.0), -1.0).is_err());
        assert!(VonMisesFisher::new(pt(0.0, 0.0), f64::NAN).is_err());
        assert!(VonMisesFisher::fit(&[]).is_err());
    }
}
