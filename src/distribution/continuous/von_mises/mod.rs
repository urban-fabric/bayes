use std::f64::consts::{PI, TAU};

use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, Moments, Support};
use crate::error::BayesError;
use crate::function::bessel::{i1_over_i0, ln_i0};

/// The von Mises distribution — the circular analogue of the Normal — with mean
/// direction `μ` and concentration `κ > 0`, on the circle `[−π, π)`. Density
/// `f(θ) = e^{κ cos(θ − μ)} / (2π I₀(κ))`. As `κ → 0` it approaches the uniform
/// circular distribution; large `κ` concentrates around `μ`.
///
/// Being circular, it implements density, sampling, and circular moments, but not
/// the real-line [`ContinuousCdf`](crate::ContinuousCdf) (there is no closed-form
/// circular CDF).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VonMises {
    mu: f64,
    kappa: f64,
}

impl VonMises {
    pub fn new(mu: f64, kappa: f64) -> Result<Self, BayesError> {
        if !mu.is_finite() {
            return Err(BayesError::NonFinite {
                name: "mu",
                value: mu,
            });
        }
        if !kappa.is_finite() {
            return Err(BayesError::NonFinite {
                name: "kappa",
                value: kappa,
            });
        }
        if kappa <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "kappa",
                value: kappa,
            });
        }
        Ok(Self { mu, kappa })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }
    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    /// The **circular variance** `1 − I₁(κ)/I₀(κ) ∈ [0, 1]` (0 = fully concentrated,
    /// 1 = uniform on the circle).
    pub fn circular_variance(&self) -> f64 {
        1.0 - i1_over_i0(self.kappa)
    }
}

/// Wrap an angle to the canonical circle `[−π, π)`.
fn wrap(theta: f64) -> f64 {
    (theta + PI).rem_euclid(TAU) - PI
}

impl Distribution<f64> for VonMises {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // Best & Fisher (1979) acceptance–rejection.
        let k = self.kappa;
        let a = 1.0 + (1.0 + 4.0 * k * k).sqrt();
        let b = (a - (2.0 * a).sqrt()) / (2.0 * k);
        let r = (1.0 + b * b) / (2.0 * b);
        loop {
            let u1: f64 = rng.random();
            let z = (PI * u1).cos();
            let f = ((1.0 + r * z) / (r + z)).clamp(-1.0, 1.0);
            let c = k * (r - f);
            let u2: f64 = rng.random();
            if c * (2.0 - c) - u2 > 0.0 || (c / u2).ln() + 1.0 - c >= 0.0 {
                let u3: f64 = rng.random();
                let sign = if u3 - 0.5 < 0.0 { -1.0 } else { 1.0 };
                return wrap(self.mu + sign * f.acos());
            }
        }
    }
}

impl Continuous for VonMises {
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        self.kappa * (x - self.mu).cos() - TAU.ln() - ln_i0(self.kappa)
    }
}

impl Support for VonMises {
    fn min(&self) -> f64 {
        -PI
    }
    fn max(&self) -> f64 {
        PI
    }
}

impl Moments for VonMises {
    /// The mean direction `μ`.
    fn mean(&self) -> Option<f64> {
        Some(self.mu)
    }
    fn mode(&self) -> Option<f64> {
        Some(self.mu)
    }
    /// Circular (differential) entropy `−κ I₁(κ)/I₀(κ) + ln(2π I₀(κ))`.
    fn entropy(&self) -> Option<f64> {
        Some(TAU.ln() + ln_i0(self.kappa) - self.kappa * i1_over_i0(self.kappa))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn density_peaks_at_the_mean_and_integrates_to_one() {
        let d = VonMises::new(0.5, 2.0).unwrap();
        // Mode at μ: density there exceeds the density a quarter-turn away.
        assert!(d.pdf(0.5) > d.pdf(0.5 + std::f64::consts::FRAC_PI_2));
        // Numerically integrate over the circle.
        let (mut area, n) = (0.0, 20_000);
        for i in 0..n {
            let theta = -PI + TAU * (i as f64 + 0.5) / n as f64;
            area += d.pdf(theta) * (TAU / n as f64);
        }
        close(area, 1.0, 1e-6);
    }

    #[test]
    fn circular_variance_bounds() {
        // Higher concentration ⇒ smaller circular variance.
        let tight = VonMises::new(0.0, 10.0).unwrap();
        let loose = VonMises::new(0.0, 0.5).unwrap();
        assert!(tight.circular_variance() < loose.circular_variance());
        assert!((0.0..=1.0).contains(&tight.circular_variance()));
        assert!((0.0..=1.0).contains(&loose.circular_variance()));
    }

    #[test]
    fn sampled_circular_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(23);
        let mu = 0.7;
        let d = VonMises::new(mu, 4.0).unwrap();
        let n = 60_000;
        let (mut sc, mut ss) = (0.0, 0.0);
        for _ in 0..n {
            let t = d.sample(&mut rng);
            assert!((-PI..PI).contains(&t)); // stays on the canonical circle
            sc += t.cos();
            ss += t.sin();
        }
        // The circular mean angle atan2(Σsin, Σcos) ≈ μ.
        close(ss.atan2(sc), mu, 0.02);
    }
}
