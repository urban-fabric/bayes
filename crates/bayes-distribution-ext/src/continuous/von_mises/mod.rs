use std::f64::consts::{PI, TAU};

use crate::{Continuous, ContinuousDensityKernel};
use bayes_kernels as kernels;
use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Moments, Support};
use bayes_core_ext::{BayesError, validate};
use euclid_numeric::Real;

/// The von Mises distribution — the circular analogue of the Normal — with mean
/// direction `μ` and concentration `κ > 0`, on the circle `[−π, π)`. Density
/// `f(θ) = e^{κ cos(θ − μ)} / (2π I₀(κ))`. As `κ → 0` it approaches the uniform
/// circular distribution; large `κ` concentrates around `μ`.
///
/// Being circular, it implements density, sampling, and circular moments, but not
/// the real-line [`ContinuousCdf`](crate::ContinuousCdf) (there is no closed-form
/// circular CDF).
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Continuous, VonMises};
///
/// // The circular analogue of a normal: symmetric about mu, and most
/// // concentrated there.
/// let v = VonMises::new(0.0, 4.0)?;
/// assert!(v.pdf(0.0_f64) > v.pdf(1.0_f64));
/// assert!((v.pdf(0.7_f64) - v.pdf(-0.7_f64)).abs() < 1e-12);
///
/// // kappa is a concentration, not a spread: larger means tighter, and the
/// // circular variance runs the other way, from 0 to 1.
/// assert!(VonMises::new(0.0, 20.0)?.pdf(0.0_f64) > v.pdf(0.0_f64));
/// assert!(v.circular_variance() < VonMises::new(0.0, 0.5)?.circular_variance());
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
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
        validate::positive(&[("kappa", kappa)])?;
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
        1.0 - self.kappa.bessel_i1_over_i0()
    }
}

/// Wrap an angle to the canonical circle `[−π, π)`.
fn wrap(theta: f64) -> f64 {
    (theta + PI).rem_euclid(TAU) - PI
}

impl Distribution<f64> for VonMises {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
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

impl<T: Real> Support<T> for VonMises {
    fn min(&self) -> T {
        T::scalar(-PI)
    }
    fn max(&self) -> T {
        T::scalar(PI)
    }
}

impl Moments for VonMises {
    /// The mean direction `μ`.
    fn mean(&self) -> f64 {
        self.mu
    }
    fn mode(&self) -> f64 {
        self.mu
    }
    /// Circular (differential) entropy `−κ I₁(κ)/I₀(κ) + ln(2π I₀(κ))`.
    fn entropy(&self) -> f64 {
        TAU.ln() + self.kappa.ln_bessel_i0() - self.kappa * self.kappa.bessel_i1_over_i0()
    }
    fn variance(&self) -> f64 {
        f64::NAN
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn skewness(&self) -> f64 {
        f64::NAN
    }

    fn kurtosis(&self) -> f64 {
        f64::NAN
    }

    fn median(&self) -> f64 {
        f64::NAN
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for VonMises {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let kappa = T::scalar(self.kappa());
        let log_normalizer = T::ln_tau() + kappa.clone().ln_bessel_i0();
        kernels::von_mises_log_pdf(x, T::scalar(self.mu()), kappa, log_normalizer)
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T: Real> Continuous<T> for VonMises {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

/// The circular mean, and Fisher's approximation for the concentration.
///
/// Angles cannot be averaged arithmetically: the mean of `0.1` and `6.2`
/// radians is not `3.15`, which points the opposite way. So the observations
/// are averaged as **unit vectors**, `μ̂ = atan2(S̄, C̄)` with
/// `C̄ = Σcos xᵢ/n` and `S̄ = Σsin xᵢ/n`.
///
/// The resultant length `R̄ = √(C̄² + S̄²)` carries the concentration through
/// `R̄ = I₁(κ)/I₀(κ)`, a ratio of Bessel functions with no closed inverse. This
/// uses Fisher's piecewise rational approximation to that inverse
/// (*Statistical Analysis of Circular Data*, 1993), which keeps the fit a
/// formula rather than a solver, at the cost of a little accuracy in `κ`.
///
/// `R̄ = 0` means the directions cancel exactly and no mean direction exists;
/// `R̄ = 1` means they coincide and `κ` is unbounded. Both are refused.
impl crate::Estimate for VonMises {
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let n = data.len() as f64;
        let cosine = data.iter().map(|x| x.cos()).sum::<f64>() / n;
        let sine = data.iter().map(|x| x.sin()).sum::<f64>() / n;
        let resultant = (cosine * cosine + sine * sine).sqrt();
        if resultant <= 0.0 || resultant >= 1.0 {
            return None;
        }
        let kappa = if resultant < 0.53 {
            resultant * (2.0 + resultant * resultant * (1.0 + 5.0 * resultant * resultant / 6.0))
        } else if resultant < 0.85 {
            -0.4 + 1.39 * resultant + 0.43 / (1.0 - resultant)
        } else {
            1.0 / (resultant * (resultant * (resultant - 4.0) + 3.0))
        };
        if !kappa.is_finite() {
            return None;
        }
        VonMises::new(sine.atan2(cosine), kappa).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn density_peaks_at_the_mean_and_integrates_to_one() {
        let d = VonMises::new(0.5, 2.0).unwrap();
        assert!(d.pdf(0.5) > d.pdf(0.5 + std::f64::consts::FRAC_PI_2));
        let (mut area, n) = (0.0, 20_000);
        for i in 0..n {
            let theta = -PI + TAU * (i as f64 + 0.5) / n as f64;
            area += d.pdf(theta) * (TAU / n as f64);
        }
        assert_close!(area, 1.0, 1e-6);
    }

    #[test]
    fn circular_variance_bounds() {
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
            assert!((-PI..PI).contains(&t));
            sc += t.cos();
            ss += t.sin();
        }
        assert_close!(ss.atan2(sc), mu, 0.02);
    }
}
