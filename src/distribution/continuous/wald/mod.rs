use std::f64::consts::{SQRT_2, TAU};

use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::normal::standard_normal_polar;
use crate::distribution::numeric::inverse_cdf_bisect;
use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::function::erf::erfc;
use crate::function::stable::erfcx;

/// The Wald (a.k.a. **inverse Gaussian**) distribution with mean `μ > 0` and shape
/// `λ > 0`, support `(0, ∞)` — the first-passage-time law of Brownian motion with
/// drift; common in reliability and response-time modelling. Density
/// `f(x) = √(λ / 2πx³) · exp(−λ(x−μ)² / 2μ²x)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wald {
    mu: f64,
    shape: f64,
}

impl Wald {
    pub fn new(mu: f64, shape: f64) -> Result<Self, BayesError> {
        for (name, v) in [("mu", mu), ("shape", shape)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        Ok(Self { mu, shape })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }
    pub fn shape(&self) -> f64 {
        self.shape
    }
}

impl Distribution<f64> for Wald {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // Michael–Schucany–Haas: transform a χ²₁ draw, then choose the root.
        let (mu, lam) = (self.mu, self.shape);
        let nu = standard_normal_polar(rng);
        let y = nu * nu;
        let x = mu + (mu * mu * y) / (2.0 * lam)
            - (mu / (2.0 * lam)) * (4.0 * mu * lam * y + mu * mu * y * y).sqrt();
        let u: f64 = rng.random();
        if u <= mu / (mu + x) { x } else { mu * mu / x }
    }
}

impl Continuous for Wald {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let d = x - self.mu;
        0.5 * (self.shape.ln() - TAU.ln() - 3.0 * x.ln())
            - self.shape * d * d / (2.0 * self.mu * self.mu * x)
    }
}

impl Support for Wald {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Wald {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (mu, lam) = (self.mu, self.shape);
        let s = (lam / x).sqrt();
        let a = s * (x / mu - 1.0);
        let b = s * (x / mu + 1.0);
        // Φ(a) + e^{2λ/μ} Φ(−b), the second term via erfcx so e^{2λ/μ} never
        // overflows: e^{2λ/μ} · ½ erfc(b/√2) = ½ erfcx(b/√2) · e^{2λ/μ − b²/2}.
        let phi_a = 0.5 * erfc(-a / SQRT_2);
        let term2 = 0.5 * erfcx(b / SQRT_2) * (2.0 * lam / mu - 0.5 * b * b).exp();
        (phi_a + term2).clamp(0.0, 1.0)
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        inverse_cdf_bisect(|x| self.cdf(x), 0.0, f64::INFINITY, p)
    }
}

impl Moments for Wald {
    fn mean(&self) -> Option<f64> {
        Some(self.mu)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.mu * self.mu * self.mu / self.shape)
    }
    fn skewness(&self) -> Option<f64> {
        Some(3.0 * (self.mu / self.shape).sqrt())
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(15.0 * self.mu / self.shape) // excess
    }
    fn mode(&self) -> Option<f64> {
        let r = self.mu / self.shape;
        Some(self.mu * ((1.0 + 9.0 * r * r / 4.0).sqrt() - 1.5 * r))
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
    fn pdf_integrates_and_cdf_quantile_round_trip() {
        let d = Wald::new(1.0, 3.0).unwrap();
        // Coarse trapezoidal check that the density integrates to ≈ 1.
        let (mut area, mut x, dx) = (0.0, 1e-4, 1e-3);
        while x < 8.0 {
            area += d.pdf(x) * dx;
            x += dx;
        }
        close(area, 1.0, 1e-2);
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            close(d.cdf(d.inverse_cdf(p)), p, 1e-6);
        }
        close(d.mean().unwrap(), 1.0, 1e-12);
        close(d.variance().unwrap(), 1.0 / 3.0, 1e-12);
    }

    #[test]
    fn cdf_is_monotone_and_bounded() {
        let d = Wald::new(2.0, 1.0).unwrap();
        let mut prev = 0.0;
        for i in 1..200 {
            let c = d.cdf(i as f64 * 0.1);
            assert!((0.0..=1.0).contains(&c));
            assert!(c >= prev - 1e-12);
            prev = c;
        }
    }

    #[test]
    fn sampled_mean_and_variance_match() {
        let mut rng = ChaCha8Rng::seed_from_u64(19);
        let d = Wald::new(2.0, 5.0).unwrap();
        let n = 100_000;
        let xs: Vec<f64> = (0..n).map(|_| d.sample(&mut rng)).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        close(mean, d.mean().unwrap(), 0.02);
        let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        close(var, d.variance().unwrap(), 0.1);
    }
}
