use std::f64::consts::PI;

use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Estimate, Moments, Support};
use crate::error::BayesError;

/// The Rayleigh distribution with scale `σ > 0`, support `[0, ∞)`. Density
/// `f(x) = (x/σ²) exp(−x²/(2σ²))` — the magnitude of a 2-D centered Gaussian vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rayleigh {
    scale: f64,
}

impl Rayleigh {
    pub fn new(scale: f64) -> Result<Self, BayesError> {
        if !scale.is_finite() {
            return Err(BayesError::NonFinite {
                name: "scale",
                value: scale,
            });
        }
        if scale <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "scale",
                value: scale,
            });
        }
        Ok(Self { scale })
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution<f64> for Rayleigh {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.inverse_cdf(rng.random::<f64>())
    }
}

impl Continuous for Rayleigh {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let s2 = self.scale * self.scale;
        (x / s2) * (-x * x / (2.0 * s2)).exp()
    }
}

impl Support for Rayleigh {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Rayleigh {
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-x * x / (2.0 * self.scale * self.scale)).exp()
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.scale * (-2.0 * (1.0 - p.clamp(0.0, 1.0)).ln()).sqrt()
    }
}

impl Moments for Rayleigh {
    fn mean(&self) -> Option<f64> {
        Some(self.scale * (PI / 2.0).sqrt())
    }
    fn variance(&self) -> Option<f64> {
        Some((4.0 - PI) / 2.0 * self.scale * self.scale)
    }
    fn median(&self) -> Option<f64> {
        Some(self.scale * (2.0 * std::f64::consts::LN_2).sqrt())
    }
    fn mode(&self) -> Option<f64> {
        Some(self.scale)
    }
    fn entropy(&self) -> Option<f64> {
        // 1 + ln(σ/√2) + γ/2.
        Some(1.0 + (self.scale / 2.0_f64.sqrt()).ln() + 0.577_215_664_901_532_9 / 2.0)
    }
}

impl Estimate for Rayleigh {
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 || data.iter().any(|&x| x < 0.0) {
            return None;
        }
        // MLE: σ̂² = (1/2n) Σ xᵢ².
        let s2 = data.iter().map(|&x| x * x).sum::<f64>() / (2.0 * data.len() as f64);
        Rayleigh::new(s2.sqrt()).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn cdf_quantile_and_moments() {
        let r = Rayleigh::new(2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            close(r.cdf(r.inverse_cdf(p)), p, 1e-12);
        }
        close(r.mean().unwrap(), 2.0 * (PI / 2.0).sqrt(), 1e-12);
        close(r.mode().unwrap(), 2.0, 1e-12);
        close(r.pdf(2.0), (2.0 / 4.0) * (-0.5f64).exp(), 1e-12); // peak at σ
    }
}
