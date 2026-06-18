use std::f64::consts::PI;

use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::lazy;

/// The Cauchy (Lorentz) distribution with location `x₀` and scale `γ > 0`. It has
/// **no finite mean or variance** — the canonical example of [`Moments`] returning
/// `None`. Support `(-∞, ∞)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cauchy {
    location: f64,
    scale: f64,
}

impl Cauchy {
    pub fn new(location: f64, scale: f64) -> Result<Self, BayesError> {
        if !location.is_finite() {
            return Err(BayesError::NonFinite {
                name: "location",
                value: location,
            });
        }
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
        Ok(Self { location, scale })
    }

    pub fn location(&self) -> f64 {
        self.location
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution<f64> for Cauchy {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.random();
        self.location + self.scale * (PI * (u - 0.5)).tan()
    }
}

impl Continuous for Cauchy {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        1.0 / (PI * self.scale * (1.0 + z * z))
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        lazy::cauchy_log_pdf(x, self.location, self.scale)
    }
}

impl Support for Cauchy {
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Cauchy {
    fn cdf(&self, x: f64) -> f64 {
        0.5 + ((x - self.location) / self.scale).atan() / PI
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.location + self.scale * (PI * (p.clamp(0.0, 1.0) - 0.5)).tan()
    }
}

impl Moments for Cauchy {
    fn entropy(&self) -> Option<f64> {
        Some((4.0 * PI * self.scale).ln())
    }
    fn median(&self) -> Option<f64> {
        Some(self.location)
    }
    fn mode(&self) -> Option<f64> {
        Some(self.location)
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
    fn cdf_quantile_and_undefined_moments() {
        let c = Cauchy::new(1.0, 2.0).unwrap();
        close(c.cdf(1.0), 0.5, 1e-12);
        close(c.cdf(3.0), 0.75, 1e-12);
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            close(c.cdf(c.inverse_cdf(p)), p, 1e-10);
        }
        assert_eq!(c.mean(), None);
        assert_eq!(c.variance(), None);
        assert_eq!(c.median(), Some(1.0));
    }
}
