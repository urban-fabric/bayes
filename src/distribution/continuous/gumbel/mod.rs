use std::f64::consts::PI;

use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::lazy;

/// Euler–Mascheroni constant.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// The Gumbel (Type-I extreme value, maximum) distribution with location `μ` and
/// scale `β > 0`, support `(−∞, ∞)`. Models the distribution of maxima.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gumbel {
    location: f64,
    scale: f64,
}

impl Gumbel {
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

impl Distribution<f64> for Gumbel {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.inverse_cdf(rng.random::<f64>())
    }
}

impl Continuous for Gumbel {
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        lazy::gumbel_log_pdf(x, self.location, self.scale)
    }
}

impl Support for Gumbel {
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Gumbel {
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        (-(-z).exp()).exp()
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        let p = p.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        self.location - self.scale * (-p.ln()).ln()
    }
}

impl Moments for Gumbel {
    fn mean(&self) -> Option<f64> {
        Some(self.location + self.scale * EULER_GAMMA)
    }
    fn variance(&self) -> Option<f64> {
        Some(PI * PI / 6.0 * self.scale * self.scale)
    }
    fn skewness(&self) -> Option<f64> {
        Some(1.139_547_099_404_649)
    }
    fn median(&self) -> Option<f64> {
        Some(self.location - self.scale * std::f64::consts::LN_2.ln())
    }
    fn mode(&self) -> Option<f64> {
        Some(self.location)
    }
    fn entropy(&self) -> Option<f64> {
        Some(self.scale.ln() + EULER_GAMMA + 1.0)
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
        let g = Gumbel::new(1.0, 2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            close(g.cdf(g.inverse_cdf(p)), p, 1e-10);
        }
        close(g.mean().unwrap(), 1.0 + 2.0 * EULER_GAMMA, 1e-12);
        close(g.mode().unwrap(), 1.0, 1e-12);
        close(g.cdf(1.0), (-1.0f64).exp(), 1e-12);
    }
}
