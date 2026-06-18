use rand::Rng;
use rand::distr::Distribution;

use super::gamma::Gamma;
use crate::distribution::numeric::inverse_cdf_bisect;
use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::function::gamma::{digamma, gamma_ur, ln_gamma};
use crate::lazy;

/// The inverse-gamma distribution with shape `α > 0` and scale `β > 0`, support
/// `(0, ∞)` — the distribution of `1/X` for `X ~ Gamma(α, rate β)`. A common
/// conjugate prior for a variance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseGamma {
    shape: f64,
    scale: f64,
    gamma: Gamma,
}

impl InverseGamma {
    pub fn new(shape: f64, scale: f64) -> Result<Self, BayesError> {
        for (name, v) in [("shape", shape), ("scale", scale)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        Ok(Self {
            shape,
            scale,
            gamma: Gamma::new(shape, scale)?,
        })
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution<f64> for InverseGamma {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        1.0 / self.gamma.sample(rng)
    }
}

impl Continuous for InverseGamma {
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
        lazy::inverse_gamma_log_pdf_on_support(x, self.shape, self.scale, ln_gamma(self.shape))
    }
}

impl Support for InverseGamma {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for InverseGamma {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            gamma_ur(self.shape, self.scale / x)
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        inverse_cdf_bisect(|x| self.cdf(x), 0.0, f64::INFINITY, p)
    }
}

impl Moments for InverseGamma {
    fn mean(&self) -> Option<f64> {
        (self.shape > 1.0).then(|| self.scale / (self.shape - 1.0))
    }
    fn variance(&self) -> Option<f64> {
        if self.shape > 2.0 {
            let a = self.shape;
            Some(self.scale * self.scale / ((a - 1.0) * (a - 1.0) * (a - 2.0)))
        } else {
            None
        }
    }
    fn mode(&self) -> Option<f64> {
        Some(self.scale / (self.shape + 1.0))
    }
    /// `H = α + ln(β Γ(α)) − (1+α)ψ(α)`.
    fn entropy(&self) -> Option<f64> {
        let (a, b) = (self.shape, self.scale);
        Some(a + b.ln() + ln_gamma(a) - (1.0 + a) * digamma(a))
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
        let ig = InverseGamma::new(3.0, 2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            close(ig.cdf(ig.inverse_cdf(p)), p, 1e-7);
        }
        close(ig.mean().unwrap(), 2.0 / 2.0, 1e-12);
        close(ig.mode().unwrap(), 2.0 / 4.0, 1e-12);
        assert!(InverseGamma::new(1.5, 1.0).unwrap().variance().is_none());
    }

    #[test]
    fn entropy_known_value() {
        // H = α + ln(β Γ(α)) − (1+α)ψ(α); for InverseGamma(3, 2): ≈ 0.69515702.
        close(
            InverseGamma::new(3.0, 2.0).unwrap().entropy().unwrap(),
            0.695_157_02,
            1e-6,
        );
    }
}
