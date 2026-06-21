use std::f64::consts::{PI, SQRT_2};

use rand::Rng;
use rand::distr::Distribution;

use super::normal::standard_normal_polar;
use crate::{ContinuousCdf, Moments, Support};
use bayes_error::BayesError;
use euclid::LazyReal;

#[cfg(test)]
use crate::Continuous;

/// The half-normal distribution with scale `σ > 0`, support `[0, ∞)` — the
/// distribution of `|Z|` for `Z ~ N(0, σ²)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfNormal {
    scale: f64,
}

impl HalfNormal {
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

impl Distribution<f64> for HalfNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        (self.scale * standard_normal_polar(rng)).abs()
    }
}

impl Support for HalfNormal {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for HalfNormal {
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            (x / (self.scale * SQRT_2)).erf()
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.scale * SQRT_2 * p.clamp(0.0, 1.0).erf_inv()
    }
}

impl Moments for HalfNormal {
    fn mean(&self) -> Option<f64> {
        Some(self.scale * (2.0 / PI).sqrt())
    }
    fn variance(&self) -> Option<f64> {
        Some(self.scale * self.scale * (1.0 - 2.0 / PI))
    }
    fn median(&self) -> Option<f64> {
        Some(self.scale * SQRT_2 * 0.5_f64.erf_inv())
    }
    fn mode(&self) -> Option<f64> {
        Some(0.0)
    }
    fn entropy(&self) -> Option<f64> {
        Some(0.5 * (PI * self.scale * self.scale / 2.0).ln() + 0.5)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn density_is_twice_the_normal_on_the_positive_axis() {
        let h = HalfNormal::new(1.5).unwrap();
        let normal_pdf = |x: f64| (-0.5 * (x / 1.5).powi(2)).exp() / (1.5 * TAU.sqrt());
        close(h.pdf(2.0), 2.0 * normal_pdf(2.0), 1e-12);
        assert_eq!(h.pdf(-1.0), 0.0);
    }

    #[test]
    fn cdf_quantile_and_moments() {
        let h = HalfNormal::new(2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            close(h.cdf(h.inverse_cdf(p)), p, 1e-9);
        }
        close(h.mean().unwrap(), 2.0 * (2.0 / PI).sqrt(), 1e-12);
    }
}
