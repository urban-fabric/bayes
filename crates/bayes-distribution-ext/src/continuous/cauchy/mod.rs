use std::f64::consts::PI;

use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{ContinuousCdf, Moments, Support};
use bayes_error::BayesError;
use bayes_kernels as kernels;
use euclid::LazyReal;

/// The Cauchy (Lorentz) distribution with location `x₀` and scale `γ > 0`. It has
/// **no finite mean or variance** — the canonical example of [`Moments`] returning
/// `NaN` for undefined analytic summaries. Support `(-∞, ∞)`.
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

impl<T: LazyReal> Support<T> for Cauchy {
    fn min(&self) -> T {
        T::neg_infinity()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: LazyReal> ContinuousCdf<T> for Cauchy {
    fn cdf(&self, x: T) -> T {
        kernels::cauchy_cdf(x, T::scalar(self.location), T::scalar(self.scale))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::cauchy_quantile(
            p.clamp(T::zero(), T::one()),
            T::scalar(self.location),
            T::scalar(self.scale),
        )
    }
}

impl Moments for Cauchy {
    fn entropy(&self) -> f64 {
        (4.0 * PI * self.scale).ln()
    }
    fn median(&self) -> f64 {
        self.location
    }
    fn mode(&self) -> f64 {
        self.location
    }
    fn mean(&self) -> f64 {
        f64::NAN
    }

    fn variance(&self) -> f64 {
        f64::NAN
    }

    fn moment2(&self) -> f64 {
        let mean = self.mean();
        self.variance() + mean * mean
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
        assert!(c.mean().is_nan());
        assert!(c.variance().is_nan());
        assert_eq!(c.median(), 1.0);
    }
}
