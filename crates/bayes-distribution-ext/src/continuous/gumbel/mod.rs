use std::f64::consts::PI;

use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{ContinuousCdf, Moments, Support};
use bayes_error::BayesError;
use bayes_kernels as kernels;
use euclid::LazyReal;

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

impl<T: LazyReal> Support<T> for Gumbel {
    fn min(&self) -> T {
        T::neg_infinity()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: LazyReal> ContinuousCdf<T> for Gumbel {
    fn cdf(&self, x: T) -> T {
        kernels::gumbel_cdf(x, T::scalar(self.location), T::scalar(self.scale))
    }
    fn inverse_cdf(&self, p: T) -> T {
        let p = p.clamp(T::scalar(f64::MIN_POSITIVE), T::scalar(1.0 - f64::EPSILON));
        kernels::gumbel_quantile(p, T::scalar(self.location), T::scalar(self.scale))
    }
}

impl Moments for Gumbel {
    fn mean(&self) -> f64 {
        self.location + self.scale * EULER_GAMMA
    }
    fn variance(&self) -> f64 {
        PI * PI / 6.0 * self.scale * self.scale
    }
    fn skewness(&self) -> f64 {
        1.139_547_099_404_649
    }
    fn median(&self) -> f64 {
        self.location - self.scale * std::f64::consts::LN_2.ln()
    }
    fn mode(&self) -> f64 {
        self.location
    }
    fn entropy(&self) -> f64 {
        self.scale.ln() + EULER_GAMMA + 1.0
    }
    fn moment2(&self) -> f64 {
        let mean = self.mean();
        self.variance() + mean * mean
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
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
    fn cdf_quantile_and_moments() {
        let g = Gumbel::new(1.0, 2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            close(g.cdf(g.inverse_cdf(p)), p, 1e-10);
        }
        close(g.mean(), 1.0 + 2.0 * EULER_GAMMA, 1e-12);
        close(g.mode(), 1.0, 1e-12);
        close(g.cdf(1.0), (-1.0f64).exp(), 1e-12);
    }
}
