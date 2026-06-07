use std::f64::consts::{E, LN_2, SQRT_2, TAU};

use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Estimate, Moments, Support};
use crate::error::BayesError;
use crate::function::erf::{erf_inv, erfc};
use crate::function::stable::{erfcx, ln_1m_exp};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normal {
    mean: f64,
    std_dev: f64,
}

impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, BayesError> {
        if !mean.is_finite() {
            return Err(BayesError::NonFinite {
                name: "mean",
                value: mean,
            });
        }
        if !std_dev.is_finite() {
            return Err(BayesError::NonFinite {
                name: "std_dev",
                value: std_dev,
            });
        }

        if std_dev <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "std_dev",
                value: std_dev,
            });
        }
        Ok(Self { mean, std_dev })
    }

    /// The distribution's parameters `(μ, σ)`. Moments are read through the
    /// [`Moments`] trait (`mean()`/`std_dev()` → `Option<f64>`); this is the
    /// infallible internal accessor for the raw parameters.
    pub(crate) fn params(&self) -> (f64, f64) {
        (self.mean, self.std_dev)
    }
}

impl Distribution<f64> for Normal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let z = standard_normal_polar(rng);
        self.mean + self.std_dev * z
    }
}

pub(crate) fn standard_normal_polar<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    loop {
        let u: f64 = 2.0 * rng.random::<f64>() - 1.0;
        let v: f64 = 2.0 * rng.random::<f64>() - 1.0;
        let s = u * u + v * v;

        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

impl Continuous for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        (-0.5 * z * z).exp() / (self.std_dev * TAU.sqrt())
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        -0.5 * z * z - (self.std_dev * TAU.sqrt()).ln()
    }
}

impl Support for Normal {
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Normal {
    fn cdf(&self, x: f64) -> f64 {
        0.5 * erfc(-(x - self.mean) / (self.std_dev * SQRT_2))
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.mean + self.std_dev * SQRT_2 * erf_inv(2.0 * p - 1.0)
    }
    /// Stable `ln Φ((x-μ)/σ)`. In the left tail (`z ≤ 0`) it uses `erfcx`, so it
    /// stays accurate where `Φ` itself underflows to `0`; in the right tail it
    /// defers to `ln(1 − sf)` via the stable `ln_sf`.
    fn ln_cdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        if z <= 0.0 {
            let t = -z / SQRT_2;
            -LN_2 + erfcx(t).ln() - t * t
        } else {
            ln_1m_exp(self.ln_sf(x))
        }
    }
    /// Stable `ln(1 − Φ((x-μ)/σ))`, the mirror of [`ln_cdf`](Self::ln_cdf).
    fn ln_sf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        if z >= 0.0 {
            let t = z / SQRT_2;
            -LN_2 + erfcx(t).ln() - t * t
        } else {
            ln_1m_exp(self.ln_cdf(x))
        }
    }
}

impl Moments for Normal {
    fn mean(&self) -> Option<f64> {
        Some(self.mean)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.std_dev * self.std_dev)
    }
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(0.0)
    }
    fn entropy(&self) -> Option<f64> {
        Some(0.5 * (TAU * E * self.std_dev * self.std_dev).ln())
    }
    fn median(&self) -> Option<f64> {
        Some(self.mean)
    }
    fn mode(&self) -> Option<f64> {
        Some(self.mean)
    }
}

impl Estimate for Normal {
    fn from_sample(data: &[f64]) -> Option<Self> {
        <Self as crate::distribution::ExponentialFamily>::fit(data)
    }
}

#[cfg(test)]
mod tests;
