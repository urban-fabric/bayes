use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Estimate, Moments, Support};
use crate::error::BayesError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniform {
    lo: f64,
    width: f64,
}

impl Uniform {
    pub const fn unit_interval() -> Self {
        Self {
            lo: 0.0,
            width: 1.0,
        }
    }

    pub fn symmetric_radius(radius: f64) -> Self {
        debug_assert!(radius.is_finite() && radius > 0.0);
        Self {
            lo: -radius,
            width: radius * 2.0,
        }
    }

    pub fn new(lo: f64, hi: f64) -> Result<Self, BayesError> {
        if !lo.is_finite() {
            return Err(BayesError::NonFinite {
                name: "lo",
                value: lo,
            });
        }
        if !hi.is_finite() {
            return Err(BayesError::NonFinite {
                name: "hi",
                value: hi,
            });
        }

        if lo >= hi {
            return Err(BayesError::EmptyRange { lo, hi });
        }
        Ok(Self { lo, width: hi - lo })
    }

    pub fn lo(&self) -> f64 {
        self.lo
    }

    pub fn hi(&self) -> f64 {
        self.lo + self.width
    }
}

impl Distribution<f64> for Uniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.lo + self.width * rng.random::<f64>()
    }
}

impl Continuous for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if x >= self.lo && x <= self.lo + self.width {
            1.0 / self.width
        } else {
            0.0
        }
    }
}

impl Support for Uniform {
    fn min(&self) -> f64 {
        self.lo
    }
    fn max(&self) -> f64 {
        self.lo + self.width
    }
}

impl ContinuousCdf for Uniform {
    fn cdf(&self, x: f64) -> f64 {
        ((x - self.lo) / self.width).clamp(0.0, 1.0)
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.lo + self.width * p.clamp(0.0, 1.0)
    }
}

impl Moments for Uniform {
    fn mean(&self) -> Option<f64> {
        Some(self.lo + self.width / 2.0)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.width * self.width / 12.0)
    }
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(-1.2) // excess kurtosis of the uniform
    }
    fn entropy(&self) -> Option<f64> {
        Some(self.width.ln())
    }
    fn median(&self) -> Option<f64> {
        Some(self.lo + self.width / 2.0)
    }
}

impl Estimate for Uniform {
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        let lo = data.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Uniform::new(lo, hi).ok()
    }
}

#[cfg(test)]
mod tests;
