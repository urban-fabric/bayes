use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Discrete, DiscreteCdf, Moments, Support};
use crate::error::BayesError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> Result<Self, BayesError> {
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return Err(BayesError::ProbabilityOutOfRange { value: p });
        }
        Ok(Self { p })
    }

    pub fn from_probability_saturating(p: f64) -> Self {
        let p = if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
        Self { p }
    }

    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution<bool> for Bernoulli {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        if self.p <= 0.0 {
            return false;
        }
        if self.p >= 1.0 {
            return true;
        }
        rng.random::<f64>() < self.p
    }
}

impl Discrete<u64> for Bernoulli {
    fn pmf(&self, k: u64) -> f64 {
        match k {
            0 => 1.0 - self.p,
            1 => self.p,
            _ => 0.0,
        }
    }
}

impl Support<u64> for Bernoulli {
    fn min(&self) -> u64 {
        0
    }
    fn max(&self) -> u64 {
        1
    }
}

impl DiscreteCdf<u64> for Bernoulli {
    fn cdf(&self, k: u64) -> f64 {
        if k == 0 { 1.0 - self.p } else { 1.0 }
    }
    fn inverse_cdf(&self, p: f64) -> u64 {
        u64::from(p > 1.0 - self.p)
    }
}

impl Moments for Bernoulli {
    fn mean(&self) -> Option<f64> {
        Some(self.p)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.p * (1.0 - self.p))
    }
    fn skewness(&self) -> Option<f64> {
        let q = 1.0 - self.p;
        Some((q - self.p) / (self.p * q).sqrt())
    }
    fn mode(&self) -> Option<f64> {
        Some(if self.p > 0.5 { 1.0 } else { 0.0 })
    }
    /// Binary entropy `H = −p ln p − (1−p) ln(1−p)`, with `0 ln 0 = 0`.
    fn entropy(&self) -> Option<f64> {
        let term = |x: f64| if x > 0.0 { x * x.ln() } else { 0.0 };
        Some(-term(self.p) - term(1.0 - self.p))
    }
}

#[cfg(test)]
mod tests;
