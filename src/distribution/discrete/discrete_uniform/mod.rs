use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Discrete, DiscreteCdf, Moments, Support};
use crate::error::BayesError;
use crate::lazy;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiscreteUniform {
    lo: i64,
    hi: i64,
}

impl DiscreteUniform {
    pub fn new(lo: i64, hi: i64) -> Result<Self, BayesError> {
        if lo >= hi {
            return Err(BayesError::EmptyIntRange { lo, hi });
        }
        Ok(Self { lo, hi })
    }

    pub fn lo(&self) -> i64 {
        self.lo
    }

    pub fn hi(&self) -> i64 {
        self.hi
    }
}

impl Distribution<i64> for DiscreteUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> i64 {
        rng.random_range(self.lo..self.hi)
    }
}

/// Support is the half-open `[lo, hi)` — the values `rng.random_range(lo..hi)`
/// yields, i.e. `n = hi - lo` equally likely integers.
impl DiscreteUniform {
    fn n(&self) -> f64 {
        (self.hi - self.lo) as f64
    }
}

impl Discrete<i64> for DiscreteUniform {
    fn pmf(&self, k: i64) -> f64 {
        if k >= self.lo && k < self.hi {
            1.0 / self.n()
        } else {
            0.0
        }
    }
    fn ln_pmf(&self, k: i64) -> f64 {
        if k >= self.lo && k < self.hi {
            lazy::discrete_uniform_log_pmf_on_support(self.n())
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl Support<i64> for DiscreteUniform {
    fn min(&self) -> i64 {
        self.lo
    }
    fn max(&self) -> i64 {
        self.hi - 1
    }
}

impl DiscreteCdf<i64> for DiscreteUniform {
    fn cdf(&self, k: i64) -> f64 {
        if k < self.lo {
            0.0
        } else if k >= self.hi - 1 {
            1.0
        } else {
            ((k - self.lo + 1) as f64) / self.n()
        }
    }
    fn inverse_cdf(&self, p: f64) -> i64 {
        if p <= 0.0 {
            return self.lo;
        }
        if p >= 1.0 {
            return self.hi - 1;
        }
        self.lo + (p * self.n()).ceil() as i64 - 1
    }
}

impl Moments for DiscreteUniform {
    fn mean(&self) -> Option<f64> {
        Some((self.lo + self.hi - 1) as f64 / 2.0)
    }
    fn variance(&self) -> Option<f64> {
        let n = self.n();
        Some((n * n - 1.0) / 12.0)
    }
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
    fn median(&self) -> Option<f64> {
        Some((self.lo + self.hi - 1) as f64 / 2.0)
    }
    /// `H = ln n` over `n` equally-likely outcomes — the maximum-entropy
    /// discrete law on its support.
    fn entropy(&self) -> Option<f64> {
        Some(self.n().ln())
    }
}

#[cfg(test)]
mod tests;
