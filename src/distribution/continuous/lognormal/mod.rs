use std::f64::consts::TAU;

use rand::Rng;
use rand::distr::Distribution;

use super::normal::Normal;
use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::lazy;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogNormal {
    underlying: Normal,
}

impl LogNormal {
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, BayesError> {
        Ok(Self {
            underlying: Normal::new(mean, std_dev)?,
        })
    }

    pub fn mu(&self) -> f64 {
        self.underlying.params().0
    }

    pub fn sigma(&self) -> f64 {
        self.underlying.params().1
    }
}

impl Distribution<f64> for LogNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.underlying.sample(rng).exp()
    }
}

impl Continuous for LogNormal {
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
        let (mu, sigma) = (self.mu(), self.sigma());
        lazy::lognormal_log_pdf_on_support(x, mu, sigma)
    }
}

impl Support for LogNormal {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for LogNormal {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.underlying.cdf(x.ln())
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.underlying.inverse_cdf(p).exp()
    }
}

impl Moments for LogNormal {
    fn mean(&self) -> Option<f64> {
        let (mu, sigma) = (self.mu(), self.sigma());
        Some((mu + sigma * sigma / 2.0).exp())
    }
    fn variance(&self) -> Option<f64> {
        let (mu, sigma) = (self.mu(), self.sigma());
        let s2 = sigma * sigma;
        Some((s2.exp() - 1.0) * (2.0 * mu + s2).exp())
    }
    fn entropy(&self) -> Option<f64> {
        let (mu, sigma) = (self.mu(), self.sigma());
        Some(mu + 0.5 * (TAU * std::f64::consts::E * sigma * sigma).ln())
    }
    fn median(&self) -> Option<f64> {
        Some(self.mu().exp())
    }
    fn mode(&self) -> Option<f64> {
        let (mu, sigma) = (self.mu(), self.sigma());
        Some((mu - sigma * sigma).exp())
    }
}

#[cfg(test)]
mod tests;
