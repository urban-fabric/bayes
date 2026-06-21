use std::f64::consts::TAU;

use rand::Rng;
use rand::distr::Distribution;

use super::normal::Normal;
use crate::{ContinuousCdf, Moments, Support};
use bayes_error::BayesError;
use bayes_kernels as kernels;
use euclid::LazyReal;

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

impl<T: LazyReal> Support<T> for LogNormal {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: LazyReal> ContinuousCdf<T> for LogNormal {
    fn cdf(&self, x: T) -> T {
        kernels::lognormal_cdf(x, T::scalar(self.mu()), T::scalar(self.sigma()))
    }
    fn inverse_cdf(&self, p: T) -> T {
        self.underlying.inverse_cdf(p).exp()
    }
}

impl Moments for LogNormal {
    fn mean(&self) -> f64 {
        let (mu, sigma) = (self.mu(), self.sigma());
        (mu + sigma * sigma / 2.0).exp()
    }
    fn variance(&self) -> f64 {
        let (mu, sigma) = (self.mu(), self.sigma());
        let s2 = sigma * sigma;
        (s2.exp() - 1.0) * (2.0 * mu + s2).exp()
    }
    fn entropy(&self) -> f64 {
        let (mu, sigma) = (self.mu(), self.sigma());
        mu + 0.5 * (TAU * std::f64::consts::E * sigma * sigma).ln()
    }
    fn median(&self) -> f64 {
        self.mu().exp()
    }
    fn mode(&self) -> f64 {
        let (mu, sigma) = (self.mu(), self.sigma());
        (mu - sigma * sigma).exp()
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
mod tests;
