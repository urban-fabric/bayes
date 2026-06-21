use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{DiscreteCdf, Moments, Support};
use bayes_error::BayesError;
use euclid::{LazyCompare, LazyNatural, LazyReal};

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

impl<K: LazyNatural> Support<K> for Bernoulli {
    fn min(&self) -> K {
        K::zero()
    }
    fn max(&self) -> K {
        K::one()
    }
}

impl<K> DiscreteCdf<K> for Bernoulli
where
    K: LazyNatural + LazyCompare,
    K::Real: LazyCompare<Mask = <K as LazyCompare>::Mask>,
{
    fn cdf(&self, k: K) -> K::Real {
        // `1 − p` at `k = 0`, else `1`.
        let is_zero = k.eq(&K::zero());
        K::Real::select(
            &is_zero,
            K::Real::one() - K::Real::scalar(self.p),
            K::Real::one(),
        )
    }
    fn inverse_cdf(&self, p: K::Real) -> K {
        // `1` once `p` clears the `k = 0` mass `1 − p`, else `0`.
        let threshold = K::Real::one() - K::Real::scalar(self.p);
        K::select(&p.gt(&threshold), K::one(), K::zero())
    }
}

impl Moments for Bernoulli {
    fn mean(&self) -> f64 {
        self.p
    }
    fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }
    fn skewness(&self) -> f64 {
        let q = 1.0 - self.p;
        (q - self.p) / (self.p * q).sqrt()
    }
    fn mode(&self) -> f64 {
        if self.p > 0.5 { 1.0 } else { 0.0 }
    }
    /// Binary entropy `H = −p ln p − (1−p) ln(1−p)`, with `0 ln 0 = 0` via `xlogy`.
    fn entropy(&self) -> f64 {
        let q = 1.0 - self.p;
        -self.p.xlogy(self.p) - q.xlogy(q)
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

    fn median(&self) -> f64 {
        f64::NAN
    }
}

#[cfg(test)]
mod tests;
