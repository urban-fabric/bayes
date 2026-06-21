use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{DiscreteCdf, Moments, Support};
use bayes_error::BayesError;
use euclid::{LazyCompare, LazyNatural, LazyReal};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    pub const MAX_LAMBDA: f64 = 1.844e19;

    pub fn new(lambda: f64) -> Result<Self, BayesError> {
        if !lambda.is_finite() {
            return Err(BayesError::NonFinite {
                name: "lambda",
                value: lambda,
            });
        }

        if lambda <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "lambda",
                value: lambda,
            });
        }
        if lambda > Self::MAX_LAMBDA {
            return Err(BayesError::AboveMaximum {
                name: "lambda",
                value: lambda,
                max: Self::MAX_LAMBDA,
            });
        }
        Ok(Self { lambda })
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Distribution<u64> for Poisson {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        if self.lambda < 30.0 {
            knuth(rng, self.lambda)
        } else {
            atkinson_pa(rng, self.lambda)
        }
    }
}

fn knuth<R: Rng + ?Sized>(rng: &mut R, lambda: f64) -> u64 {
    let limit = (-lambda).exp();
    let mut count: u64 = 0;
    let mut product: f64 = rng.random::<f64>();
    while product >= limit {
        count += 1;
        product *= rng.random::<f64>();
    }
    count
}

fn atkinson_pa<R: Rng + ?Sized>(rng: &mut R, lambda: f64) -> u64 {
    let c = 0.767 - 3.36 / lambda;
    let beta = std::f64::consts::PI / (3.0 * lambda).sqrt();
    let alpha = beta * lambda;
    let k = c.ln() - lambda - beta.ln();

    loop {
        let u: f64 = rng.random();

        if u <= 0.0 || u >= 1.0 {
            continue;
        }
        let x = (alpha - ((1.0 - u) / u).ln()) / beta;
        let n_floor = (x + 0.5).floor();
        if n_floor < 0.0 {
            continue;
        }
        let n = n_floor as u64;

        let v: f64 = rng.random();
        let y = alpha - beta * x;
        let temp = 1.0 + y.exp();
        let lhs = y + (v / (temp * temp)).ln();
        let rhs = k + (n as f64) * lambda.ln() - n.ln_factorial();
        if lhs <= rhs {
            return n;
        }
    }
}

impl<K: LazyNatural> Support<K> for Poisson {
    fn min(&self) -> K {
        K::zero()
    }
    fn max(&self) -> K {
        K::from_u64(u64::MAX)
    }
}

impl<K> DiscreteCdf<K> for Poisson
where
    K: LazyNatural + LazyCompare,
    K::Real: LazyCompare<Mask = <K as LazyCompare>::Mask>,
{
    fn cdf(&self, k: K) -> K::Real {
        K::Real::scalar(self.lambda).gamma_ur(k.to_real() + K::Real::one())
    }
    fn inverse_cdf(&self, p: K::Real) -> K {
        // Smallest `k` with `cdf(k) ≥ p`. The host integer search is expressed through
        // `while_loop`, so the same routine runs on a tensor of counts; `cdf(0) > 0 ≥ p`
        // covers `p ≤ 0` (the loop never runs).
        K::while_loop(K::zero(), |k| self.cdf(k.clone()).lt(&p), |k| k + K::one())
    }
}

impl Moments for Poisson {
    fn mean(&self) -> f64 {
        self.lambda
    }
    fn variance(&self) -> f64 {
        self.lambda
    }
    fn skewness(&self) -> f64 {
        self.lambda.sqrt().recip()
    }
    fn kurtosis(&self) -> f64 {
        self.lambda.recip()
    }
    fn mode(&self) -> f64 {
        self.lambda.floor()
    }
    fn moment2(&self) -> f64 {
        let mean = self.mean();
        self.variance() + mean * mean
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        f64::NAN
    }

    fn median(&self) -> f64 {
        f64::NAN
    }
}

#[cfg(test)]
mod tests;
