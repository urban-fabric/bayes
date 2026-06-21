use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Discrete, DiscreteCdf, Moments, Support};
use bayes_error::BayesError;
use euclid::{LazyBool, LazyCompare, LazyNatural, LazyReal};

/// The binomial distribution: the number of successes in `n` independent trials,
/// each with success probability `p ∈ [0, 1]`. Support `{0, …, n}`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binomial {
    n: u64,
    p: f64,
}

impl Binomial {
    pub fn new(n: u64, p: f64) -> Result<Self, BayesError> {
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return Err(BayesError::ProbabilityOutOfRange { value: p });
        }
        Ok(Self { n, p })
    }

    pub fn trials(&self) -> u64 {
        self.n
    }
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution<u64> for Binomial {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        (0..self.n).filter(|_| rng.random::<f64>() < self.p).count() as u64
    }
}

impl<K: LazyNatural> Support<K> for Binomial {
    fn min(&self) -> K {
        K::zero()
    }
    fn max(&self) -> K {
        K::from_u64(self.n)
    }
}

impl<K> DiscreteCdf<K> for Binomial
where
    K: LazyNatural + LazyCompare,
    K::Real: LazyCompare<Mask = <K as LazyCompare>::Mask>,
{
    fn cdf(&self, k: K) -> K::Real {
        // `1` at or above `n`, else the regularized incomplete beta `I_{1−p}(n−k, k+1)`.
        let at_or_above = k.ge(&K::from_u64(self.n));
        let one = K::Real::one();
        let n = K::Real::scalar(self.n as f64);
        let kr = k.to_real();
        let interior =
            (one.clone() - K::Real::scalar(self.p)).beta_reg(n - kr.clone(), kr + one.clone());
        K::Real::select(&at_or_above, one, interior)
    }
    fn inverse_cdf(&self, p: K::Real) -> K {
        // Smallest `k ≤ n` with `cdf(k) ≥ p`, the bounded host search via `while_loop`.
        let n = K::from_u64(self.n);
        K::while_loop(
            K::zero(),
            |k| k.lt(&n).and(&self.cdf(k.clone()).lt(&p)),
            |k| k + K::one(),
        )
    }
}

impl Moments for Binomial {
    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }
    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
    fn skewness(&self) -> f64 {
        let v = self.n as f64 * self.p * (1.0 - self.p);
        (1.0 - 2.0 * self.p) / v.sqrt()
    }
    fn mode(&self) -> f64 {
        ((self.n as f64 + 1.0) * self.p).floor()
    }
    /// `H = −Σ_{k=0}^{n} P(k) ln P(k)`, the exact finite-support sum (no
    /// elementary closed form exists). Zero-probability outcomes are skipped.
    fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for k in 0..=self.n {
            let lp = self.ln_pmf(k);
            if lp.is_finite() {
                h -= lp.exp() * lp;
            }
        }
        h
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
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn pmf_sums_to_one_and_cdf_agrees() {
        let b = Binomial::new(10, 0.3).unwrap();
        let mut cum = 0.0;
        for k in 0..=10u64 {
            cum += b.pmf(k);
            close(b.cdf(k), cum, 1e-9);
        }
        close(cum, 1.0, 1e-10);
        close(b.mean(), 3.0, 1e-12);
        close(b.variance(), 10.0 * 0.3 * 0.7, 1e-12);
    }

    #[test]
    fn pmf_landmark() {
        let b = Binomial::new(5, 0.4).unwrap();
        close(b.pmf(0_u64), 0.6f64.powi(5), 1e-12);
        close(b.pmf(5_u64), 0.4f64.powi(5), 1e-12);
    }
}
