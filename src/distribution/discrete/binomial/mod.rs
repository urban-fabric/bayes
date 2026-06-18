use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Discrete, DiscreteCdf, Moments, Support};
use crate::error::BayesError;
use crate::function::beta::beta_reg;
use crate::function::factorial::ln_binomial;
use crate::lazy;

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

impl Discrete<u64> for Binomial {
    fn pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return 0.0;
        }
        self.ln_pmf(k).exp()
    }
    fn ln_pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return f64::NEG_INFINITY;
        }
        if self.p <= 0.0 {
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        if self.p >= 1.0 {
            return if k == self.n { 0.0 } else { f64::NEG_INFINITY };
        }
        let kf = k as f64;
        let nf = self.n as f64;
        lazy::binomial_log_pmf(kf, nf - kf, self.p, ln_binomial(self.n, k))
    }
}

impl Support<u64> for Binomial {
    fn min(&self) -> u64 {
        0
    }
    fn max(&self) -> u64 {
        self.n
    }
}

impl DiscreteCdf<u64> for Binomial {
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.n {
            return 1.0;
        }
        beta_reg((self.n - k) as f64, (k + 1) as f64, 1.0 - self.p)
    }
    fn inverse_cdf(&self, p: f64) -> u64 {
        if p <= 0.0 {
            return 0;
        }
        let mut k = 0;
        while k < self.n && self.cdf(k) < p {
            k += 1;
        }
        k
    }
}

impl Moments for Binomial {
    fn mean(&self) -> Option<f64> {
        Some(self.n as f64 * self.p)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.n as f64 * self.p * (1.0 - self.p))
    }
    fn skewness(&self) -> Option<f64> {
        let v = self.n as f64 * self.p * (1.0 - self.p);
        Some((1.0 - 2.0 * self.p) / v.sqrt())
    }
    fn mode(&self) -> Option<f64> {
        Some(((self.n as f64 + 1.0) * self.p).floor())
    }
    /// `H = −Σ_{k=0}^{n} P(k) ln P(k)`, the exact finite-support sum (no
    /// elementary closed form exists). Zero-probability outcomes are skipped.
    fn entropy(&self) -> Option<f64> {
        let mut h = 0.0;
        for k in 0..=self.n {
            let lp = self.ln_pmf(k);
            if lp.is_finite() {
                h -= lp.exp() * lp;
            }
        }
        Some(h)
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
        close(b.mean().unwrap(), 3.0, 1e-12);
        close(b.variance().unwrap(), 10.0 * 0.3 * 0.7, 1e-12);
    }

    #[test]
    fn pmf_landmark() {
        let b = Binomial::new(5, 0.4).unwrap();
        close(b.pmf(0), 0.6f64.powi(5), 1e-12);
        close(b.pmf(5), 0.4f64.powi(5), 1e-12);
    }
}
