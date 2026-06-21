use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Discrete, DiscreteCdf, Moments, Support};
use crate::error::BayesError;
use crate::lazy;

/// The geometric distribution: the number of independent `Bernoulli(p)` trials up
/// to and including the first success. Support `{1, 2, 3, …}`, `p ∈ (0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometric {
    p: f64,
}

impl Geometric {
    pub fn new(p: f64) -> Result<Self, BayesError> {
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return Err(BayesError::ProbabilityOutOfRange { value: p });
        }
        if p <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "p",
                value: p,
            });
        }
        Ok(Self { p })
    }

    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution<u64> for Geometric {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        if self.p >= 1.0 {
            return 1;
        }
        let u = 1.0 - rng.random::<f64>();
        (u.ln() / (1.0 - self.p).ln()).ceil().max(1.0) as u64
    }
}

impl Discrete<u64> for Geometric {
    fn pmf(&self, k: u64) -> f64 {
        if k == 0 {
            return 0.0;
        }
        (1.0 - self.p).powi((k - 1) as i32) * self.p
    }
    fn ln_pmf(&self, k: u64) -> f64 {
        if k == 0 {
            return f64::NEG_INFINITY;
        }
        if self.p >= 1.0 {
            return if k == 1 { 0.0 } else { f64::NEG_INFINITY };
        }
        lazy::geometric_log_pmf(k as i64, self.p)
    }
}

impl Support<u64> for Geometric {
    fn min(&self) -> u64 {
        1
    }
    fn max(&self) -> u64 {
        u64::MAX
    }
}

impl DiscreteCdf<u64> for Geometric {
    fn cdf(&self, k: u64) -> f64 {
        if k == 0 {
            0.0
        } else {
            1.0 - (1.0 - self.p).powi(k as i32)
        }
    }
    fn inverse_cdf(&self, p: f64) -> u64 {
        if p <= 0.0 {
            return 1;
        }
        if self.p >= 1.0 {
            return 1;
        }
        ((1.0 - p).ln() / (1.0 - self.p).ln()).ceil().max(1.0) as u64
    }
}

impl Moments for Geometric {
    fn mean(&self) -> Option<f64> {
        Some(self.p.recip())
    }
    fn variance(&self) -> Option<f64> {
        Some((1.0 - self.p) / (self.p * self.p))
    }
    fn skewness(&self) -> Option<f64> {
        Some((2.0 - self.p) / (1.0 - self.p).sqrt())
    }
    /// `H = (−(1−p) ln(1−p) − p ln p) / p`; `0` for the degenerate `p = 1`.
    fn entropy(&self) -> Option<f64> {
        let p = self.p;
        if p >= 1.0 {
            return Some(0.0);
        }
        let q = 1.0 - p;
        Some((-q * q.ln() - p * p.ln()) / p)
    }
    fn mode(&self) -> Option<f64> {
        Some(1.0)
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
    fn pmf_cdf_and_moments() {
        let g = Geometric::new(0.25).unwrap();
        let mut cum = 0.0;
        for k in 1..=50u64 {
            cum += g.pmf(k);
            close(g.cdf(k), cum, 1e-9);
        }
        close(g.mean().unwrap(), 4.0, 1e-12);
        close(g.variance().unwrap(), 0.75 / 0.0625, 1e-12);
        close(g.pmf(1), 0.25, 1e-12);
    }

    #[test]
    fn quantile_round_trips() {
        let g = Geometric::new(0.3).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            let k = g.inverse_cdf(p);
            assert!(g.cdf(k) >= p && (k == 1 || g.cdf(k - 1) < p), "p={p} k={k}");
        }
    }
}
