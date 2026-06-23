use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Discrete, DiscreteCdf, DiscreteMassKernel, Moments, Support};
use bayes_core_ext::BayesError;
use euclid::{Compare, Element, Natural, Real};

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

impl<K: Natural> Support<K> for Geometric {
    fn min(&self) -> K {
        K::one()
    }
    fn max(&self) -> K {
        K::from_u64(u64::MAX)
    }
}

impl<K> DiscreteCdf<K> for Geometric
where
    K: Natural + Compare,
    K::Real: Compare<Mask = <K as Compare>::Mask>,
{
    fn cdf(&self, k: K) -> K::Real {
        // `1 − (1−p)^k = 1 − exp(k·ln(1−p))`, which is `0` at `k = 0` with no branch.
        let one = K::Real::one();
        one.clone() - (k.to_real() * (one - K::Real::scalar(self.p)).ln()).exp()
    }
    fn inverse_cdf(&self, p: K::Real) -> K {
        // Smallest `k ≥ 1` with `cdf(k) ≥ p`, by carrier-agnostic integer bisection.
        bayes_kernels::discrete_inverse_cdf(K::one(), K::from_u64(u64::MAX), p, |k| self.cdf(k))
    }
}

impl Moments for Geometric {
    fn mean(&self) -> f64 {
        self.p.recip()
    }
    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }
    fn skewness(&self) -> f64 {
        (2.0 - self.p) / (1.0 - self.p).sqrt()
    }
    /// `H = (−(1−p) ln(1−p) − p ln p) / p`; `0` for the degenerate `p = 1`.
    fn entropy(&self) -> f64 {
        let p = self.p;
        if p >= 1.0 {
            return 0.0;
        }
        let q = 1.0 - p;
        (-q * q.ln() - p * p.ln()) / p
    }
    fn mode(&self) -> f64 {
        1.0
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// `6 + p²/(1−p)` (trial-count parameterization).
    fn kurtosis(&self) -> f64 {
        6.0 + self.p * self.p / (1.0 - self.p)
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
    fn pmf_cdf_and_moments() {
        let g = Geometric::new(0.25).unwrap();
        let mut cum = 0.0;
        for k in 1..=50u64 {
            cum += g.pmf(k);
            close(g.cdf(k), cum, 1e-9);
        }
        close(g.mean(), 4.0, 1e-12);
        close(g.variance(), 0.75 / 0.0625, 1e-12);
        close(g.pmf(1_u64), 0.25, 1e-12);
    }

    #[test]
    fn quantile_round_trips() {
        let g = Geometric::new(0.3).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            let k: u64 = g.inverse_cdf(p);
            assert!(g.cdf(k) >= p && (k == 1 || g.cdf(k - 1) < p), "p={p} k={k}");
        }
    }
}

// Mass kernel and discrete distribution traits.
impl<N: Natural> DiscreteMassKernel<N> for Geometric {
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let p = N::Real::scalar(self.p());
        (k.to_real() - N::Real::one()) * (N::Real::one() - p.clone()).ln() + p.ln()
    }

    fn pmf_kernel(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<N> Discrete<N> for Geometric
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let one_count = N::one();
        let zero = N::Real::zero();

        if self.p() >= 1.0 {
            let valid = k.eq(&one_count);
            return <N::Real as Compare>::select(&valid, zero, N::Real::neg_infinity());
        }

        let valid = k.ge(&one_count);
        <N::Real as Compare>::select(&valid, self.ln_pmf_kernel(k), N::Real::neg_infinity())
    }
}
