use rand::Rng;
use rand::distr::Distribution;

use super::poisson::Poisson;
use crate::continuous::Gamma;
use crate::{DiscreteCdf, Moments, Support};
use bayes_error::BayesError;
use euclid::{LazyCompare, LazyNatural, LazyReal};

#[cfg(test)]
use crate::Discrete;

/// The negative binomial distribution: the number of failures before the `r`-th
/// success in independent `Bernoulli(p)` trials (`r > 0` may be real — the
/// generalized / Pólya form). Support `{0, 1, 2, …}`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NegativeBinomial {
    r: f64,
    p: f64,
}

impl NegativeBinomial {
    pub fn new(r: f64, p: f64) -> Result<Self, BayesError> {
        if !r.is_finite() {
            return Err(BayesError::NonFinite {
                name: "r",
                value: r,
            });
        }
        if r <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "r",
                value: r,
            });
        }
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return Err(BayesError::ProbabilityOutOfRange { value: p });
        }
        if p <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "p",
                value: p,
            });
        }
        Ok(Self { r, p })
    }

    pub fn successes(&self) -> f64 {
        self.r
    }
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution<u64> for NegativeBinomial {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        let rate = self.p / (1.0 - self.p);
        let lambda = match Gamma::new(self.r, rate) {
            Ok(g) => g.sample(rng),
            Err(_) => return 0,
        };
        match Poisson::new(lambda.max(f64::MIN_POSITIVE)) {
            Ok(po) => po.sample(rng),
            Err(_) => 0,
        }
    }
}

impl<K: LazyNatural> Support<K> for NegativeBinomial {
    fn min(&self) -> K {
        K::zero()
    }
    fn max(&self) -> K {
        K::from_u64(u64::MAX)
    }
}

impl<K> DiscreteCdf<K> for NegativeBinomial
where
    K: LazyNatural + LazyCompare,
    K::Real: LazyCompare<Mask = <K as LazyCompare>::Mask>,
{
    fn cdf(&self, k: K) -> K::Real {
        K::Real::scalar(self.p).beta_reg(K::Real::scalar(self.r), k.to_real() + K::Real::one())
    }
    fn inverse_cdf(&self, p: K::Real) -> K {
        // Smallest `k` with `cdf(k) ≥ p`, the host search expressed via `while_loop`.
        K::while_loop(K::zero(), |k| self.cdf(k.clone()).lt(&p), |k| k + K::one())
    }
}

impl Moments for NegativeBinomial {
    fn mean(&self) -> f64 {
        self.r * (1.0 - self.p) / self.p
    }
    fn variance(&self) -> f64 {
        self.r * (1.0 - self.p) / (self.p * self.p)
    }
    fn skewness(&self) -> f64 {
        (2.0 - self.p) / (self.r * (1.0 - self.p)).sqrt()
    }
    fn mode(&self) -> f64 {
        if self.r > 1.0 {
            ((self.r - 1.0) * (1.0 - self.p) / self.p).floor()
        } else {
            0.0
        }
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

    fn entropy(&self) -> f64 {
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
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn geometric_special_case() {
        let nb = NegativeBinomial::new(1.0, 0.3).unwrap();
        close(nb.pmf(0_u64), 0.3, 1e-12);
        close(nb.pmf(1_u64), 0.3 * 0.7, 1e-12);
        close(nb.mean(), 0.7 / 0.3, 1e-12);
    }

    #[test]
    fn pmf_cdf_agree_and_sampled_mean() {
        let nb = NegativeBinomial::new(4.0, 0.4).unwrap();
        let mut cum = 0.0;
        for k in 0..60u64 {
            cum += nb.pmf(k);
            close(nb.cdf(k), cum, 1e-7);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let n = 60_000;
        let s: u64 = (0..n).map(|_| nb.sample(&mut rng)).sum();
        close(s as f64 / n as f64, nb.mean(), 0.05);
    }
}
