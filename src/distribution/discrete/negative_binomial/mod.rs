use rand::Rng;
use rand::distr::Distribution;

use super::poisson::Poisson;
use crate::distribution::continuous::Gamma;
use crate::distribution::{Discrete, DiscreteCdf, Moments, Support};
use crate::error::BayesError;
use crate::function::beta::beta_reg;
use crate::function::factorial::ln_factorial;
use crate::function::gamma::ln_gamma;
use crate::lazy;

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

impl Discrete<u64> for NegativeBinomial {
    fn pmf(&self, k: u64) -> f64 {
        self.ln_pmf(k).exp()
    }
    fn ln_pmf(&self, k: u64) -> f64 {
        if self.p >= 1.0 {
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        let kf = k as f64;
        let ln_coefficient = ln_gamma(kf + self.r) - ln_gamma(self.r) - ln_factorial(k);
        lazy::negative_binomial_log_pmf(kf, self.r, self.p, ln_coefficient)
    }
}

impl Support<u64> for NegativeBinomial {
    fn min(&self) -> u64 {
        0
    }
    fn max(&self) -> u64 {
        u64::MAX
    }
}

impl DiscreteCdf<u64> for NegativeBinomial {
    fn cdf(&self, k: u64) -> f64 {
        beta_reg(self.r, (k + 1) as f64, self.p)
    }
    fn inverse_cdf(&self, p: f64) -> u64 {
        if p <= 0.0 {
            return 0;
        }
        let mut k = 0u64;
        while self.cdf(k) < p {
            k += 1;
        }
        k
    }
}

impl Moments for NegativeBinomial {
    fn mean(&self) -> Option<f64> {
        Some(self.r * (1.0 - self.p) / self.p)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.r * (1.0 - self.p) / (self.p * self.p))
    }
    fn skewness(&self) -> Option<f64> {
        Some((2.0 - self.p) / (self.r * (1.0 - self.p)).sqrt())
    }
    fn mode(&self) -> Option<f64> {
        if self.r > 1.0 {
            Some(((self.r - 1.0) * (1.0 - self.p) / self.p).floor())
        } else {
            Some(0.0)
        }
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
        close(nb.pmf(0), 0.3, 1e-12);
        close(nb.pmf(1), 0.3 * 0.7, 1e-12);
        close(nb.mean().unwrap(), 0.7 / 0.3, 1e-12);
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
        close(s as f64 / n as f64, nb.mean().unwrap(), 0.05);
    }
}
