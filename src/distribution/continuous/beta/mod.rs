use rand::Rng;
use rand::distr::Distribution;

use super::gamma::sample_standard_gamma;
use crate::distribution::numeric::inverse_cdf_bisect;
use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::function::beta::{beta_reg, ln_beta};
use crate::function::gamma::digamma;

/// The beta distribution on `(0, 1)` with shape parameters `α, β > 0`. Density
/// `f(x) = x^{α-1}(1-x)^{β-1} / B(α, β)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beta {
    alpha: f64,
    beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Result<Self, BayesError> {
        for (name, v) in [("alpha", alpha), ("beta", beta)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        Ok(Self { alpha, beta })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl Distribution<f64> for Beta {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let ga = sample_standard_gamma(rng, self.alpha);
        let gb = sample_standard_gamma(rng, self.beta);
        ga / (ga + gb)
    }
}

impl Continuous for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return f64::NEG_INFINITY;
        }
        (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln()
            - ln_beta(self.alpha, self.beta)
    }
}

impl Support for Beta {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        1.0
    }
}

impl ContinuousCdf for Beta {
    fn cdf(&self, x: f64) -> f64 {
        beta_reg(self.alpha, self.beta, x.clamp(0.0, 1.0))
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        inverse_cdf_bisect(|x| self.cdf(x), 0.0, 1.0, p)
    }
}

impl Moments for Beta {
    fn mean(&self) -> Option<f64> {
        Some(self.alpha / (self.alpha + self.beta))
    }
    fn variance(&self) -> Option<f64> {
        let s = self.alpha + self.beta;
        Some(self.alpha * self.beta / (s * s * (s + 1.0)))
    }
    fn skewness(&self) -> Option<f64> {
        let s = self.alpha + self.beta;
        Some(
            2.0 * (self.beta - self.alpha) * (s + 1.0).sqrt()
                / ((s + 2.0) * (self.alpha * self.beta).sqrt()),
        )
    }
    fn mode(&self) -> Option<f64> {
        if self.alpha > 1.0 && self.beta > 1.0 {
            Some((self.alpha - 1.0) / (self.alpha + self.beta - 2.0))
        } else {
            None
        }
    }
    /// `H = ln B(α,β) − (α−1)ψ(α) − (β−1)ψ(β) + (α+β−2)ψ(α+β)`.
    fn entropy(&self) -> Option<f64> {
        let (a, b) = (self.alpha, self.beta);
        Some(
            ln_beta(a, b) - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b)
                + (a + b - 2.0) * digamma(a + b),
        )
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
    fn uniform_is_beta_one_one() {
        let b = Beta::new(1.0, 1.0).unwrap();
        close(b.cdf(0.37), 0.37, 1e-10);
        close(b.pdf(0.5), 1.0, 1e-10);
    }

    #[test]
    fn moments_and_quantile() {
        let b = Beta::new(2.0, 5.0).unwrap();
        close(b.mean().unwrap(), 2.0 / 7.0, 1e-12);
        for &p in &[0.1, 0.5, 0.9] {
            close(b.cdf(b.inverse_cdf(p)), p, 1e-8);
        }
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let b = Beta::new(2.0, 5.0).unwrap();
        let n = 60_000;
        let s: f64 = (0..n).map(|_| b.sample(&mut rng)).sum();
        close(s / n as f64, b.mean().unwrap(), 0.01);
    }
}
