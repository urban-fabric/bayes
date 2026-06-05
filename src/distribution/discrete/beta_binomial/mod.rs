use rand::Rng;
use rand::distr::Distribution;

use crate::distribution::continuous::Beta;
use crate::distribution::discrete::Binomial;
use crate::distribution::{Discrete, DiscreteCdf, Moments, Support};
use crate::error::BayesError;
use crate::function::beta::ln_beta;
use crate::function::factorial::ln_binomial;

/// The **beta-binomial** distribution: a binomial whose success probability is
/// itself `Beta(α, β)`-distributed — equivalently, the marginal of `K | p ~
/// Binomial(n, p)` with `p ~ Beta(α, β)`. Support `{0, …, n}`.
///
/// It is the **posterior-predictive** of the Beta–Binomial conjugate model
/// (see [`crate::distribution::conjugate::BetaBinomialModel`]) and the canonical
/// over-dispersed alternative to the binomial.
///
/// `P(K = k) = C(n, k) · B(k + α, n − k + β) / B(α, β)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaBinomial {
    n: u64,
    alpha: f64,
    beta: f64,
}

impl BetaBinomial {
    pub fn new(n: u64, alpha: f64, beta: f64) -> Result<Self, BayesError> {
        for (name, v) in [("alpha", alpha), ("beta", beta)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        Ok(Self { n, alpha, beta })
    }

    /// Build directly from a `Beta(α, β)` over the success probability and a fixed
    /// number of `trials` — the form produced by a Beta–Binomial conjugate update
    /// (infallible, since a `Beta` already guarantees positive shapes).
    pub fn from_beta(trials: u64, prior: Beta) -> Self {
        Self {
            n: trials,
            alpha: prior.alpha(),
            beta: prior.beta(),
        }
    }

    pub fn trials(&self) -> u64 {
        self.n
    }
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl Distribution<u64> for BetaBinomial {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        // Compound draw: p ~ Beta(α, β), then K ~ Binomial(n, p).
        let p = match Beta::new(self.alpha, self.beta) {
            Ok(b) => b.sample(rng),
            Err(_) => return 0,
        };
        match Binomial::new(self.n, p) {
            Ok(b) => b.sample(rng),
            Err(_) => 0,
        }
    }
}

impl Discrete<u64> for BetaBinomial {
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
        let kf = k as f64;
        let nf = self.n as f64;
        ln_binomial(self.n, k) + ln_beta(kf + self.alpha, nf - kf + self.beta)
            - ln_beta(self.alpha, self.beta)
    }
}

impl Support<u64> for BetaBinomial {
    fn min(&self) -> u64 {
        0
    }
    fn max(&self) -> u64 {
        self.n
    }
}

impl DiscreteCdf<u64> for BetaBinomial {
    fn cdf(&self, k: u64) -> f64 {
        // No simple closed form; sum the mass up to k (support is finite, ≤ n).
        if k >= self.n {
            return 1.0;
        }
        (0..=k).map(|i| self.pmf(i)).sum::<f64>().min(1.0)
    }
    fn inverse_cdf(&self, p: f64) -> u64 {
        if p <= 0.0 {
            return 0;
        }
        let mut cum = 0.0;
        for k in 0..self.n {
            cum += self.pmf(k);
            if cum >= p {
                return k;
            }
        }
        self.n
    }
}

impl Moments for BetaBinomial {
    fn mean(&self) -> Option<f64> {
        let s = self.alpha + self.beta;
        Some(self.n as f64 * self.alpha / s)
    }
    fn variance(&self) -> Option<f64> {
        let nf = self.n as f64;
        let s = self.alpha + self.beta;
        // n·α·β·(α+β+n) / [(α+β)²·(α+β+1)].
        Some(nf * self.alpha * self.beta * (s + nf) / (s * s * (s + 1.0)))
    }
    fn mode(&self) -> Option<f64> {
        // Mode of the underlying Beta scaled by n, when it has an interior mode.
        if self.alpha > 1.0 && self.beta > 1.0 {
            Some((self.n as f64) * (self.alpha - 1.0) / (self.alpha + self.beta - 2.0))
        } else {
            None
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
    fn reduces_to_binomial_at_large_equal_shapes() {
        // Beta(α, α) with huge α concentrates p at 0.5 ⇒ ~Binomial(n, 0.5).
        let bb = BetaBinomial::new(10, 1e6, 1e6).unwrap();
        let bin = Binomial::new(10, 0.5).unwrap();
        for k in 0..=10u64 {
            close(bb.pmf(k), bin.pmf(k), 1e-3);
        }
    }

    #[test]
    fn uniform_when_shapes_are_one() {
        // Beta(1,1) ⇒ K uniform on {0,…,n}: P(K=k) = 1/(n+1).
        let bb = BetaBinomial::new(4, 1.0, 1.0).unwrap();
        for k in 0..=4u64 {
            close(bb.pmf(k), 1.0 / 5.0, 1e-12);
        }
    }

    #[test]
    fn pmf_sums_to_one_cdf_agrees_and_moments() {
        let bb = BetaBinomial::new(12, 2.0, 5.0).unwrap();
        let mut cum = 0.0;
        for k in 0..=12u64 {
            cum += bb.pmf(k);
            close(bb.cdf(k), cum, 1e-12);
        }
        close(cum, 1.0, 1e-12);
        close(bb.mean().unwrap(), 12.0 * 2.0 / 7.0, 1e-12);
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let bb = BetaBinomial::new(20, 3.0, 4.0).unwrap();
        let n = 60_000;
        let s: u64 = (0..n).map(|_| bb.sample(&mut rng)).sum();
        close(s as f64 / n as f64, bb.mean().unwrap(), 0.03);
    }
}
