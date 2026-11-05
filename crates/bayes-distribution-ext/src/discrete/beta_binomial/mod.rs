use rand::Rng;
use rand::distr::Distribution;

use crate::continuous::Beta;
use crate::discrete::Binomial;
use crate::{Discrete, DiscreteCdf, DiscreteMassKernel, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Natural, Real};

/// The **beta-binomial** distribution: a binomial whose success probability is
/// itself `Beta(α, β)`-distributed — equivalently, the marginal of `K | p ~
/// Binomial(n, p)` with `p ~ Beta(α, β)`. Support `{0, …, n}`.
///
/// It is the **posterior-predictive** of the Beta–Binomial conjugate model
/// (see [`crate::conjugate::BetaBinomialModel`]) and the canonical
/// over-dispersed alternative to the binomial.
///
/// `P(K = k) = C(n, k) · B(k + α, n − k + β) / B(α, β)`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{BetaBinomial, Binomial, Discrete, Moments};
///
/// // With both shapes 1 the mixing prior is uniform, and the result is uniform
/// // on {0, ..., n}: every count equally likely.
/// let flat = BetaBinomial::new(10, 1.0, 1.0)?;
/// assert!((flat.pmf(0_u64) - 1.0 / 11.0).abs() < 1e-12);
/// assert!((flat.pmf(7_u64) - 1.0 / 11.0).abs() < 1e-12);
///
/// // That is wildly over-dispersed next to the binomial it generalizes, which
/// // is the whole reason to reach for it.
/// assert!(flat.variance() > Binomial::new(10, 0.5)?.variance());
///
/// // Large shapes concentrate the prior and recover the binomial.
/// let tight = BetaBinomial::new(10, 1e6, 1e6)?;
/// assert!((tight.variance() - Binomial::new(10, 0.5)?.variance()).abs() < 1e-3);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaBinomial {
    n: u64,
    alpha: f64,
    beta: f64,
}

impl BetaBinomial {
    pub fn new(n: u64, alpha: f64, beta: f64) -> Result<Self, BayesError> {
        validate::positive(&[("alpha", alpha), ("beta", beta)])?;
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
    fn mean(&self) -> f64 {
        let s = self.alpha + self.beta;
        self.n as f64 * self.alpha / s
    }
    fn variance(&self) -> f64 {
        let nf = self.n as f64;
        let s = self.alpha + self.beta;
        nf * self.alpha * self.beta * (s + nf) / (s * s * (s + 1.0))
    }
    fn mode(&self) -> f64 {
        if self.alpha > 1.0 && self.beta > 1.0 {
            (self.n as f64) * (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        } else {
            f64::NAN
        }
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn skewness(&self) -> f64 {
        f64::NAN
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

// Mass kernel and discrete distribution traits.
impl<N: Natural> DiscreteMassKernel<N> for BetaBinomial {
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let n = N::from_u64(self.trials());
        let k_real = k.to_real();
        let n_real = n.to_real();
        let alpha = N::Real::scalar(self.alpha());
        let beta = N::Real::scalar(self.beta());
        let ln_beta_posterior =
            (k_real.clone() + alpha.clone()).ln_beta(n_real - k_real + beta.clone());
        kernels::beta_binomial_log_pmf(n.ln_binomial(&k), ln_beta_posterior, alpha.ln_beta(beta))
    }

    fn pmf_kernel(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<N> Discrete<N> for BetaBinomial
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let n = N::from_u64(self.trials());
        let valid = k.le(&n);
        <N::Real as Compare>::select(&valid, self.ln_pmf_kernel(k), N::Real::neg_infinity())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn reduces_to_binomial_at_large_equal_shapes() {
        let bb = BetaBinomial::new(10, 1e6, 1e6).unwrap();
        let bin = Binomial::new(10, 0.5).unwrap();
        for k in 0..=10u64 {
            assert_close!(bb.pmf(k), bin.pmf(k), 1e-3);
        }
    }

    #[test]
    fn uniform_when_shapes_are_one() {
        let bb = BetaBinomial::new(4, 1.0, 1.0).unwrap();
        for k in 0..=4u64 {
            assert_close!(bb.pmf(k), 1.0 / 5.0, 1e-12);
        }
    }

    #[test]
    fn pmf_sums_to_one_cdf_agrees_and_moments() {
        let bb = BetaBinomial::new(12, 2.0, 5.0).unwrap();
        let mut cum = 0.0;
        for k in 0..=12u64 {
            cum += bb.pmf(k);
            assert_close!(bb.cdf(k), cum, 1e-12);
        }
        assert_close!(cum, 1.0, 1e-12);
        assert_close!(bb.mean(), 12.0 * 2.0 / 7.0, 1e-12);
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let bb = BetaBinomial::new(20, 3.0, 4.0).unwrap();
        let n = 60_000;
        let s: u64 = (0..n).map(|_| bb.sample(&mut rng)).sum();
        assert_close!(s as f64 / n as f64, bb.mean(), 0.03);
    }
}
