use rand::Rng;
use rand::distr::Distribution;

use super::poisson::Poisson;
use crate::continuous::Gamma;
use crate::estimation::sample::Summary;
use crate::{Discrete, DiscreteCdf, DiscreteMassKernel, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use euclid_numeric::{Compare, Element, Natural, Real};

/// The negative binomial distribution: the number of failures before the `r`-th
/// success in independent `Bernoulli(p)` trials (`r > 0` may be real — the
/// generalized / Pólya form). Support `{0, 1, 2, …}`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Discrete, Moments, NegativeBinomial};
///
/// // Failures before the 5th success, so the support starts at 0.
/// let nb = NegativeBinomial::new(5.0, 0.4)?;
/// assert!((nb.mean() - 7.5).abs() < 1e-12);    // r (1 - p) / p
///
/// // Always *over*-dispersed, which is exactly what separates it from the
/// // Poisson and makes it the standard remedy for overdispersed counts.
/// assert!(nb.variance() > nb.mean());
///
/// // One required success is the geometric, counted as failures.
/// assert!((NegativeBinomial::new(1.0, 0.4)?.pmf(0_u64) - 0.4).abs() < 1e-12);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NegativeBinomial {
    r: f64,
    p: f64,
}

impl NegativeBinomial {
    pub fn new(r: f64, p: f64) -> Result<Self, BayesError> {
        validate::positive(&[("r", r)])?;
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

impl<K: Natural> Support<K> for NegativeBinomial {
    fn min(&self) -> K {
        K::zero()
    }
    fn max(&self) -> K {
        K::from_u64(u64::MAX)
    }
}

impl<K> DiscreteCdf<K> for NegativeBinomial
where
    K: Natural + Compare,
    K::Real: Compare<Mask = <K as Compare>::Mask>,
{
    fn cdf(&self, k: K) -> K::Real {
        K::Real::scalar(self.p).beta_reg(K::Real::scalar(self.r), k.to_real() + K::Real::one())
    }
    fn inverse_cdf(&self, p: K::Real) -> K {
        // Smallest `k` with `cdf(k) ≥ p`, by carrier-agnostic integer bisection.
        bayes_kernels::discrete_inverse_cdf(K::zero(), K::from_u64(u64::MAX), p, |k| self.cdf(k))
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
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// `6/r + p²/(r(1−p))`; reduces to the geometric value at `r = 1`.
    fn kurtosis(&self) -> f64 {
        6.0 / self.r + self.p * self.p / (self.r * (1.0 - self.p))
    }

    fn entropy(&self) -> f64 {
        f64::NAN
    }

    fn median(&self) -> f64 {
        f64::NAN
    }
}

// Mass kernel and discrete distribution traits.
impl<N: Natural> DiscreteMassKernel<N> for NegativeBinomial {
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let r = N::Real::scalar(self.successes());
        let p = N::Real::scalar(self.p());
        let k_real = k.to_real();
        let ln_coefficient =
            (k_real.clone() + r.clone()).ln_gamma() - r.clone().ln_gamma() - k.ln_factorial();
        ln_coefficient + r * p.clone().ln() + k_real * (N::Real::one() - p).ln()
    }

    fn pmf_kernel(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<N> Discrete<N> for NegativeBinomial
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        if self.p() >= 1.0 {
            let zero_count = N::zero();
            let valid = k.eq(&zero_count);
            return <N::Real as Compare>::select(&valid, N::Real::zero(), N::Real::neg_infinity());
        }

        self.ln_pmf_kernel(k)
    }
}

/// Method of moments, on an **overdispersed** sample.
///
/// The variance-to-mean ratio is `1/p` whatever `r` happens to be, so
/// `p̂ = m/v` and then `r̂ = m²/(v − m)`.
///
/// `None` unless `v > m`. A negative binomial always has variance exceeding its
/// mean — that is exactly what distinguishes it from the Poisson it generalizes
/// — so an underdispersed sample has no fit in this family, and
/// [`Poisson`] is the one it is asking for.
impl crate::Estimate for NegativeBinomial {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::new(data)?;
        if s.min < 0.0 || data.iter().any(|x| x.fract() != 0.0) {
            return None;
        }
        if s.mean <= 0.0 || s.variance <= s.mean {
            return None;
        }
        let p = s.mean / s.variance;
        NegativeBinomial::new(s.mean * s.mean / (s.variance - s.mean), p).ok()
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
    fn geometric_special_case() {
        let nb = NegativeBinomial::new(1.0, 0.3).unwrap();
        assert_close!(nb.pmf(0_u64), 0.3, 1e-12);
        assert_close!(nb.pmf(1_u64), 0.3 * 0.7, 1e-12);
        assert_close!(nb.mean(), 0.7 / 0.3, 1e-12);
    }

    #[test]
    fn pmf_cdf_agree_and_sampled_mean() {
        let nb = NegativeBinomial::new(4.0, 0.4).unwrap();
        let mut cum = 0.0;
        for k in 0..60u64 {
            cum += nb.pmf(k);
            assert_close!(nb.cdf(k), cum, 1e-7);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let n = 60_000;
        let s: u64 = (0..n).map(|_| nb.sample(&mut rng)).sum();
        assert_close!(s as f64 / n as f64, nb.mean(), 0.05);
    }
}
