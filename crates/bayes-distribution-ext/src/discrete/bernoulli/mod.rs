use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::estimation::sample::Summary;
use crate::{Discrete, DiscreteCdf, DiscreteMassKernel, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Element, Natural, Real};

/// The **Bernoulli** distribution: one trial with success probability
/// `p ∈ [0, 1]`. Support `{0, 1}`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Bernoulli, Discrete, Moments};
///
/// let b = Bernoulli::new(0.3)?;
/// assert!((b.pmf(1_u64) - 0.3).abs() < 1e-12);
/// assert!((b.pmf(0_u64) - 0.7).abs() < 1e-12);
///
/// // The variance p(1 - p) peaks at a half, where the outcome is least
/// // predictable.
/// assert!((b.variance() - 0.21).abs() < 1e-12);
/// assert!(Bernoulli::new(0.5)?.variance() > b.variance());
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> Result<Self, BayesError> {
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            return Err(BayesError::ProbabilityOutOfRange { value: p });
        }
        Ok(Self { p })
    }

    pub fn from_probability_saturating(p: f64) -> Self {
        let p = if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
        Self { p }
    }

    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution<bool> for Bernoulli {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        if self.p <= 0.0 {
            return false;
        }
        if self.p >= 1.0 {
            return true;
        }
        rng.random::<f64>() < self.p
    }
}

impl<K: Natural> Support<K> for Bernoulli {
    fn min(&self) -> K {
        K::zero()
    }
    fn max(&self) -> K {
        K::one()
    }
}

impl<K> DiscreteCdf<K> for Bernoulli
where
    K: Natural + Compare,
    K::Real: Compare<Mask = <K as Compare>::Mask>,
{
    fn cdf(&self, k: K) -> K::Real {
        // `1 − p` at `k = 0`, else `1`.
        let is_zero = k.eq(&K::zero());
        K::Real::select(
            &is_zero,
            K::Real::one() - K::Real::scalar(self.p),
            K::Real::one(),
        )
    }
    fn inverse_cdf(&self, p: K::Real) -> K {
        // `1` once `p` clears the `k = 0` mass `1 − p`, else `0`.
        let threshold = K::Real::one() - K::Real::scalar(self.p);
        K::select(&p.gt(&threshold), K::one(), K::zero())
    }
}

impl Moments for Bernoulli {
    fn mean(&self) -> f64 {
        self.p
    }
    fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }
    fn skewness(&self) -> f64 {
        let q = 1.0 - self.p;
        (q - self.p) / (self.p * q).sqrt()
    }
    fn mode(&self) -> f64 {
        if self.p > 0.5 { 1.0 } else { 0.0 }
    }
    /// Binary entropy `H = −p ln p − (1−p) ln(1−p)`, with `0 ln 0 = 0` via `xlogy`.
    fn entropy(&self) -> f64 {
        let q = 1.0 - self.p;
        -self.p.xlogy(self.p) - q.xlogy(q)
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// `(1 − 6p(1−p)) / (p(1−p))`.
    fn kurtosis(&self) -> f64 {
        let pq = self.p * (1.0 - self.p);
        (1.0 - 6.0 * pq) / pq
    }

    fn median(&self) -> f64 {
        f64::NAN
    }
}

/// `p̂ = x̄`, the maximum likelihood estimator — after checking the sample really
/// is zeros and ones.
///
/// The check is not pedantry. Without it a sample of counts, or of `{1, 2}`
/// labels, would fit happily and return a `p` above 1 for the constructor to
/// reject with a message about a probability rather than about the data.
impl crate::Estimate for Bernoulli {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::new(data)?;
        if data.iter().any(|&x| x != 0.0 && x != 1.0) {
            return None;
        }
        Bernoulli::new(s.mean).ok()
    }
}

#[cfg(test)]
mod tests;

// Mass kernel and discrete distribution traits.
impl<N: Natural> DiscreteMassKernel<N> for Bernoulli {
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        kernels::bernoulli_log_pmf(k.to_real(), N::Real::scalar(self.p()))
    }

    fn pmf_kernel(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<N> Discrete<N> for Bernoulli
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let zero_count = N::zero();
        let one_count = N::one();
        let zero = N::Real::zero();

        if self.p() <= 0.0 {
            let valid = k.eq(&zero_count);
            return <N::Real as Compare>::select(&valid, zero, N::Real::neg_infinity());
        }

        if self.p() >= 1.0 {
            let valid = k.eq(&one_count);
            return <N::Real as Compare>::select(&valid, zero, N::Real::neg_infinity());
        }

        let is_zero = k.eq(&zero_count);
        let is_one = k.eq(&one_count);
        let valid = is_zero.or(&is_one);
        <N::Real as Compare>::select(&valid, self.ln_pmf_kernel(k), N::Real::neg_infinity())
    }
}
