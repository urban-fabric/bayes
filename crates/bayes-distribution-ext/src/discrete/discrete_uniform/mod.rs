use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::estimation::sample::Summary;
use crate::{IntegerDiscrete, IntegerDiscreteCdf, IntegerMassKernel, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Integer, Real};

/// The **discrete uniform** distribution on the half-open integer range
/// `{lo, …, hi − 1}`, each value equally likely.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{DiscreteUniform, IntegerDiscrete, Moments};
///
/// // The range is **half-open**, so a six-sided die is `new(1, 7)`.
/// let die = DiscreteUniform::new(1, 7)?;
/// assert!((die.pmf(6_i64) - 1.0 / 6.0).abs() < 1e-12);
/// assert_eq!(die.pmf(7_i64), 0.0);
/// assert!((die.mean() - 3.5).abs() < 1e-12);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiscreteUniform {
    lo: i64,
    hi: i64,
}

impl DiscreteUniform {
    pub fn new(lo: i64, hi: i64) -> Result<Self, BayesError> {
        // `lo >= hi` is correct here, not `lo > hi`: this family's range is the half-open
        // `[lo, hi)` documented below, so `lo == hi` carries no values at all. Loosening it
        // would also hand an empty range to `rng.random_range(lo..hi)`, which panics.
        if lo >= hi {
            return Err(BayesError::EmptyIntRange { lo, hi });
        }
        Ok(Self { lo, hi })
    }

    pub fn lo(&self) -> i64 {
        self.lo
    }

    pub fn hi(&self) -> i64 {
        self.hi
    }
}

impl Distribution<i64> for DiscreteUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> i64 {
        rng.random_range(self.lo..self.hi)
    }
}

/// Support is the half-open `[lo, hi)` — the values `rng.random_range(lo..hi)`
/// yields, i.e. `n = hi - lo` equally likely integers.
impl DiscreteUniform {
    fn n(&self) -> f64 {
        (self.hi - self.lo) as f64
    }
}

impl Support<i64> for DiscreteUniform {
    fn min(&self) -> i64 {
        self.lo
    }
    fn max(&self) -> i64 {
        self.hi - 1
    }
}

impl IntegerDiscreteCdf<i64> for DiscreteUniform {
    fn cdf(&self, k: i64) -> f64 {
        if k < self.lo {
            0.0
        } else if k >= self.hi - 1 {
            1.0
        } else {
            ((k - self.lo + 1) as f64) / self.n()
        }
    }
    fn inverse_cdf(&self, p: f64) -> i64 {
        if p <= 0.0 {
            return self.lo;
        }
        if p >= 1.0 {
            return self.hi - 1;
        }
        self.lo + (p * self.n()).ceil() as i64 - 1
    }
}

impl Moments for DiscreteUniform {
    fn mean(&self) -> f64 {
        (self.lo + self.hi - 1) as f64 / 2.0
    }
    fn variance(&self) -> f64 {
        let n = self.n();
        (n * n - 1.0) / 12.0
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    fn median(&self) -> f64 {
        (self.lo + self.hi - 1) as f64 / 2.0
    }
    /// `H = ln n` over `n` equally-likely outcomes — the maximum-entropy
    /// discrete law on its support.
    fn entropy(&self) -> f64 {
        self.n().ln()
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn kurtosis(&self) -> f64 {
        f64::NAN
    }

    fn mode(&self) -> f64 {
        f64::NAN
    }
}

/// The sample range, mapped onto this family's **half-open** convention.
///
/// [`DiscreteUniform::new`] has support `{lo, …, hi − 1}`, so the fit is
/// `lo = min` and `hi = max + 1`. Passing `max` directly is the obvious thing
/// and the wrong thing: it would leave the largest observed value outside the
/// support of the distribution fitted to it, giving that observation zero
/// probability under its own fit.
impl crate::Estimate for DiscreteUniform {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::new(data)?;
        if data.iter().any(|x| x.fract() != 0.0) {
            return None;
        }
        if s.min < i64::MIN as f64 || s.max >= i64::MAX as f64 {
            return None;
        }
        DiscreteUniform::new(s.min as i64, s.max as i64 + 1).ok()
    }
}

#[cfg(test)]
mod tests;

// Mass kernel and discrete distribution traits.
impl<I: Integer> IntegerMassKernel<I> for DiscreteUniform {
    fn ln_pmf_kernel(&self, _k: I) -> I::Real {
        kernels::discrete_uniform_log_pmf_on_support(I::Real::scalar(
            (self.hi() - self.lo()) as f64,
        ))
    }

    fn pmf_kernel(&self, k: I) -> I::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<I> IntegerDiscrete<I> for DiscreteUniform
where
    I: Integer + Compare,
    I::Real: Compare<Mask = <I as Compare>::Mask>,
{
    fn pmf(&self, k: I) -> I::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: I) -> I::Real {
        let lo = I::from_i64(self.lo());
        let hi = I::from_i64(self.hi());
        let valid = k.ge(&lo).and(&k.lt(&hi));
        <I::Real as Compare>::select(&valid, self.ln_pmf_kernel(k), I::Real::neg_infinity())
    }
}
