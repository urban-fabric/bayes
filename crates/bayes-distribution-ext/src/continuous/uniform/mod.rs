use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Estimate, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Real};

/// The **continuous uniform** distribution on `[lo, hi]` with `lo < hi`:
/// constant density `1/(hi − lo)` across the interval.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Continuous, ContinuousCdf, Moments, Uniform};
///
/// let u = Uniform::new(0.0, 10.0)?;
/// assert!((u.pdf(4.0_f64) - 0.1).abs() < 1e-12);
/// assert!((u.mean() - 5.0).abs() < 1e-12);
/// assert!((u.variance() - 100.0 / 12.0).abs() < 1e-12);
///
/// // Its CDF is linear between the bounds, which is what makes it the
/// // inverse-transform starting point for every other family.
/// assert!((u.cdf(2.5_f64) - 0.25).abs() < 1e-12);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniform {
    lo: f64,
    width: f64,
}

impl Uniform {
    pub const fn unit_interval() -> Self {
        Self {
            lo: 0.0,
            width: 1.0,
        }
    }

    pub fn symmetric_radius(radius: f64) -> Self {
        debug_assert!(radius.is_finite() && radius > 0.0);
        Self {
            lo: -radius,
            width: radius * 2.0,
        }
    }

    pub fn new(lo: f64, hi: f64) -> Result<Self, BayesError> {
        if !lo.is_finite() {
            return Err(BayesError::NonFinite {
                name: "lo",
                value: lo,
            });
        }
        if !hi.is_finite() {
            return Err(BayesError::NonFinite {
                name: "hi",
                value: hi,
            });
        }

        if lo >= hi {
            return Err(BayesError::EmptyRange { lo, hi });
        }
        Ok(Self { lo, width: hi - lo })
    }

    pub fn lo(&self) -> f64 {
        self.lo
    }

    pub fn hi(&self) -> f64 {
        self.lo + self.width
    }
}

impl Distribution<f64> for Uniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.lo + self.width * rng.random::<f64>()
    }
}

impl<T: Real> Support<T> for Uniform {
    fn min(&self) -> T {
        T::scalar(self.lo)
    }
    fn max(&self) -> T {
        T::scalar(self.lo + self.width)
    }
}

impl<T: Real> ContinuousCdf<T> for Uniform {
    fn cdf(&self, x: T) -> T {
        kernels::uniform_cdf(x, T::scalar(self.lo), T::scalar(self.width))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::uniform_quantile(
            p.clamp(T::zero(), T::one()),
            T::scalar(self.lo),
            T::scalar(self.width),
        )
    }
}

impl Moments for Uniform {
    fn mean(&self) -> f64 {
        self.lo + self.width / 2.0
    }
    fn variance(&self) -> f64 {
        self.width * self.width / 12.0
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    fn kurtosis(&self) -> f64 {
        -1.2
    }
    fn entropy(&self) -> f64 {
        self.width.ln()
    }
    fn median(&self) -> f64 {
        self.lo + self.width / 2.0
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn mode(&self) -> f64 {
        f64::NAN
    }
}

impl Estimate for Uniform {
    /// The sample range.
    ///
    /// The finiteness check is load bearing and was once missing. `f64::min`
    /// and `f64::max` **ignore** a `NaN` rather than propagating it — by IEEE
    /// design, since they are the `minNum`/`maxNum` operations — so folding
    /// with them over a sample containing a `NaN` silently skips it and returns
    /// a confident fit from the remaining points. Every other estimator in the
    /// crate inherits its rejection from
    /// [`Summary`](crate::estimation::sample::Summary); this one folds by hand
    /// and has to say so itself.
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let lo = data.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Uniform::new(lo, hi).ok()
    }
}

#[cfg(test)]
mod tests;

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Uniform {
    fn ln_pdf_kernel(&self, _x: T) -> T {
        kernels::uniform_log_pdf_on_support(T::scalar(self.hi() - self.lo()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for Uniform
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let lo = T::scalar(self.lo());
        let hi = T::scalar(self.hi());
        let valid = x.ge(&lo).and(&x.le(&hi));
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}
