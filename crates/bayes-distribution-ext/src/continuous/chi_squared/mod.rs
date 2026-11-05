use rand::Rng;
use rand::distr::Distribution;

use super::gamma::Gamma;
use crate::estimation::sample::Summary;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Real};

/// The chi-squared distribution with `k > 0` degrees of freedom — equivalently
/// `Gamma(shape = k/2, rate = 1/2)`. Support `(0, ∞)`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{ChiSquared, Continuous, Exponential, Moments};
///
/// // Two degrees of freedom is exactly an exponential with rate 1/2.
/// let chi = ChiSquared::new(2.0)?;
/// assert!((chi.pdf(3.0_f64) - Exponential::new(0.5)?.pdf(3.0_f64)).abs() < 1e-12);
///
/// // The mean *is* the degrees of freedom, and the variance is twice it.
/// assert!((chi.mean() - 2.0).abs() < 1e-12);
/// assert!((chi.variance() - 4.0).abs() < 1e-12);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChiSquared {
    freedom: f64,
    gamma: Gamma,
}

impl ChiSquared {
    pub fn new(freedom: f64) -> Result<Self, BayesError> {
        validate::positive(&[("freedom", freedom)])?;
        Ok(Self {
            freedom,
            gamma: Gamma::new(freedom / 2.0, 0.5)?,
        })
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }
}

impl Distribution<f64> for ChiSquared {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.gamma.sample(rng)
    }
}

impl<T: Real> Support<T> for ChiSquared {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for ChiSquared {
    fn cdf(&self, x: T) -> T {
        self.gamma.cdf(x)
    }
    fn inverse_cdf(&self, p: T) -> T {
        self.gamma.inverse_cdf(p)
    }

    /// `ln Q(k/2, x/2)` — the upper regularized incomplete gamma, which is what
    /// a chi-squared p-value actually is.
    ///
    /// The kernel takes **half** the degrees of freedom, as its siblings do.
    fn ln_sf(&self, x: T) -> T {
        kernels::chi_squared_ln_sf(x, T::scalar(self.freedom() / 2.0))
    }
}

impl Moments for ChiSquared {
    fn mean(&self) -> f64 {
        self.freedom
    }
    fn variance(&self) -> f64 {
        2.0 * self.freedom
    }
    fn skewness(&self) -> f64 {
        (8.0 / self.freedom).sqrt()
    }
    fn kurtosis(&self) -> f64 {
        12.0 / self.freedom
    }
    fn mode(&self) -> f64 {
        (self.freedom - 2.0).max(0.0)
    }
    /// `H = k/2 + ln(2 Γ(k/2)) + (1 − k/2)ψ(k/2)`.
    fn entropy(&self) -> f64 {
        let h = self.freedom / 2.0;
        h + std::f64::consts::LN_2 + h.ln_gamma() + (1.0 - h) * h.digamma()
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn median(&self) -> f64 {
        f64::NAN
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for ChiSquared {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let half_freedom = T::scalar(0.5 * self.freedom());
        kernels::chi_squared_log_pdf_on_support(x, half_freedom.clone(), half_freedom.ln_gamma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for ChiSquared
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        // `is_finite` belongs in the support test rather than beside it: at an infinite
        // argument the kernel evaluates a log term against a linear one, which is
        // `inf - inf = NaN`, so `pdf(inf)` came back `NaN` when the density there is 0.
        let valid = x.gt(&zero).and(&x.is_finite());
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

/// `k̂ = x̄`, since a chi-squared's mean *is* its degrees of freedom.
///
/// The variance offers a second estimate of the same parameter, `v/2`, and the
/// two disagreeing is the clearest sign the data is not chi-squared at all.
/// The mean is the one used because it is the efficient of the two.
impl crate::Estimate for ChiSquared {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::new(data)?;
        if s.min <= 0.0 {
            return None;
        }
        ChiSquared::new(s.mean).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;

    #[test]
    fn moments_and_quantile() {
        let c = ChiSquared::new(5.0).unwrap();
        assert_close!(c.mean(), 5.0, 1e-12);
        assert_close!(c.variance(), 10.0, 1e-12);
        assert_close!(c.inverse_cdf(0.5), 4.351_460_191, 1e-4);
        for &p in &[0.1, 0.5, 0.9] {
            assert_close!(c.cdf(c.inverse_cdf(p)), p, 1e-8);
        }
    }

    #[test]
    fn entropy_matches_the_underlying_gamma() {
        // χ²(k) ≡ Gamma(k/2, rate 1/2), so the entropies must coincide.
        for &k in &[1.0, 2.0, 5.0, 9.0] {
            assert_close!(
                ChiSquared::new(k).unwrap().entropy(),
                Gamma::new(k / 2.0, 0.5).unwrap().entropy(),
                1e-12,
            );
        }
    }
}
