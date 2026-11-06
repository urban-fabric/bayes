use rand::Rng;
use rand::distr::Distribution;

use super::gamma::Gamma;
use crate::estimation::sample::Summary;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

/// The inverse-gamma distribution with shape `α > 0` and scale `β > 0`, support
/// `(0, ∞)` — the distribution of `1/X` for `X ~ Gamma(α, rate β)`. A common
/// conjugate prior for a variance.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{InverseGamma, Moments};
///
/// // The conjugate prior for a normal's variance. Mean scale/(shape - 1).
/// let ig = InverseGamma::new(3.0, 2.0)?;
/// assert!((ig.mean() - 1.0).abs() < 1e-12);
///
/// // Moments appear one at a time as the shape grows: the mean needs shape
/// // above 1 and the variance needs it above 2.
/// assert!(ig.variance().is_finite());
/// assert!(InverseGamma::new(1.5, 2.0)?.variance().is_nan());
/// assert!(InverseGamma::new(1.5, 2.0)?.mean().is_finite());
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseGamma {
    shape: f64,
    scale: f64,
    gamma: Gamma,
}

impl InverseGamma {
    pub fn new(shape: f64, scale: f64) -> Result<Self, BayesError> {
        validate::positive(&[("shape", shape), ("scale", scale)])?;
        Ok(Self {
            shape,
            scale,
            gamma: Gamma::new(shape, scale)?,
        })
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution<f64> for InverseGamma {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        1.0 / self.gamma.sample(rng)
    }
}

impl<T: Real> Support<T> for InverseGamma {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for InverseGamma {
    fn cdf(&self, x: T) -> T {
        kernels::inverse_gamma_cdf(x, T::scalar(self.shape), T::scalar(self.scale))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(|x| self.cdf(x), T::zero(), T::infinity(), p)
    }

    /// `ln P(a, β/x)`: the **lower** regularized incomplete gamma, because this
    /// family's CDF is the upper one.
    fn ln_sf(&self, x: T) -> T {
        kernels::inverse_gamma_ln_sf(x, T::scalar(self.shape()), T::scalar(self.scale()))
    }
}

impl Moments for InverseGamma {
    fn mean(&self) -> f64 {
        if self.shape > 1.0 {
            self.scale / (self.shape - 1.0)
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        if self.shape > 2.0 {
            let a = self.shape;
            self.scale * self.scale / ((a - 1.0) * (a - 1.0) * (a - 2.0))
        } else {
            f64::NAN
        }
    }
    fn mode(&self) -> f64 {
        self.scale / (self.shape + 1.0)
    }
    /// `H = α + ln(β Γ(α)) − (1+α)ψ(α)`.
    fn entropy(&self) -> f64 {
        let (a, b) = (self.shape, self.scale);
        a + b.ln() + a.ln_gamma() - (1.0 + a) * a.digamma()
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

    fn median(&self) -> f64 {
        f64::NAN
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for InverseGamma {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let shape = T::scalar(self.shape());
        kernels::inverse_gamma_log_pdf_on_support(
            x,
            shape.clone(),
            T::scalar(self.scale()),
            shape.ln_gamma(),
        )
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for InverseGamma
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

/// Method of moments. The ratio `v/m² = 1/(α − 2)` isolates the shape, giving
/// `α = 2 + m²/v`, and then `β = m(α − 1)`.
///
/// The fit therefore always lands at `α > 2`, the region where both moments
/// exist. That is a real limitation rather than an artefact: an inverse-gamma
/// with a heavier tail has infinite variance, so no finite sample variance
/// corresponds to one, and this estimator cannot reach it.
impl crate::Estimate for InverseGamma {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::with_spread(data)?;
        if s.min <= 0.0 {
            return None;
        }
        let shape = 2.0 + s.mean * s.mean / s.variance;
        InverseGamma::new(shape, s.mean * (shape - 1.0)).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;

    #[test]
    fn cdf_quantile_and_moments() {
        let ig = InverseGamma::new(3.0, 2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            assert_close!(ig.cdf(ig.inverse_cdf(p)), p, 1e-7);
        }
        assert_close!(ig.mean(), 2.0 / 2.0, 1e-12);
        assert_close!(ig.mode(), 2.0 / 4.0, 1e-12);
        assert!(InverseGamma::new(1.5, 1.0).unwrap().variance().is_nan());
    }

    #[test]
    fn entropy_known_value() {
        // H = α + ln(β Γ(α)) − (1+α)ψ(α); for InverseGamma(3, 2): ≈ 0.69515702.
        assert_close!(
            InverseGamma::new(3.0, 2.0).unwrap().entropy(),
            0.695_157_02,
            1e-6,
        );
    }
}
