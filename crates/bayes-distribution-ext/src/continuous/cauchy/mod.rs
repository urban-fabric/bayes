use std::f64::consts::PI;

use crate::{Continuous, ContinuousDensityKernel};
use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::estimation::sample::quantile;
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::Real;

/// The Cauchy (Lorentz) distribution with location `x₀` and scale `γ > 0`. It has
/// **no finite mean or variance** — the canonical example of [`Moments`] returning
/// `NaN` for undefined analytic summaries. Support `(-∞, ∞)`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Cauchy, ContinuousCdf, Moments};
///
/// let c = Cauchy::new(0.0, 1.0)?;
/// // The quartiles sit exactly one scale either side of the location, which is
/// // why a Cauchy is fitted from its interquartile range rather than its
/// // moments.
/// assert!((c.cdf(1.0_f64) - 0.75).abs() < 1e-12);
/// assert!((c.cdf(-1.0_f64) - 0.25).abs() < 1e-12);
///
/// // It has no mean at all: the defining integral diverges. That is reported
/// // as NaN rather than as the zero its symmetry might suggest.
/// assert!(c.mean().is_nan());
/// assert!(c.variance().is_nan());
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cauchy {
    location: f64,
    scale: f64,
}

impl Cauchy {
    pub fn new(location: f64, scale: f64) -> Result<Self, BayesError> {
        if !location.is_finite() {
            return Err(BayesError::NonFinite {
                name: "location",
                value: location,
            });
        }
        validate::positive(&[("scale", scale)])?;
        Ok(Self { location, scale })
    }

    pub fn location(&self) -> f64 {
        self.location
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution<f64> for Cauchy {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.random();
        self.location + self.scale * (PI * (u - 0.5)).tan()
    }
}

impl<T: Real> Support<T> for Cauchy {
    fn min(&self) -> T {
        T::neg_infinity()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Cauchy {
    fn cdf(&self, x: T) -> T {
        kernels::cauchy_cdf(x, T::scalar(self.location), T::scalar(self.scale))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::cauchy_quantile(
            p.clamp(T::zero(), T::one()),
            T::scalar(self.location),
            T::scalar(self.scale),
        )
    }

    /// `atan(1/z)/π` above the location, which has no subtraction and gives the
    /// correct `1/(πz)` tail.
    fn ln_sf(&self, x: T) -> T {
        kernels::cauchy_ln_sf(x, T::scalar(self.location()), T::scalar(self.scale()))
    }

    /// Inverse survival by **reflection**, not by `inverse_cdf(1 − p)`.
    ///
    /// A Cauchy is symmetric about its location, so `isf(p) = 2·location − Q(p)`
    /// holds exactly. That keeps `p` itself as the argument, where the generic
    /// default forms `1 − p` and loses every `p` below about `1e-16`.
    fn isf(&self, p: T) -> T {
        T::two() * T::scalar(self.location()) - self.inverse_cdf(p)
    }
}

impl Moments for Cauchy {
    fn entropy(&self) -> f64 {
        (4.0 * PI * self.scale).ln()
    }
    fn median(&self) -> f64 {
        self.location
    }
    fn mode(&self) -> f64 {
        self.location
    }
    fn mean(&self) -> f64 {
        f64::NAN
    }

    fn variance(&self) -> f64 {
        f64::NAN
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
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Cauchy {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::cauchy_log_pdf(x, T::scalar(self.location()), T::scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T: Real> Continuous<T> for Cauchy {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

/// Location from the **median**, scale from the **interquartile range**.
///
/// The usual estimators do not exist here. A Cauchy has no mean, so the sample
/// mean converges to nothing — it is itself Cauchy distributed, with the same
/// scale, however large `n` gets — and no variance, so the sample variance
/// grows without bound. Order statistics are what remain, and they are
/// consistent: the quartiles sit at `location ± scale`, making the IQR exactly
/// `2·scale`.
impl crate::Estimate for Cauchy {
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        let location = quantile(data, 0.5)?;
        let spread = quantile(data, 0.75)? - quantile(data, 0.25)?;
        Cauchy::new(location, spread / 2.0).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;

    #[test]
    fn cdf_quantile_and_undefined_moments() {
        let c = Cauchy::new(1.0, 2.0).unwrap();
        assert_close!(c.cdf(1.0), 0.5, 1e-12);
        assert_close!(c.cdf(3.0), 0.75, 1e-12);
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert_close!(c.cdf(c.inverse_cdf(p)), p, 1e-10);
        }
        assert!(c.mean().is_nan());
        assert!(c.variance().is_nan());
        assert_eq!(c.median(), 1.0);
    }
}
