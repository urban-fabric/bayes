use std::f64::consts::{PI, SQRT_2};

use crate::{Continuous, ContinuousDensityKernel};
use rand::Rng;
use rand::distr::Distribution;

use super::normal::standard_normal_polar;
use crate::estimation::sample::Summary;
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

/// The half-normal distribution with scale `σ > 0`, support `[0, ∞)` — the
/// distribution of `|Z|` for `Z ~ N(0, σ²)`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Continuous, HalfNormal, Moments, Normal};
///
/// // The absolute value of a centred normal, so its density is twice the
/// // normal's on the positive half-line.
/// let h = HalfNormal::new(1.0)?;
/// let n = Normal::new(0.0, 1.0)?;
/// assert!((h.pdf(0.7_f64) - 2.0 * n.pdf(0.7_f64)).abs() < 1e-12);
///
/// // Folding moves the mean off zero to sigma * sqrt(2/pi).
/// assert!((h.mean() - (2.0 / std::f64::consts::PI).sqrt()).abs() < 1e-12);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfNormal {
    scale: f64,
}

impl HalfNormal {
    pub fn new(scale: f64) -> Result<Self, BayesError> {
        validate::positive(&[("scale", scale)])?;
        Ok(Self { scale })
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution<f64> for HalfNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        (self.scale * standard_normal_polar(rng)).abs()
    }
}

impl<T: Real> Support<T> for HalfNormal {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for HalfNormal {
    fn cdf(&self, x: T) -> T {
        kernels::half_normal_cdf(x, T::scalar(self.scale))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::half_normal_quantile(p.clamp(T::zero(), T::one()), T::scalar(self.scale))
    }

    /// `ln erfc(x/(σ√2))`, through `erfcx` so it stays finite past the point
    /// where `erfc` itself underflows.
    fn ln_sf(&self, x: T) -> T {
        kernels::half_normal_ln_sf(x, T::scalar(self.scale()))
    }
    /// `σ√2·erfc_inv(p)`, which takes the tail probability directly.
    fn isf(&self, p: T) -> T {
        kernels::half_normal_isf(p, T::scalar(self.scale()))
    }
}

impl Moments for HalfNormal {
    fn mean(&self) -> f64 {
        self.scale * (2.0 / PI).sqrt()
    }
    fn variance(&self) -> f64 {
        self.scale * self.scale * (1.0 - 2.0 / PI)
    }
    fn median(&self) -> f64 {
        self.scale * SQRT_2 * 0.5_f64.erf_inv()
    }
    fn mode(&self) -> f64 {
        0.0
    }
    fn entropy(&self) -> f64 {
        0.5 * (PI * self.scale * self.scale / 2.0).ln() + 0.5
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// `√2(4−π) / (π−2)^{3/2}` — a scale-free constant (≈ 0.995).
    fn skewness(&self) -> f64 {
        let pi = std::f64::consts::PI;
        std::f64::consts::SQRT_2 * (4.0 - pi) / (pi - 2.0).powf(1.5)
    }

    /// `8(π−3) / (π−2)²` — a scale-free constant (≈ 0.869).
    fn kurtosis(&self) -> f64 {
        let pi = std::f64::consts::PI;
        8.0 * (pi - 3.0) / ((pi - 2.0) * (pi - 2.0))
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for HalfNormal {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::half_normal_log_pdf_on_support(x, T::scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for HalfNormal
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.ge(&zero);
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

/// The maximum likelihood estimator `σ̂ = √(Σx²/n)`.
///
/// Matching the mean would also be consistent — `E[X] = σ√(2/π)` — but the
/// likelihood equation for a half-normal solves explicitly, so there is no
/// reason to accept the less efficient estimator.
///
/// The second moment is taken about **zero**, not about the sample mean. This
/// family's location is fixed at the origin, so that is the point deviations
/// are measured from; centring on `x̄` would fit the wrong quantity.
impl crate::Estimate for HalfNormal {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::new(data)?;
        if s.min < 0.0 {
            return None;
        }
        HalfNormal::new(s.second_moment().sqrt()).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;
    use std::f64::consts::TAU;

    #[test]
    fn density_is_twice_the_normal_on_the_positive_axis() {
        let h = HalfNormal::new(1.5).unwrap();
        let normal_pdf = |x: f64| (-0.5 * (x / 1.5).powi(2)).exp() / (1.5 * TAU.sqrt());
        assert_close!(h.pdf(2.0), 2.0 * normal_pdf(2.0), 1e-12);
        assert_eq!(h.pdf(-1.0), 0.0);
    }

    #[test]
    fn cdf_quantile_and_moments() {
        let h = HalfNormal::new(2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            assert_close!(h.cdf(h.inverse_cdf(p)), p, 1e-9);
        }
        assert_close!(h.mean(), 2.0 * (2.0 / PI).sqrt(), 1e-12);
    }
}
