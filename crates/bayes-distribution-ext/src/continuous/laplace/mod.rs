use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Estimate, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::Real;

/// Sample median by linear interpolation between order statistics (the "type 7"
/// estimator), matching `bayes_statistics::median`. Inlined here so the family
/// layer does not depend on the statistics layer (which depends back on it).
fn median(data: &[f64]) -> f64 {
    let mut v: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.is_empty() {
        return f64::NAN;
    }
    if v.len() == 1 {
        return v[0];
    }
    let h = (v.len() as f64 - 1.0) * 0.5;
    let lo = h.floor() as usize;
    let frac = h - lo as f64;
    if lo + 1 < v.len() {
        v[lo] + frac * (v[lo + 1] - v[lo])
    } else {
        v[lo]
    }
}

/// The Laplace (double-exponential) distribution with `location μ` and
/// `scale b > 0`, support `(-∞, ∞)`. Density `f(x) = 1/(2b) exp(-|x-μ|/b)`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Continuous, ContinuousCdf, Laplace, Moments, Normal};
///
/// // Two exponential tails back to back, so the density peaks at 1/(2b) and
/// // has a corner there rather than a smooth maximum.
/// let l = Laplace::new(0.0, 1.0)?;
/// assert!((l.pdf(0.0_f64) - 0.5).abs() < 1e-12);
/// assert!((l.cdf(0.0_f64) - 0.5).abs() < 1e-12);
/// assert!((l.variance() - 2.0).abs() < 1e-12);      // 2b^2
///
/// // Heavier tailed than a normal with the same variance.
/// assert!(l.pdf(6.0_f64) > Normal::new(0.0, 2.0_f64.sqrt())?.pdf(6.0_f64));
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Laplace {
    location: f64,
    scale: f64,
}

impl Laplace {
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

impl Distribution<f64> for Laplace {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.random::<f64>() - 0.5;
        self.location - self.scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

impl<T: Real> Support<T> for Laplace {
    fn min(&self) -> T {
        T::neg_infinity()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Laplace {
    fn cdf(&self, x: T) -> T {
        kernels::laplace_cdf(x, T::scalar(self.location), T::scalar(self.scale))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::laplace_quantile(
            p.clamp(T::zero(), T::one()),
            T::scalar(self.location),
            T::scalar(self.scale),
        )
    }

    /// `−ln 2 − (x − μ)/b` above the location — linear, so exact everywhere.
    fn ln_sf(&self, x: T) -> T {
        kernels::laplace_ln_sf(x, T::scalar(self.location()), T::scalar(self.scale()))
    }

    /// Inverse survival by **reflection**, not by `inverse_cdf(1 − p)`.
    ///
    /// A Laplace is symmetric about its location, so `isf(p) = 2·location − Q(p)`
    /// holds exactly. That keeps `p` itself as the argument, where the generic
    /// default forms `1 − p` and loses every `p` below about `1e-16`.
    fn isf(&self, p: T) -> T {
        T::two() * T::scalar(self.location()) - self.inverse_cdf(p)
    }
}

impl Moments for Laplace {
    fn mean(&self) -> f64 {
        self.location
    }
    fn variance(&self) -> f64 {
        2.0 * self.scale * self.scale
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    fn kurtosis(&self) -> f64 {
        3.0
    }
    fn entropy(&self) -> f64 {
        (2.0 * std::f64::consts::E * self.scale).ln()
    }
    fn median(&self) -> f64 {
        self.location
    }
    fn mode(&self) -> f64 {
        self.location
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Estimate for Laplace {
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        let m = median(data);
        let b = data.iter().map(|&x| (x - m).abs()).sum::<f64>() / data.len() as f64;
        Laplace::new(m, b).ok()
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Laplace {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::laplace_log_pdf(x, T::scalar(self.location()), T::scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T: Real> Continuous<T> for Laplace {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;

    #[test]
    fn cdf_quantile_pdf_and_moments() {
        let l = Laplace::new(1.0, 2.0).unwrap();
        assert_close!(l.cdf(1.0), 0.5, 1e-12);
        assert_close!(l.pdf(1.0), 1.0 / (2.0 * 2.0), 1e-12);
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            assert_close!(l.cdf(l.inverse_cdf(p)), p, 1e-12);
        }
        assert_close!(l.mean(), 1.0, 1e-12);
        assert_close!(l.variance(), 8.0, 1e-12);
    }
}
