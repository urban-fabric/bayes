use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::estimation::sample::Summary;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::Real;

/// The logistic distribution with `location` `μ` and `scale` `s > 0`, support
/// `(-∞, ∞)`. Density `f(x) = e^{-z} / (s(1 + e^{-z})²)` with `z = (x − μ)/s`; the
/// CDF is the logistic sigmoid.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{ContinuousCdf, Logistic, Moments};
///
/// // The standard logistic's CDF *is* the sigmoid, which is why this family
/// // underlies logistic regression.
/// let l = Logistic::new(0.0, 1.0)?;
/// assert!((l.cdf(2.0_f64) - 1.0 / (1.0 + (-2.0_f64).exp())).abs() < 1e-12);
///
/// // Its variance is pi^2 s^2 / 3, so a unit scale is not a unit variance.
/// assert!((l.variance() - std::f64::consts::PI.powi(2) / 3.0).abs() < 1e-12);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Logistic {
    location: f64,
    scale: f64,
}

impl Logistic {
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

impl Distribution<f64> for Logistic {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.random();
        self.location + self.scale * (u / (1.0 - u)).ln()
    }
}

impl<T: Real> Support<T> for Logistic {
    fn min(&self) -> T {
        T::neg_infinity()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Logistic {
    fn cdf(&self, x: T) -> T {
        kernels::logistic_cdf(x, T::scalar(self.location), T::scalar(self.scale))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::logistic_quantile(p, T::scalar(self.location), T::scalar(self.scale))
    }
    fn ln_cdf(&self, x: T) -> T {
        kernels::logistic_ln_cdf(x, T::scalar(self.location), T::scalar(self.scale))
    }
    fn ln_sf(&self, x: T) -> T {
        kernels::logistic_ln_sf(x, T::scalar(self.location), T::scalar(self.scale))
    }

    /// Inverse survival by **reflection**, not by `inverse_cdf(1 − p)`.
    ///
    /// A logistic is symmetric about its location, so `isf(p) = 2·location − Q(p)`
    /// holds exactly. That keeps `p` itself as the argument, where the generic
    /// default forms `1 − p` and loses every `p` below about `1e-16`.
    fn isf(&self, p: T) -> T {
        T::two() * T::scalar(self.location()) - self.inverse_cdf(p)
    }
}

impl Moments for Logistic {
    fn mean(&self) -> f64 {
        self.location
    }
    fn variance(&self) -> f64 {
        self.scale * self.scale * std::f64::consts::PI.powi(2) / 3.0
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    fn kurtosis(&self) -> f64 {
        1.2
    }
    fn entropy(&self) -> f64 {
        self.scale.ln() + 2.0
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

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Logistic {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::logistic_log_pdf(x, T::scalar(self.location()), T::scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T: Real> Continuous<T> for Logistic {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        // The support is all of R, but an infinite argument still drives the kernel to
        // `inf - inf = NaN`. The density at an infinite argument is 0.
        let finite = x.is_finite();
        T::select(&finite, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

/// Method of moments: the mean is the location, and `Var = π²s²/3` the scale.
impl crate::Estimate for Logistic {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::with_spread(data)?;
        let scale = s.std_dev() * 3.0_f64.sqrt() / std::f64::consts::PI;
        Logistic::new(s.mean, scale).ok()
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
    fn cdf_pdf_quantile_and_moments() {
        let d = Logistic::new(1.0, 2.0).unwrap();
        assert_close!(d.cdf(1.0), 0.5, 1e-12);
        for &p in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            assert_close!(d.cdf(d.inverse_cdf(p)), p, 1e-10);
        }
        assert_close!(d.ln_cdf(0.5).exp(), d.cdf(0.5), 1e-12);
        assert_close!(d.ln_sf(0.5).exp(), d.sf(0.5), 1e-12);
        assert_close!(d.mean(), 1.0, 1e-12);
        assert_close!(
            d.variance(),
            4.0 * std::f64::consts::PI.powi(2) / 3.0,
            1e-12,
        );
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let d = Logistic::new(-1.0, 1.5).unwrap();
        let n = 80_000;
        let s: f64 = (0..n).map(|_| d.sample(&mut rng)).sum();
        assert_close!(s / n as f64, d.mean(), 0.05);
    }
}
