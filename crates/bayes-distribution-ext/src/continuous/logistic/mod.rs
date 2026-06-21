use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid::Real;

/// The logistic distribution with `location` `μ` and `scale` `s > 0`, support
/// `(-∞, ∞)`. Density `f(x) = e^{-z} / (s(1 + e^{-z})²)` with `z = (x − μ)/s`; the
/// CDF is the logistic sigmoid.
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
        if !scale.is_finite() {
            return Err(BayesError::NonFinite {
                name: "scale",
                value: scale,
            });
        }
        if scale <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "scale",
                value: scale,
            });
        }
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn cdf_pdf_quantile_and_moments() {
        let d = Logistic::new(1.0, 2.0).unwrap();
        close(d.cdf(1.0), 0.5, 1e-12);
        for &p in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            close(d.cdf(d.inverse_cdf(p)), p, 1e-10);
        }
        close(d.ln_cdf(0.5).exp(), d.cdf(0.5), 1e-12);
        close(d.ln_sf(0.5).exp(), d.sf(0.5), 1e-12);
        close(d.mean(), 1.0, 1e-12);
        close(
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
        close(s / n as f64, d.mean(), 0.05);
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
        self.ln_pdf_kernel(x)
    }
}
