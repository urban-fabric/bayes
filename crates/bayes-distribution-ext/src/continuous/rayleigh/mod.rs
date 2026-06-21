use std::f64::consts::PI;

use crate::{Continuous, ContinuousDensityKernel};
use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{ContinuousCdf, Estimate, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid::{Compare, Real};

/// The Rayleigh distribution with scale `σ > 0`, support `[0, ∞)`. Density
/// `f(x) = (x/σ²) exp(−x²/(2σ²))` — the magnitude of a 2-D centered Gaussian vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rayleigh {
    scale: f64,
}

impl Rayleigh {
    pub fn new(scale: f64) -> Result<Self, BayesError> {
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
        Ok(Self { scale })
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution<f64> for Rayleigh {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.inverse_cdf(rng.random::<f64>())
    }
}

impl<T: Real> Support<T> for Rayleigh {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Rayleigh {
    fn cdf(&self, x: T) -> T {
        kernels::rayleigh_cdf(x, T::scalar(self.scale))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::rayleigh_quantile(p.clamp(T::zero(), T::one()), T::scalar(self.scale))
    }
}

impl Moments for Rayleigh {
    fn mean(&self) -> f64 {
        self.scale * (PI / 2.0).sqrt()
    }
    fn variance(&self) -> f64 {
        (4.0 - PI) / 2.0 * self.scale * self.scale
    }
    fn median(&self) -> f64 {
        self.scale * (2.0 * std::f64::consts::LN_2).sqrt()
    }
    fn mode(&self) -> f64 {
        self.scale
    }
    fn entropy(&self) -> f64 {
        1.0 + (self.scale / 2.0_f64.sqrt()).ln() + 0.577_215_664_901_532_9 / 2.0
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

impl Estimate for Rayleigh {
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 || data.iter().any(|&x| x < 0.0) {
            return None;
        }
        let s2 = data.iter().map(|&x| x * x).sum::<f64>() / (2.0 * data.len() as f64);
        Rayleigh::new(s2.sqrt()).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn cdf_quantile_and_moments() {
        let r = Rayleigh::new(2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            close(r.cdf(r.inverse_cdf(p)), p, 1e-12);
        }
        close(r.mean(), 2.0 * (PI / 2.0).sqrt(), 1e-12);
        close(r.mode(), 2.0, 1e-12);
        close(r.pdf(2.0), (2.0 / 4.0) * (-0.5f64).exp(), 1e-12);
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Rayleigh {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::rayleigh_log_pdf_on_support(x, T::scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for Rayleigh
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
