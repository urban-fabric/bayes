use std::f64::consts::{PI, SQRT_2};

use crate::{Continuous, ContinuousDensityKernel};
use rand::Rng;
use rand::distr::Distribution;

use super::normal::standard_normal_polar;
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid::{Compare, Real};

/// The half-normal distribution with scale `σ > 0`, support `[0, ∞)` — the
/// distribution of `|Z|` for `Z ~ N(0, σ²)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfNormal {
    scale: f64,
}

impl HalfNormal {
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

    fn skewness(&self) -> f64 {
        f64::NAN
    }

    fn kurtosis(&self) -> f64 {
        f64::NAN
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn density_is_twice_the_normal_on_the_positive_axis() {
        let h = HalfNormal::new(1.5).unwrap();
        let normal_pdf = |x: f64| (-0.5 * (x / 1.5).powi(2)).exp() / (1.5 * TAU.sqrt());
        close(h.pdf(2.0), 2.0 * normal_pdf(2.0), 1e-12);
        assert_eq!(h.pdf(-1.0), 0.0);
    }

    #[test]
    fn cdf_quantile_and_moments() {
        let h = HalfNormal::new(2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            close(h.cdf(h.inverse_cdf(p)), p, 1e-9);
        }
        close(h.mean(), 2.0 * (2.0 / PI).sqrt(), 1e-12);
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
