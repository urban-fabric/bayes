use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid::{Compare, Real};

/// The Lomax (Pareto type II) distribution with shape `α > 0` and scale `λ > 0`,
/// support `[0, ∞)` — a Pareto shifted to start at 0, and the heavy-tailed
/// posterior-predictive of the Gamma–Exponential conjugate model. Density
/// `f(x) = (α/λ)(1 + x/λ)^{-(α+1)}`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lomax {
    shape: f64,
    scale: f64,
}

impl Lomax {
    pub fn new(shape: f64, scale: f64) -> Result<Self, BayesError> {
        for (name, v) in [("shape", shape), ("scale", scale)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        Ok(Self { shape, scale })
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution<f64> for Lomax {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.random();
        self.scale * ((1.0 - u).powf(-1.0 / self.shape) - 1.0)
    }
}

impl<T: Real> Support<T> for Lomax {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Lomax {
    fn cdf(&self, x: T) -> T {
        kernels::lomax_cdf(x, T::scalar(self.shape), T::scalar(self.scale))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::lomax_quantile(
            p.clamp(T::zero(), T::one()),
            T::scalar(self.scale),
            T::scalar(self.shape),
        )
    }
    fn ln_sf(&self, x: T) -> T {
        kernels::lomax_ln_sf(x, T::scalar(self.shape), T::scalar(self.scale))
    }
}

impl Moments for Lomax {
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
            self.scale * self.scale * a / ((a - 1.0) * (a - 1.0) * (a - 2.0))
        } else {
            f64::NAN
        }
    }
    fn median(&self) -> f64 {
        self.scale * (2.0_f64.powf(1.0 / self.shape) - 1.0)
    }
    fn mode(&self) -> f64 {
        0.0
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

    fn entropy(&self) -> f64 {
        f64::NAN
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Lomax {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::lomax_log_pdf_on_support(x, T::scalar(self.shape()), T::scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for Lomax
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
        let d = Lomax::new(3.0, 2.0).unwrap();
        close(d.cdf(0.0), 0.0, 1e-12);
        for &p in &[0.1, 0.5, 0.9] {
            close(d.cdf(d.inverse_cdf(p)), p, 1e-10);
        }
        close(d.mean(), 2.0 / 2.0, 1e-12);
        close(d.median(), 2.0 * (2.0_f64.powf(1.0 / 3.0) - 1.0), 1e-12);
        assert!(Lomax::new(1.5, 1.0).unwrap().variance().is_nan());
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let d = Lomax::new(4.0, 3.0).unwrap();
        let n = 200_000;
        let s: f64 = (0..n).map(|_| d.sample(&mut rng)).sum();
        close(s / n as f64, d.mean(), 0.05);
    }
}
