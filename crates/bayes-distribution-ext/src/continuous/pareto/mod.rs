use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Estimate, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

/// The Pareto distribution with `scale x_m > 0` (the minimum) and `shape α > 0`,
/// support `[x_m, ∞)`. Density `f(x) = α x_mᵅ / x^{α+1}` for `x ≥ x_m`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{ContinuousCdf, Moments, Pareto};
///
/// // Scale-free: doubling the argument always removes the same fraction of the
/// // remaining tail, so F(2 * scale) = 1 - 2^-shape whatever the scale is.
/// let p = Pareto::new(1.0, 3.0)?;
/// assert!((p.cdf(2.0_f64) - 0.875).abs() < 1e-12);
/// assert!((p.mean() - 1.5).abs() < 1e-12);     // scale * shape / (shape - 1)
///
/// // The shape is a tail index: below 2 there is no variance.
/// assert!(Pareto::new(1.0, 1.5)?.variance().is_nan());
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pareto {
    scale: f64,
    shape: f64,
}

impl Pareto {
    pub fn new(scale: f64, shape: f64) -> Result<Self, BayesError> {
        validate::positive(&[("scale", scale), ("shape", shape)])?;
        Ok(Self { scale, shape })
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
    pub fn shape(&self) -> f64 {
        self.shape
    }
}

impl Distribution<f64> for Pareto {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u = 1.0 - rng.random::<f64>();
        self.scale / u.powf(1.0 / self.shape)
    }
}

impl<T: Real> Support<T> for Pareto {
    fn min(&self) -> T {
        T::scalar(self.scale)
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Pareto {
    fn cdf(&self, x: T) -> T {
        kernels::pareto_cdf(x, T::scalar(self.scale), T::scalar(self.shape))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::pareto_quantile(
            p.clamp(T::zero(), T::one()),
            T::scalar(self.scale),
            T::scalar(self.shape),
        )
    }

    /// `α(ln x_m − ln x)`: a power-law survival is exactly linear in logs.
    fn ln_sf(&self, x: T) -> T {
        kernels::pareto_ln_sf(x, T::scalar(self.scale()), T::scalar(self.shape()))
    }
    /// `x_m p^{−1/α}`.
    fn isf(&self, p: T) -> T {
        kernels::pareto_isf(p, T::scalar(self.scale()), T::scalar(self.shape()))
    }
}

impl Moments for Pareto {
    fn mean(&self) -> f64 {
        if self.shape > 1.0 {
            self.shape * self.scale / (self.shape - 1.0)
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
        self.scale * 2.0_f64.powf(1.0 / self.shape)
    }
    fn mode(&self) -> f64 {
        self.scale
    }
    fn entropy(&self) -> f64 {
        (self.scale / self.shape).ln() + 1.0 / self.shape + 1.0
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

impl Estimate for Pareto {
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 || data.iter().any(|&x| x <= 0.0) {
            return None;
        }
        let xm = data.iter().copied().fold(f64::INFINITY, f64::min);
        let s: f64 = data.iter().map(|&x| (x / xm).ln()).sum();
        if s <= 0.0 {
            return None;
        }
        Pareto::new(xm, data.len() as f64 / s).ok()
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Pareto {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::pareto_log_pdf_on_support(x, T::scalar(self.scale()), T::scalar(self.shape()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for Pareto
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let min = T::scalar(self.scale());
        let valid = x.ge(&min);
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;

    #[test]
    fn cdf_quantile_and_moments() {
        let p = Pareto::new(2.0, 3.0).unwrap();
        assert_close!(p.cdf(2.0), 0.0, 1e-12);
        assert_close!(p.cdf(4.0), 1.0 - (0.5f64).powi(3), 1e-12);
        for &q in &[0.1, 0.5, 0.9] {
            assert_close!(p.cdf(p.inverse_cdf(q)), q, 1e-10);
        }
        assert_close!(p.mean(), 3.0 * 2.0 / 2.0, 1e-12);
        assert!(Pareto::new(1.0, 1.5).unwrap().variance().is_nan());
        assert!(Pareto::new(1.0, 0.8).unwrap().mean().is_nan());
    }
}
