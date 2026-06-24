use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Estimate, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid::Real;

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn cdf_quantile_pdf_and_moments() {
        let l = Laplace::new(1.0, 2.0).unwrap();
        close(l.cdf(1.0), 0.5, 1e-12);
        close(l.pdf(1.0), 1.0 / (2.0 * 2.0), 1e-12);
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            close(l.cdf(l.inverse_cdf(p)), p, 1e-12);
        }
        close(l.mean(), 1.0, 1e-12);
        close(l.variance(), 8.0, 1e-12);
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
