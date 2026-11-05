use std::f64::consts::{E, TAU};

use crate::{Continuous, ContinuousDensityKernel};
use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{ContinuousCdf, Estimate, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid_numeric::Real;

/// The **normal** (Gaussian) distribution with mean `mu` and standard deviation
/// `sigma > 0`. Support `ℝ`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{ContinuousCdf, Moments, Normal};
///
/// let n = Normal::new(0.0, 1.0)?;
/// assert!((n.cdf(0.0_f64) - 0.5).abs() < 1e-12);
/// // The number behind every 95% interval.
/// assert!((n.inverse_cdf(0.975_f64) - 1.959_963_984_540).abs() < 1e-9);
///
/// // Shifting and scaling maps the whole family onto the standard one.
/// let shifted = Normal::new(10.0, 2.0)?;
/// assert!((shifted.cdf(12.0_f64) - n.cdf(1.0_f64)).abs() < 1e-12);
/// assert!((shifted.variance() - 4.0).abs() < 1e-12);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normal {
    mean: f64,
    std_dev: f64,
}

impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, BayesError> {
        if !mean.is_finite() {
            return Err(BayesError::NonFinite {
                name: "mean",
                value: mean,
            });
        }
        if !std_dev.is_finite() {
            return Err(BayesError::NonFinite {
                name: "std_dev",
                value: std_dev,
            });
        }

        if std_dev <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "std_dev",
                value: std_dev,
            });
        }
        Ok(Self { mean, std_dev })
    }

    /// The distribution's parameters `(μ, σ)`. Moments are read through the
    /// [`Moments`] trait (`mean()`/`std_dev()`); this is the infallible accessor
    /// for the raw parameters (consumed by the matrix-coupled conjugate models in
    /// the sibling crate `ito`).
    pub fn params(&self) -> (f64, f64) {
        (self.mean, self.std_dev)
    }

    /// Independent sum `X + Y` for Normal random variables.
    pub fn sum_independent(self, other: Self) -> Self {
        Self {
            mean: self.mean + other.mean,
            std_dev: (self.std_dev * self.std_dev + other.std_dev * other.std_dev).sqrt(),
        }
    }

    /// Independent difference `X - Y` for Normal random variables.
    pub fn difference_independent(self, other: Self) -> Self {
        Self {
            mean: self.mean - other.mean,
            std_dev: (self.std_dev * self.std_dev + other.std_dev * other.std_dev).sqrt(),
        }
    }

    /// Affine image `aX + b`.
    pub fn affine(self, scale: f64, shift: f64) -> Result<Self, BayesError> {
        if !scale.is_finite() {
            return Err(BayesError::NonFinite {
                name: "scale",
                value: scale,
            });
        }
        if !shift.is_finite() {
            return Err(BayesError::NonFinite {
                name: "shift",
                value: shift,
            });
        }
        Self::new(scale * self.mean + shift, scale.abs() * self.std_dev)
    }

    /// Independent linear combination `shift + Σ aᵢXᵢ`.
    ///
    /// Returns an error for an empty list, non-finite coefficients, or a
    /// degenerate all-zero coefficient vector.
    pub fn linear_combination_independent(
        terms: &[(f64, Self)],
        shift: f64,
    ) -> Result<Self, BayesError> {
        if terms.is_empty() {
            return Err(BayesError::Malformed {
                what: "normal linear combination needs at least one term",
            });
        }
        if !shift.is_finite() || terms.iter().any(|(a, _)| !a.is_finite()) {
            return Err(BayesError::Malformed {
                what: "normal linear-combination coefficients must be finite",
            });
        }
        let mean = shift + terms.iter().map(|(a, n)| a * n.mean).sum::<f64>();
        let variance = terms
            .iter()
            .map(|(a, n)| a * a * n.std_dev * n.std_dev)
            .sum::<f64>();
        if variance <= 0.0 {
            return Err(BayesError::Malformed {
                what: "normal linear combination has zero variance",
            });
        }
        Self::new(mean, variance.sqrt())
    }
}

impl Distribution<f64> for Normal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let z = standard_normal_polar(rng);
        self.mean + self.std_dev * z
    }
}

pub fn standard_normal_polar<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    loop {
        let u: f64 = 2.0 * rng.random::<f64>() - 1.0;
        let v: f64 = 2.0 * rng.random::<f64>() - 1.0;
        let s = u * u + v * v;

        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

impl<T: Real> Support<T> for Normal {
    fn min(&self) -> T {
        T::neg_infinity()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Normal {
    fn cdf(&self, x: T) -> T {
        kernels::normal_cdf(x, T::scalar(self.mean), T::scalar(self.std_dev))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::normal_quantile(p, T::scalar(self.mean), T::scalar(self.std_dev))
    }
    /// Stable `ln Φ((x-μ)/σ)`. In the left tail (`z ≤ 0`) it uses `erfcx`, so it
    /// stays accurate where `Φ` itself underflows to `0`; in the right tail it
    /// defers to `ln(1 − sf)`.
    fn ln_cdf(&self, x: T) -> T {
        kernels::normal_ln_cdf(x, T::scalar(self.mean), T::scalar(self.std_dev))
    }
    /// Stable `ln(1 − Φ((x-μ)/σ))`, the mirror of [`ln_cdf`](Self::ln_cdf).
    fn ln_sf(&self, x: T) -> T {
        kernels::normal_ln_sf(x, T::scalar(self.mean), T::scalar(self.std_dev))
    }

    /// Inverse survival by **reflection**: a normal is symmetric about its
    /// mean, so `isf(p) = 2μ − Q(p)` exactly. The generic default forms `1 − p`
    /// and returns infinity for every `p` below about `1e-16` — which is the
    /// whole range a small one-sided p-value lives in.
    fn isf(&self, p: T) -> T {
        T::two() * T::scalar(self.params().0) - self.inverse_cdf(p)
    }
}

impl Moments for Normal {
    fn mean(&self) -> f64 {
        self.mean
    }
    fn variance(&self) -> f64 {
        self.std_dev * self.std_dev
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    fn kurtosis(&self) -> f64 {
        0.0
    }
    fn entropy(&self) -> f64 {
        0.5 * (TAU * E * self.std_dev * self.std_dev).ln()
    }
    fn median(&self) -> f64 {
        self.mean
    }
    fn mode(&self) -> f64 {
        self.mean
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Estimate for Normal {
    fn from_sample(data: &[f64]) -> Option<Self> {
        <Self as crate::ExponentialFamily>::fit(data)
    }
}

#[cfg(test)]
mod tests;

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Normal {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_log_pdf(x, T::scalar(mean), T::scalar(std_dev))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T: Real> Continuous<T> for Normal {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}
