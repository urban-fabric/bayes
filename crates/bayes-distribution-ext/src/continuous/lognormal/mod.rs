use std::f64::consts::TAU;

use crate::{Continuous, ContinuousDensityKernel};
use rand::Rng;
use rand::distr::Distribution;

use super::normal::Normal;
use crate::estimation::sample::Summary;
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

/// The **log-normal** distribution: `exp(X)` for `X ~ Normal(mu, sigma)`, where
/// `mu` and `sigma` are the mean and standard deviation of the *logarithm*.
/// Support `(0, ∞)`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{LogNormal, Moments};
///
/// // The parameters describe the *log* of the variable. So the median is
/// // exp(mu) exactly, while the mean is larger by a factor exp(sigma^2 / 2) —
/// // the gap that makes a log-normal's mean an unreliable summary of it.
/// let ln = LogNormal::new(0.0, 1.0)?;
/// assert!((ln.median() - 1.0).abs() < 1e-9);
/// assert!((ln.mean() - 0.5_f64.exp()).abs() < 1e-12);
/// assert!(ln.mean() > ln.median());
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogNormal {
    underlying: Normal,
}

impl LogNormal {
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, BayesError> {
        Ok(Self {
            underlying: Normal::new(mean, std_dev)?,
        })
    }

    pub fn mu(&self) -> f64 {
        self.underlying.params().0
    }

    pub fn sigma(&self) -> f64 {
        self.underlying.params().1
    }
}

impl Distribution<f64> for LogNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.underlying.sample(rng).exp()
    }
}

impl<T: Real> Support<T> for LogNormal {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for LogNormal {
    fn cdf(&self, x: T) -> T {
        kernels::lognormal_cdf(x, T::scalar(self.mu()), T::scalar(self.sigma()))
    }
    fn inverse_cdf(&self, p: T) -> T {
        self.underlying.inverse_cdf(p).exp()
    }

    /// The normal's log-survival at `ln x`, inheriting its `erfcx` tail.
    fn ln_sf(&self, x: T) -> T {
        kernels::lognormal_ln_sf(x, T::scalar(self.mu()), T::scalar(self.sigma()))
    }
}

impl Moments for LogNormal {
    fn mean(&self) -> f64 {
        let (mu, sigma) = (self.mu(), self.sigma());
        (mu + sigma * sigma / 2.0).exp()
    }
    fn variance(&self) -> f64 {
        let (mu, sigma) = (self.mu(), self.sigma());
        let s2 = sigma * sigma;
        (s2.exp() - 1.0) * (2.0 * mu + s2).exp()
    }
    fn entropy(&self) -> f64 {
        let (mu, sigma) = (self.mu(), self.sigma());
        mu + 0.5 * (TAU * std::f64::consts::E * sigma * sigma).ln()
    }
    fn median(&self) -> f64 {
        self.mu().exp()
    }
    fn mode(&self) -> f64 {
        let (mu, sigma) = (self.mu(), self.sigma());
        (mu - sigma * sigma).exp()
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn skewness(&self) -> f64 {
        let w = (self.sigma() * self.sigma()).exp();
        (w + 2.0) * (w - 1.0).sqrt()
    }

    fn kurtosis(&self) -> f64 {
        let s2 = self.sigma() * self.sigma();
        (4.0 * s2).exp() + 2.0 * (3.0 * s2).exp() + 3.0 * (2.0 * s2).exp() - 6.0
    }
}

/// Maximum likelihood, which for a log-normal is the normal fit applied to
/// `ln x`: `μ̂` and `σ̂` are the mean and standard deviation of the **logs**.
///
/// They are not the mean and standard deviation of the data, and substituting
/// those is a common silent error. The sample mean estimates `exp(μ + σ²/2)`,
/// so using it as `μ̂` biases the fit upward by a factor that grows with the
/// spread — at `σ = 1` the mean of the data is already 65% above `e^μ`.
impl crate::Estimate for LogNormal {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::of_logs(data)?;
        LogNormal::new(s.mean, s.std_dev()).ok()
    }
}

#[cfg(test)]
mod tests;

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for LogNormal {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::lognormal_log_pdf_on_support(x, T::scalar(self.mu()), T::scalar(self.sigma()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for LogNormal
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}
