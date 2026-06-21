use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{ContinuousCdf, Estimate, Moments, Support};
use bayes_error::BayesError;
use bayes_kernels as kernels;
use euclid::LazyReal;

#[cfg(test)]
use crate::Continuous;

/// The exponential distribution with rate `λ > 0` (mean `1/λ`), support `[0, ∞)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exponential {
    rate: f64,
}

impl Exponential {
    pub fn new(rate: f64) -> Result<Self, BayesError> {
        if !rate.is_finite() {
            return Err(BayesError::NonFinite {
                name: "rate",
                value: rate,
            });
        }
        if rate <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "rate",
                value: rate,
            });
        }
        Ok(Self { rate })
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl Distribution<f64> for Exponential {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        -(1.0 - rng.random::<f64>()).ln() / self.rate
    }
}

impl<T: LazyReal> Support<T> for Exponential {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: LazyReal> ContinuousCdf<T> for Exponential {
    fn cdf(&self, x: T) -> T {
        kernels::exponential_cdf(x, T::scalar(self.rate))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::exponential_quantile(p.clamp(T::zero(), T::one()), T::scalar(self.rate))
    }
    /// `ln(1 − e^{−λx})`, stable near `x = 0` via `ln_1m_exp`.
    fn ln_cdf(&self, x: T) -> T {
        kernels::exponential_ln_cdf(x, T::scalar(self.rate))
    }
    /// `ln P(X > x) = −λx` exactly — no underflow, unlike `ln(1 − cdf)`.
    fn ln_sf(&self, x: T) -> T {
        kernels::exponential_ln_sf(x, T::scalar(self.rate))
    }
}

impl Moments for Exponential {
    fn mean(&self) -> f64 {
        self.rate.recip()
    }
    fn variance(&self) -> f64 {
        self.rate.powi(-2)
    }
    fn skewness(&self) -> f64 {
        2.0
    }
    fn kurtosis(&self) -> f64 {
        6.0
    }
    fn entropy(&self) -> f64 {
        1.0 - self.rate.ln()
    }
    fn median(&self) -> f64 {
        std::f64::consts::LN_2 / self.rate
    }
    fn mode(&self) -> f64 {
        0.0
    }
    fn moment2(&self) -> f64 {
        let mean = self.mean();
        self.variance() + mean * mean
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Estimate for Exponential {
    fn from_sample(data: &[f64]) -> Option<Self> {
        <Self as crate::ExponentialFamily>::fit(data)
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
    fn cdf_pdf_quantile_moments() {
        let e = Exponential::new(2.0).unwrap();
        close(e.cdf(0.0), 0.0, 1e-12);
        close(e.cdf(1.0), 1.0 - (-2.0f64).exp(), 1e-12);
        close(e.pdf(0.0), 2.0, 1e-12);
        for &p in &[0.1, 0.5, 0.9] {
            close(e.cdf(e.inverse_cdf(p)), p, 1e-12);
        }
        close(e.mean(), 0.5, 1e-12);
        close(e.variance(), 0.25, 1e-12);
        close(e.median(), std::f64::consts::LN_2 / 2.0, 1e-12);
    }
}
