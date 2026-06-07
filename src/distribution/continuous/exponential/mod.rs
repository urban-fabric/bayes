use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Estimate, Moments, Support};
use crate::error::BayesError;
use crate::function::stable::ln_1m_exp;

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

impl Continuous for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate * (-self.rate * x).exp()
        }
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            self.rate.ln() - self.rate * x
        }
    }
}

impl Support for Exponential {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Exponential {
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-self.rate * x).exp()
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        -(1.0 - p.clamp(0.0, 1.0)).ln() / self.rate
    }
    /// `ln(1 − e^{−λx})`, stable near `x = 0` via `ln_1m_exp`.
    fn ln_cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            f64::NEG_INFINITY
        } else {
            ln_1m_exp(-self.rate * x)
        }
    }
    /// `ln P(X > x) = −λx` exactly — no underflow, unlike `ln(1 − cdf)`.
    fn ln_sf(&self, x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { -self.rate * x }
    }
}

impl Moments for Exponential {
    fn mean(&self) -> Option<f64> {
        Some(self.rate.recip())
    }
    fn variance(&self) -> Option<f64> {
        Some(self.rate.powi(-2))
    }
    fn skewness(&self) -> Option<f64> {
        Some(2.0)
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(6.0)
    }
    fn entropy(&self) -> Option<f64> {
        Some(1.0 - self.rate.ln())
    }
    fn median(&self) -> Option<f64> {
        Some(std::f64::consts::LN_2 / self.rate)
    }
    fn mode(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl Estimate for Exponential {
    fn from_sample(data: &[f64]) -> Option<Self> {
        <Self as crate::distribution::ExponentialFamily>::fit(data)
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
        close(e.mean().unwrap(), 0.5, 1e-12);
        close(e.variance().unwrap(), 0.25, 1e-12);
        close(e.median().unwrap(), std::f64::consts::LN_2 / 2.0, 1e-12);
    }
}
