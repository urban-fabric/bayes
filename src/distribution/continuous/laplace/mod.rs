use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Estimate, Moments, Support};
use crate::error::BayesError;
use crate::statistics::median;

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
        // Inverse transform via U − ½.
        let u: f64 = rng.random::<f64>() - 0.5;
        self.location - self.scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

impl Continuous for Laplace {
    fn pdf(&self, x: f64) -> f64 {
        (-(x - self.location).abs() / self.scale).exp() / (2.0 * self.scale)
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        -(x - self.location).abs() / self.scale - (2.0 * self.scale).ln()
    }
}

impl Support for Laplace {
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Laplace {
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        if x < self.location {
            0.5 * z.exp()
        } else {
            1.0 - 0.5 * (-z).exp()
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        let p = p.clamp(0.0, 1.0);
        let q = p - 0.5;
        self.location - self.scale * q.signum() * (1.0 - 2.0 * q.abs()).ln()
    }
}

impl Moments for Laplace {
    fn mean(&self) -> Option<f64> {
        Some(self.location)
    }
    fn variance(&self) -> Option<f64> {
        Some(2.0 * self.scale * self.scale)
    }
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(3.0) // excess kurtosis
    }
    fn entropy(&self) -> Option<f64> {
        Some((2.0 * std::f64::consts::E * self.scale).ln())
    }
    fn median(&self) -> Option<f64> {
        Some(self.location)
    }
    fn mode(&self) -> Option<f64> {
        Some(self.location)
    }
}

impl Estimate for Laplace {
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        // MLE: μ̂ = median, b̂ = mean(|xᵢ − median|).
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
        close(l.cdf(1.0), 0.5, 1e-12); // median at location
        close(l.pdf(1.0), 1.0 / (2.0 * 2.0), 1e-12); // peak 1/(2b)
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            close(l.cdf(l.inverse_cdf(p)), p, 1e-12);
        }
        close(l.mean().unwrap(), 1.0, 1e-12);
        close(l.variance().unwrap(), 8.0, 1e-12); // 2b² = 8
    }
}
