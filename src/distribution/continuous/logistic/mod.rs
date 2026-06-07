use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::function::stable::ln_1p_exp;

/// The logistic distribution with `location` `μ` and `scale` `s > 0`, support
/// `(-∞, ∞)`. Density `f(x) = e^{-z} / (s(1 + e^{-z})²)` with `z = (x − μ)/s`; the
/// CDF is the logistic sigmoid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Logistic {
    location: f64,
    scale: f64,
}

impl Logistic {
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

impl Distribution<f64> for Logistic {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.random();
        self.location + self.scale * (u / (1.0 - u)).ln()
    }
}

impl Continuous for Logistic {
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        -z - self.scale.ln() - 2.0 * ln_1p_exp(-z)
    }
}

impl Support for Logistic {
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Logistic {
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        1.0 / (1.0 + (-z).exp())
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.location + self.scale * (p / (1.0 - p)).ln()
    }
    fn ln_cdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        -ln_1p_exp(-z)
    }
    fn ln_sf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        -ln_1p_exp(z)
    }
}

impl Moments for Logistic {
    fn mean(&self) -> Option<f64> {
        Some(self.location)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.scale * self.scale * std::f64::consts::PI.powi(2) / 3.0)
    }
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(1.2)
    }
    fn entropy(&self) -> Option<f64> {
        Some(self.scale.ln() + 2.0)
    }
    fn median(&self) -> Option<f64> {
        Some(self.location)
    }
    fn mode(&self) -> Option<f64> {
        Some(self.location)
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
        let d = Logistic::new(1.0, 2.0).unwrap();
        close(d.cdf(1.0), 0.5, 1e-12);
        for &p in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            close(d.cdf(d.inverse_cdf(p)), p, 1e-10);
        }
        close(d.ln_cdf(0.5).exp(), d.cdf(0.5), 1e-12);
        close(d.ln_sf(0.5).exp(), d.sf(0.5), 1e-12);
        close(d.mean().unwrap(), 1.0, 1e-12);
        close(
            d.variance().unwrap(),
            4.0 * std::f64::consts::PI.powi(2) / 3.0,
            1e-12,
        );
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let d = Logistic::new(-1.0, 1.5).unwrap();
        let n = 80_000;
        let s: f64 = (0..n).map(|_| d.sample(&mut rng)).sum();
        close(s / n as f64, d.mean().unwrap(), 0.05);
    }
}
