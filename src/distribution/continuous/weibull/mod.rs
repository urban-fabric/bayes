use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::function::gamma::gamma;

/// The Weibull distribution with `scale λ > 0` and `shape k > 0`, support `[0, ∞)`.
/// Density `f(x) = (k/λ)(x/λ)^{k-1} exp(-(x/λ)^k)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weibull {
    scale: f64,
    shape: f64,
}

impl Weibull {
    pub fn new(scale: f64, shape: f64) -> Result<Self, BayesError> {
        for (name, v) in [("scale", scale), ("shape", shape)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        Ok(Self { scale, shape })
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
    pub fn shape(&self) -> f64 {
        self.shape
    }
}

impl Distribution<f64> for Weibull {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u = 1.0 - rng.random::<f64>();
        self.scale * (-u.ln()).powf(1.0 / self.shape)
    }
}

impl Continuous for Weibull {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let z = x / self.scale;
        (self.shape / self.scale) * z.powf(self.shape - 1.0) * (-z.powf(self.shape)).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = x / self.scale;
        (self.shape / self.scale).ln() + (self.shape - 1.0) * z.ln() - z.powf(self.shape)
    }
}

impl Support for Weibull {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Weibull {
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-(x / self.scale).powf(self.shape)).exp()
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.scale * (-(1.0 - p.clamp(0.0, 1.0)).ln()).powf(1.0 / self.shape)
    }
}

impl Moments for Weibull {
    fn mean(&self) -> Option<f64> {
        Some(self.scale * gamma(1.0 + 1.0 / self.shape))
    }
    fn variance(&self) -> Option<f64> {
        let g1 = gamma(1.0 + 1.0 / self.shape);
        let g2 = gamma(1.0 + 2.0 / self.shape);
        Some(self.scale * self.scale * (g2 - g1 * g1))
    }
    fn median(&self) -> Option<f64> {
        Some(self.scale * std::f64::consts::LN_2.powf(1.0 / self.shape))
    }
    fn mode(&self) -> Option<f64> {
        if self.shape > 1.0 {
            Some(self.scale * ((self.shape - 1.0) / self.shape).powf(1.0 / self.shape))
        } else {
            Some(0.0)
        }
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
    fn weibull_reduces_to_exponential_at_shape_one() {
        let w = Weibull::new(2.0, 1.0).unwrap();
        close(w.cdf(2.0), 1.0 - (-1.0f64).exp(), 1e-12);
        close(w.mean().unwrap(), 2.0, 1e-10);
    }

    #[test]
    fn cdf_quantile_and_sampled_mean() {
        let w = Weibull::new(1.5, 2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            close(w.cdf(w.inverse_cdf(p)), p, 1e-12);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(8);
        let n = 60_000;
        let s: f64 = (0..n).map(|_| w.sample(&mut rng)).sum();
        close(s / n as f64, w.mean().unwrap(), 0.02);
    }
}
