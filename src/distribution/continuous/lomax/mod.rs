use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::lazy;

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

impl Continuous for Lomax {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return f64::NEG_INFINITY;
        }
        lazy::lomax_log_pdf_on_support(x, self.shape, self.scale)
    }
}

impl Support for Lomax {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Lomax {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            1.0 - (1.0 + x / self.scale).powf(-self.shape)
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.scale * ((1.0 - p.clamp(0.0, 1.0)).powf(-1.0 / self.shape) - 1.0)
    }
    fn ln_sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            -self.shape * (1.0 + x / self.scale).ln()
        }
    }
}

impl Moments for Lomax {
    fn mean(&self) -> Option<f64> {
        (self.shape > 1.0).then(|| self.scale / (self.shape - 1.0))
    }
    fn variance(&self) -> Option<f64> {
        if self.shape > 2.0 {
            let a = self.shape;
            Some(self.scale * self.scale * a / ((a - 1.0) * (a - 1.0) * (a - 2.0)))
        } else {
            None
        }
    }
    fn median(&self) -> Option<f64> {
        Some(self.scale * (2.0_f64.powf(1.0 / self.shape) - 1.0))
    }
    fn mode(&self) -> Option<f64> {
        Some(0.0)
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
        close(d.mean().unwrap(), 2.0 / 2.0, 1e-12);
        close(
            d.median().unwrap(),
            2.0 * (2.0_f64.powf(1.0 / 3.0) - 1.0),
            1e-12,
        );
        assert!(Lomax::new(1.5, 1.0).unwrap().variance().is_none());
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let d = Lomax::new(4.0, 3.0).unwrap();
        let n = 200_000;
        let s: f64 = (0..n).map(|_| d.sample(&mut rng)).sum();
        close(s / n as f64, d.mean().unwrap(), 0.05);
    }
}
