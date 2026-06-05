use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Estimate, Moments, Support};
use crate::error::BayesError;

/// The Pareto distribution with `scale x_m > 0` (the minimum) and `shape α > 0`,
/// support `[x_m, ∞)`. Density `f(x) = α x_mᵅ / x^{α+1}` for `x ≥ x_m`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pareto {
    scale: f64,
    shape: f64,
}

impl Pareto {
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

impl Distribution<f64> for Pareto {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // Inverse transform: x_m / U^{1/α} (U uniform on (0,1]).
        let u = 1.0 - rng.random::<f64>();
        self.scale / u.powf(1.0 / self.shape)
    }
}

impl Continuous for Pareto {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.scale {
            0.0
        } else {
            self.shape * self.scale.powf(self.shape) / x.powf(self.shape + 1.0)
        }
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.scale {
            f64::NEG_INFINITY
        } else {
            self.shape.ln() + self.shape * self.scale.ln() - (self.shape + 1.0) * x.ln()
        }
    }
}

impl Support for Pareto {
    fn min(&self) -> f64 {
        self.scale
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Pareto {
    fn cdf(&self, x: f64) -> f64 {
        if x < self.scale {
            0.0
        } else {
            1.0 - (self.scale / x).powf(self.shape)
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.scale / (1.0 - p.clamp(0.0, 1.0)).powf(1.0 / self.shape)
    }
}

impl Moments for Pareto {
    fn mean(&self) -> Option<f64> {
        (self.shape > 1.0).then(|| self.shape * self.scale / (self.shape - 1.0))
    }
    fn variance(&self) -> Option<f64> {
        if self.shape > 2.0 {
            let a = self.shape;
            Some(self.scale * self.scale * a / ((a - 1.0) * (a - 1.0) * (a - 2.0)))
        } else {
            None // infinite for α ≤ 2
        }
    }
    fn median(&self) -> Option<f64> {
        Some(self.scale * 2.0_f64.powf(1.0 / self.shape))
    }
    fn mode(&self) -> Option<f64> {
        Some(self.scale)
    }
    fn entropy(&self) -> Option<f64> {
        Some((self.scale / self.shape).ln() + 1.0 / self.shape + 1.0)
    }
}

impl Estimate for Pareto {
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 || data.iter().any(|&x| x <= 0.0) {
            return None;
        }
        let xm = data.iter().copied().fold(f64::INFINITY, f64::min);
        // MLE shape: n / Σ ln(xᵢ / x_m).
        let s: f64 = data.iter().map(|&x| (x / xm).ln()).sum();
        if s <= 0.0 {
            return None;
        }
        Pareto::new(xm, data.len() as f64 / s).ok()
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
    fn cdf_quantile_and_moments() {
        let p = Pareto::new(2.0, 3.0).unwrap();
        close(p.cdf(2.0), 0.0, 1e-12);
        close(p.cdf(4.0), 1.0 - (0.5f64).powi(3), 1e-12);
        for &q in &[0.1, 0.5, 0.9] {
            close(p.cdf(p.inverse_cdf(q)), q, 1e-10);
        }
        close(p.mean().unwrap(), 3.0 * 2.0 / 2.0, 1e-12);
        // α = 1.5 ⇒ finite mean, infinite variance.
        assert!(Pareto::new(1.0, 1.5).unwrap().variance().is_none());
        assert_eq!(Pareto::new(1.0, 0.8).unwrap().mean(), None);
    }
}
