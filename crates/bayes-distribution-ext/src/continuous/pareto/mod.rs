use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{ContinuousCdf, Estimate, Moments, Support};
use bayes_error::BayesError;
use bayes_kernels as kernels;
use euclid::LazyReal;

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
        let u = 1.0 - rng.random::<f64>();
        self.scale / u.powf(1.0 / self.shape)
    }
}

impl<T: LazyReal> Support<T> for Pareto {
    fn min(&self) -> T {
        T::scalar(self.scale)
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: LazyReal> ContinuousCdf<T> for Pareto {
    fn cdf(&self, x: T) -> T {
        kernels::pareto_cdf(x, T::scalar(self.scale), T::scalar(self.shape))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::pareto_quantile(
            p.clamp(T::zero(), T::one()),
            T::scalar(self.scale),
            T::scalar(self.shape),
        )
    }
}

impl Moments for Pareto {
    fn mean(&self) -> f64 {
        if self.shape > 1.0 {
            self.shape * self.scale / (self.shape - 1.0)
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        if self.shape > 2.0 {
            let a = self.shape;
            self.scale * self.scale * a / ((a - 1.0) * (a - 1.0) * (a - 2.0))
        } else {
            f64::NAN
        }
    }
    fn median(&self) -> f64 {
        self.scale * 2.0_f64.powf(1.0 / self.shape)
    }
    fn mode(&self) -> f64 {
        self.scale
    }
    fn entropy(&self) -> f64 {
        (self.scale / self.shape).ln() + 1.0 / self.shape + 1.0
    }
    fn moment2(&self) -> f64 {
        let mean = self.mean();
        self.variance() + mean * mean
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn skewness(&self) -> f64 {
        f64::NAN
    }

    fn kurtosis(&self) -> f64 {
        f64::NAN
    }
}

impl Estimate for Pareto {
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 || data.iter().any(|&x| x <= 0.0) {
            return None;
        }
        let xm = data.iter().copied().fold(f64::INFINITY, f64::min);
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
        close(p.mean(), 3.0 * 2.0 / 2.0, 1e-12);
        assert!(Pareto::new(1.0, 1.5).unwrap().variance().is_nan());
        assert!(Pareto::new(1.0, 0.8).unwrap().mean().is_nan());
    }
}
