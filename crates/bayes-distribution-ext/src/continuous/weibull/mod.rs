use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{ContinuousCdf, Moments, Support};
use bayes_error::BayesError;
use bayes_kernels as kernels;
use euclid::LazyReal;

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

impl<T: LazyReal> Support<T> for Weibull {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: LazyReal> ContinuousCdf<T> for Weibull {
    fn cdf(&self, x: T) -> T {
        kernels::weibull_cdf(x, T::scalar(self.scale), T::scalar(self.shape))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::weibull_quantile(
            p.clamp(T::zero(), T::one()),
            T::scalar(self.scale),
            T::scalar(self.shape),
        )
    }
}

impl Moments for Weibull {
    fn mean(&self) -> f64 {
        self.scale * (1.0 + 1.0 / self.shape).gamma()
    }
    fn variance(&self) -> f64 {
        let g1 = (1.0 + 1.0 / self.shape).gamma();
        let g2 = (1.0 + 2.0 / self.shape).gamma();
        self.scale * self.scale * (g2 - g1 * g1)
    }
    fn median(&self) -> f64 {
        self.scale * std::f64::consts::LN_2.powf(1.0 / self.shape)
    }
    fn mode(&self) -> f64 {
        if self.shape > 1.0 {
            self.scale * ((self.shape - 1.0) / self.shape).powf(1.0 / self.shape)
        } else {
            0.0
        }
    }
    /// `H = γ(1 − 1/k) + ln(λ/k) + 1`, with `γ` the Euler–Mascheroni constant.
    fn entropy(&self) -> f64 {
        const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
        let (k, l) = (self.shape, self.scale);
        EULER_GAMMA * (1.0 - 1.0 / k) + (l / k).ln() + 1.0
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
        close(w.mean(), 2.0, 1e-10);
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
        close(s / n as f64, w.mean(), 0.02);
    }
}
