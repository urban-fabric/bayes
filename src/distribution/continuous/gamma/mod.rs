use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::normal::standard_normal_polar;
use crate::distribution::numeric::inverse_cdf_bisect;
use crate::distribution::{Continuous, ContinuousCdf, Estimate, Moments, Support};
use crate::error::BayesError;
use crate::function::gamma::{digamma, gamma_lr, ln_gamma};
use crate::lazy;

/// The gamma distribution with `shape (k) > 0` and `rate (β) > 0` (mean `k/β`),
/// support `(0, ∞)`. Density `f(x) = βᵏ x^{k-1} e^{-βx} / Γ(k)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gamma {
    shape: f64,
    rate: f64,
}

impl Gamma {
    pub fn new(shape: f64, rate: f64) -> Result<Self, BayesError> {
        for (name, v) in [("shape", shape), ("rate", rate)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        Ok(Self { shape, rate })
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Independent sum of Gamma variables with a common rate.
    ///
    /// If `X_i ~ Gamma(shape_i, rate)` independently, then
    /// `Σ X_i ~ Gamma(Σ shape_i, rate)`. Returns an error for an empty list or
    /// mismatched rates because the sum is not generally Gamma in that case.
    pub fn sum_independent_same_rate(parts: &[Self]) -> Result<Self, BayesError> {
        let Some(first) = parts.first() else {
            return Err(BayesError::Malformed {
                what: "gamma sum needs at least one distribution",
            });
        };
        let rate = first.rate;
        let mut shape = 0.0;
        for part in parts {
            let scale = rate.abs().max(part.rate.abs()).max(1.0);
            if (part.rate - rate).abs() > 1e-12 * scale {
                return Err(BayesError::Malformed {
                    what: "gamma sum requires a common rate",
                });
            }
            shape += part.shape;
        }
        Self::new(shape, rate)
    }
}

/// Marsaglia–Tsang sampler for a standard gamma (rate 1), any `shape > 0`.
pub(crate) fn sample_standard_gamma<R: Rng + ?Sized>(rng: &mut R, shape: f64) -> f64 {
    if shape < 1.0 {
        let g = sample_standard_gamma(rng, shape + 1.0);
        return g * rng.random::<f64>().powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal_polar(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.random();
        if u < 1.0 - 0.033_1 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

impl Distribution<f64> for Gamma {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_standard_gamma(rng, self.shape) / self.rate
    }
}

impl Continuous for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        lazy::gamma_log_pdf_on_support(x, self.shape, self.rate, ln_gamma(self.shape))
    }
}

impl Support for Gamma {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Gamma {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            gamma_lr(self.shape, self.rate * x)
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        inverse_cdf_bisect(|x| self.cdf(x), 0.0, f64::INFINITY, p)
    }
}

impl Moments for Gamma {
    fn mean(&self) -> Option<f64> {
        Some(self.shape / self.rate)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.shape / (self.rate * self.rate))
    }
    fn skewness(&self) -> Option<f64> {
        Some(2.0 / self.shape.sqrt())
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(6.0 / self.shape)
    }
    fn entropy(&self) -> Option<f64> {
        Some(
            self.shape - self.rate.ln()
                + ln_gamma(self.shape)
                + (1.0 - self.shape) * digamma(self.shape),
        )
    }
    fn mode(&self) -> Option<f64> {
        if self.shape >= 1.0 {
            Some((self.shape - 1.0) / self.rate)
        } else {
            Some(0.0)
        }
    }
}

impl Estimate for Gamma {
    fn from_sample(data: &[f64]) -> Option<Self> {
        if data.len() < 2 || data.iter().any(|&x| x <= 0.0) {
            return None;
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
        if var <= 0.0 {
            return None;
        }
        Gamma::new(mean * mean / var, mean / var).ok()
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
    fn gamma_reduces_to_exponential_at_shape_one() {
        let g = Gamma::new(1.0, 2.0).unwrap();
        close(g.cdf(1.0), 1.0 - (-2.0f64).exp(), 1e-10);
        close(g.pdf(0.5), 2.0 * (-1.0f64).exp(), 1e-10);
    }

    #[test]
    fn cdf_quantile_round_trip_and_moments() {
        let g = Gamma::new(3.5, 1.5).unwrap();
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            close(g.cdf(g.inverse_cdf(p)), p, 1e-8);
        }
        close(g.mean().unwrap(), 3.5 / 1.5, 1e-12);
        close(g.variance().unwrap(), 3.5 / (1.5 * 1.5), 1e-12);
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let g = Gamma::new(2.7, 0.8).unwrap();
        let n = 60_000;
        let s: f64 = (0..n).map(|_| g.sample(&mut rng)).sum();
        close(s / n as f64, g.mean().unwrap(), 0.03);
    }

    #[test]
    fn common_rate_gamma_sums_stay_gamma() {
        let a = Gamma::new(2.0, 3.0).unwrap();
        let b = Gamma::new(5.0, 3.0).unwrap();
        let sum = Gamma::sum_independent_same_rate(&[a, b]).unwrap();
        close(sum.shape(), 7.0, 1e-12);
        close(sum.rate(), 3.0, 1e-12);
        close(
            sum.mean().unwrap(),
            a.mean().unwrap() + b.mean().unwrap(),
            1e-12,
        );
        assert!(Gamma::sum_independent_same_rate(&[]).is_err());
        assert!(Gamma::sum_independent_same_rate(&[a, Gamma::new(1.0, 4.0).unwrap()]).is_err());
    }
}
