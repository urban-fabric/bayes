use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::normal::standard_normal_polar;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid::{Compare, Real};

/// The Wald (a.k.a. **inverse Gaussian**) distribution with mean `μ > 0` and shape
/// `λ > 0`, support `(0, ∞)` — the first-passage-time law of Brownian motion with
/// drift; common in reliability and response-time modelling. Density
/// `f(x) = √(λ / 2πx³) · exp(−λ(x−μ)² / 2μ²x)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wald {
    mu: f64,
    shape: f64,
}

impl Wald {
    pub fn new(mu: f64, shape: f64) -> Result<Self, BayesError> {
        for (name, v) in [("mu", mu), ("shape", shape)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        Ok(Self { mu, shape })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }
    pub fn shape(&self) -> f64 {
        self.shape
    }
}

impl Distribution<f64> for Wald {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let (mu, lam) = (self.mu, self.shape);
        let nu = standard_normal_polar(rng);
        let y = nu * nu;
        let x = mu + (mu * mu * y) / (2.0 * lam)
            - (mu / (2.0 * lam)) * (4.0 * mu * lam * y + mu * mu * y * y).sqrt();
        let u: f64 = rng.random();
        if u <= mu / (mu + x) { x } else { mu * mu / x }
    }
}

impl<T: Real> Support<T> for Wald {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Wald {
    fn cdf(&self, x: T) -> T {
        kernels::wald_cdf(x, T::scalar(self.mu), T::scalar(self.shape))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(|x| self.cdf(x), T::zero(), T::infinity(), p)
    }
}

impl Moments for Wald {
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        self.mu * self.mu * self.mu / self.shape
    }
    fn skewness(&self) -> f64 {
        3.0 * (self.mu / self.shape).sqrt()
    }
    fn kurtosis(&self) -> f64 {
        15.0 * self.mu / self.shape
    }
    fn mode(&self) -> f64 {
        let r = self.mu / self.shape;
        self.mu * ((1.0 + 9.0 * r * r / 4.0).sqrt() - 1.5 * r)
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn entropy(&self) -> f64 {
        f64::NAN
    }

    fn median(&self) -> f64 {
        f64::NAN
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Wald {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::wald_log_pdf_on_support(x, T::scalar(self.mu()), T::scalar(self.shape()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for Wald
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
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
    fn pdf_integrates_and_cdf_quantile_round_trip() {
        let d = Wald::new(1.0, 3.0).unwrap();
        let (mut area, mut x, dx) = (0.0, 1e-4, 1e-3);
        while x < 8.0 {
            area += d.pdf(x) * dx;
            x += dx;
        }
        close(area, 1.0, 1e-2);
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            close(d.cdf(d.inverse_cdf(p)), p, 1e-6);
        }
        close(d.mean(), 1.0, 1e-12);
        close(d.variance(), 1.0 / 3.0, 1e-12);
    }

    #[test]
    fn cdf_is_monotone_and_bounded() {
        let d = Wald::new(2.0, 1.0).unwrap();
        let mut prev = 0.0;
        for i in 1..200 {
            let c = d.cdf(i as f64 * 0.1);
            assert!((0.0..=1.0).contains(&c));
            assert!(c >= prev - 1e-12);
            prev = c;
        }
    }

    #[test]
    fn sampled_mean_and_variance_match() {
        let mut rng = ChaCha8Rng::seed_from_u64(19);
        let d = Wald::new(2.0, 5.0).unwrap();
        let n = 100_000;
        let xs: Vec<f64> = (0..n).map(|_| d.sample(&mut rng)).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        close(mean, d.mean(), 0.02);
        let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        close(var, d.variance(), 0.1);
    }
}
