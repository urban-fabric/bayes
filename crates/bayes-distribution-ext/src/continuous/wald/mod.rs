use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::estimation::sample::Summary;
use crate::normal::standard_normal_polar;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Real};

/// The Wald (a.k.a. **inverse Gaussian**) distribution with mean `μ > 0` and shape
/// `λ > 0`, support `(0, ∞)` — the first-passage-time law of Brownian motion with
/// drift; common in reliability and response-time modelling. Density
/// `f(x) = √(λ / 2πx³) · exp(−λ(x−μ)² / 2μ²x)`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Moments, Wald};
///
/// // The first-passage time of a Brownian motion with drift, so it lives on
/// // the positive half-line and leans right.
/// let w = Wald::new(1.0, 3.0)?;
/// assert!((w.mean() - 1.0).abs() < 1e-12);
/// assert!((w.variance() - 1.0 / 3.0).abs() < 1e-12);   // mu^3 / lambda
/// assert!(w.skewness() > 0.0);
///
/// // Raising the shape makes it more nearly normal.
/// assert!(Wald::new(1.0, 300.0)?.skewness() < w.skewness());
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wald {
    mu: f64,
    shape: f64,
}

impl Wald {
    pub fn new(mu: f64, shape: f64) -> Result<Self, BayesError> {
        validate::positive(&[("mu", mu), ("shape", shape)])?;
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
    /// `−u² + ln(½(erfcx(u) − erfcx(v)))`, which cancels the `exp(2λ/μ)`
    /// factor the textbook form cannot evaluate.
    fn ln_sf(&self, x: T) -> T {
        kernels::wald_ln_sf(x, T::scalar(self.mu), T::scalar(self.shape))
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
        // `is_finite` belongs in the support test rather than beside it: at an infinite
        // argument the kernel evaluates a log term against a linear one, which is
        // `inf - inf = NaN`, so `pdf(inf)` came back `NaN` when the density there is 0.
        let valid = x.gt(&zero).and(&x.is_finite());
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

/// Maximum likelihood, which is explicit for this family:
///
/// ```text
///     μ̂ = x̄,      λ̂ = n / Σ(1/xᵢ − 1/x̄)
/// ```
///
/// The second is a harmonic-mean discrepancy, and it is the exact MLE. Matching
/// the variance would instead give `λ̂ = μ̂³/v`, which is consistent but less
/// efficient — the inverse Gaussian likelihood depends on the data only through
/// `x̄` and that discrepancy, so these two statistics are jointly sufficient and
/// nothing is left on the table.
impl crate::Estimate for Wald {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::new(data)?;
        if s.min <= 0.0 {
            return None;
        }
        let discrepancy = data.iter().map(|&x| 1.0 / x - 1.0 / s.mean).sum::<f64>();
        if discrepancy <= 0.0 {
            return None;
        }
        Wald::new(s.mean, s.n / discrepancy).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn pdf_integrates_and_cdf_quantile_round_trip() {
        let d = Wald::new(1.0, 3.0).unwrap();
        let (mut area, mut x, dx) = (0.0, 1e-4, 1e-3);
        while x < 8.0 {
            area += d.pdf(x) * dx;
            x += dx;
        }
        assert_close!(area, 1.0, 1e-2);
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert_close!(d.cdf(d.inverse_cdf(p)), p, 1e-6);
        }
        assert_close!(d.mean(), 1.0, 1e-12);
        assert_close!(d.variance(), 1.0 / 3.0, 1e-12);
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
        assert_close!(mean, d.mean(), 0.02);
        let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        assert_close!(var, d.variance(), 0.1);
    }
}
