use rand::Rng;
use rand::distr::Distribution;

use super::gamma::Gamma;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid::{Compare, Real};

/// The chi-squared distribution with `k > 0` degrees of freedom — equivalently
/// `Gamma(shape = k/2, rate = 1/2)`. Support `(0, ∞)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChiSquared {
    freedom: f64,
    gamma: Gamma,
}

impl ChiSquared {
    pub fn new(freedom: f64) -> Result<Self, BayesError> {
        if !freedom.is_finite() {
            return Err(BayesError::NonFinite {
                name: "freedom",
                value: freedom,
            });
        }
        if freedom <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "freedom",
                value: freedom,
            });
        }
        Ok(Self {
            freedom,
            gamma: Gamma::new(freedom / 2.0, 0.5)?,
        })
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }
}

impl Distribution<f64> for ChiSquared {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.gamma.sample(rng)
    }
}

impl<T: Real> Support<T> for ChiSquared {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for ChiSquared {
    fn cdf(&self, x: T) -> T {
        self.gamma.cdf(x)
    }
    fn inverse_cdf(&self, p: T) -> T {
        self.gamma.inverse_cdf(p)
    }
}

impl Moments for ChiSquared {
    fn mean(&self) -> f64 {
        self.freedom
    }
    fn variance(&self) -> f64 {
        2.0 * self.freedom
    }
    fn skewness(&self) -> f64 {
        (8.0 / self.freedom).sqrt()
    }
    fn kurtosis(&self) -> f64 {
        12.0 / self.freedom
    }
    fn mode(&self) -> f64 {
        (self.freedom - 2.0).max(0.0)
    }
    /// `H = k/2 + ln(2 Γ(k/2)) + (1 − k/2)ψ(k/2)`.
    fn entropy(&self) -> f64 {
        let h = self.freedom / 2.0;
        h + std::f64::consts::LN_2 + h.ln_gamma() + (1.0 - h) * h.digamma()
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn median(&self) -> f64 {
        f64::NAN
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
    fn moments_and_quantile() {
        let c = ChiSquared::new(5.0).unwrap();
        close(c.mean(), 5.0, 1e-12);
        close(c.variance(), 10.0, 1e-12);
        close(c.inverse_cdf(0.5), 4.351_460_191, 1e-4);
        for &p in &[0.1, 0.5, 0.9] {
            close(c.cdf(c.inverse_cdf(p)), p, 1e-8);
        }
    }

    #[test]
    fn entropy_matches_the_underlying_gamma() {
        // χ²(k) ≡ Gamma(k/2, rate 1/2), so the entropies must coincide.
        for &k in &[1.0, 2.0, 5.0, 9.0] {
            close(
                ChiSquared::new(k).unwrap().entropy(),
                Gamma::new(k / 2.0, 0.5).unwrap().entropy(),
                1e-12,
            );
        }
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for ChiSquared {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let half_freedom = T::scalar(0.5 * self.freedom());
        kernels::chi_squared_log_pdf_on_support(x, half_freedom.clone(), half_freedom.ln_gamma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for ChiSquared
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
