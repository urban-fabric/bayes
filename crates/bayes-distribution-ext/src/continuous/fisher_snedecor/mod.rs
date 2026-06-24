use rand::Rng;
use rand::distr::Distribution;

use super::chi_squared::ChiSquared;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid::{Compare, Real};

/// The F (Fisher–Snedecor) distribution with numerator/denominator degrees of
/// freedom `d1, d2 > 0`, support `(0, ∞)` — the ratio of two scaled χ².
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FisherSnedecor {
    d1: f64,
    d2: f64,
    chi1: ChiSquared,
    chi2: ChiSquared,
}

impl FisherSnedecor {
    pub fn new(d1: f64, d2: f64) -> Result<Self, BayesError> {
        for (name, v) in [("d1", d1), ("d2", d2)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        Ok(Self {
            d1,
            d2,
            chi1: ChiSquared::new(d1)?,
            chi2: ChiSquared::new(d2)?,
        })
    }

    pub fn d1(&self) -> f64 {
        self.d1
    }
    pub fn d2(&self) -> f64 {
        self.d2
    }
}

impl Distribution<f64> for FisherSnedecor {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        (self.chi1.sample(rng) / self.d1) / (self.chi2.sample(rng) / self.d2)
    }
}

impl<T: Real> Support<T> for FisherSnedecor {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for FisherSnedecor {
    fn cdf(&self, x: T) -> T {
        kernels::fisher_snedecor_cdf(x, T::scalar(self.d1), T::scalar(self.d2))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(|x| self.cdf(x), T::zero(), T::infinity(), p)
    }
}

impl Moments for FisherSnedecor {
    fn mean(&self) -> f64 {
        if self.d2 > 2.0 {
            self.d2 / (self.d2 - 2.0)
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        if self.d2 > 4.0 {
            let (d1, d2) = (self.d1, self.d2);
            2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0))
        } else {
            f64::NAN
        }
    }
    fn mode(&self) -> f64 {
        if self.d1 > 2.0 {
            (self.d1 - 2.0) / self.d1 * self.d2 / (self.d2 + 2.0)
        } else {
            0.0
        }
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

    fn entropy(&self) -> f64 {
        f64::NAN
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
    fn cdf_quantile_and_moments() {
        let f = FisherSnedecor::new(5.0, 10.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            close(f.cdf(f.inverse_cdf(p)), p, 1e-6);
        }
        close(f.mean(), 10.0 / 8.0, 1e-12);
        assert!(FisherSnedecor::new(5.0, 2.0).unwrap().mean().is_nan());
        close(f.inverse_cdf(0.95), 3.325_835, 1e-3);
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for FisherSnedecor {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let d1 = T::scalar(self.d1());
        let d2 = T::scalar(self.d2());
        let half_d1 = d1.clone() * T::half();
        let half_d2 = d2.clone() * T::half();
        let ln_beta = half_d1.clone().ln_beta(half_d2.clone());
        kernels::fisher_snedecor_log_pdf_on_support(x, d1, d2, half_d1, half_d2, ln_beta)
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for FisherSnedecor
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
