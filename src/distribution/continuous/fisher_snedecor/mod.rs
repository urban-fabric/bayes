use rand::Rng;
use rand::distr::Distribution;

use super::chi_squared::ChiSquared;
use crate::distribution::numeric::inverse_cdf_bisect;
use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::function::beta::{beta_reg, ln_beta};
use crate::lazy;

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

impl Continuous for FisherSnedecor {
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
        let (a, b) = (self.d1 / 2.0, self.d2 / 2.0);
        lazy::fisher_snedecor_log_pdf_on_support(x, self.d1, self.d2, a, b, ln_beta(a, b))
    }
}

impl Support for FisherSnedecor {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for FisherSnedecor {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        beta_reg(
            self.d1 / 2.0,
            self.d2 / 2.0,
            self.d1 * x / (self.d1 * x + self.d2),
        )
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        inverse_cdf_bisect(|x| self.cdf(x), 0.0, f64::INFINITY, p)
    }
}

impl Moments for FisherSnedecor {
    fn mean(&self) -> Option<f64> {
        (self.d2 > 2.0).then(|| self.d2 / (self.d2 - 2.0))
    }
    fn variance(&self) -> Option<f64> {
        if self.d2 > 4.0 {
            let (d1, d2) = (self.d1, self.d2);
            Some(2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0)))
        } else {
            None
        }
    }
    fn mode(&self) -> Option<f64> {
        if self.d1 > 2.0 {
            Some((self.d1 - 2.0) / self.d1 * self.d2 / (self.d2 + 2.0))
        } else {
            Some(0.0)
        }
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
        close(f.mean().unwrap(), 10.0 / 8.0, 1e-12);
        assert_eq!(FisherSnedecor::new(5.0, 2.0).unwrap().mean(), None);
        close(f.inverse_cdf(0.95), 3.325_835, 1e-3);
    }
}
