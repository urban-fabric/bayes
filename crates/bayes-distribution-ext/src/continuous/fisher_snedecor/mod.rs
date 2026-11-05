use rand::Rng;
use rand::distr::Distribution;

use super::chi_squared::ChiSquared;
use crate::estimation::sample::Summary;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Real};

/// The F (Fisher–Snedecor) distribution with numerator/denominator degrees of
/// freedom `d1, d2 > 0`, support `(0, ∞)` — the ratio of two scaled χ².
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{ContinuousCdf, FisherSnedecor, Moments, StudentT};
///
/// // F(1, nu) is a squared Student's t: P(F <= x) = P(|T| <= sqrt(x)).
/// let f = FisherSnedecor::new(1.0, 10.0)?;
/// let t = StudentT::new(10.0)?;
/// let x = 2.5_f64;
/// assert!((f.cdf(x) - (t.cdf(x.sqrt()) - t.cdf(-x.sqrt()))).abs() < 1e-10);
///
/// // The mean depends only on the denominator degrees of freedom, and does not
/// // exist at all below 2 of them.
/// assert!((f.mean() - 10.0 / 8.0).abs() < 1e-12);
/// assert!(FisherSnedecor::new(3.0, 2.0)?.mean().is_nan());
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FisherSnedecor {
    d1: f64,
    d2: f64,
    chi1: ChiSquared,
    chi2: ChiSquared,
}

impl FisherSnedecor {
    pub fn new(d1: f64, d2: f64) -> Result<Self, BayesError> {
        validate::positive(&[("d1", d1), ("d2", d2)])?;
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

    /// `ln I_{d2/(d1x+d2)}(d2/2, d1/2)` — the reflected incomplete beta, which
    /// is the survival with no subtraction.
    fn ln_sf(&self, x: T) -> T {
        kernels::fisher_snedecor_ln_sf(x, T::scalar(self.d1()), T::scalar(self.d2()))
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
        // `is_finite` belongs in the support test rather than beside it: at an infinite
        // argument the kernel evaluates a log term against a linear one, which is
        // `inf - inf = NaN`, so `pdf(inf)` came back `NaN` when the density there is 0.
        let valid = x.gt(&zero).and(&x.is_finite());
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

/// Method of moments, solved in the only order the equations permit.
///
/// The mean `d₂/(d₂ − 2)` involves the denominator degrees of freedom alone, so
/// it goes first: `d₂ = 2m/(m − 1)`. Substituting into
///
/// ```text
///     v = 2 d₂² (d₁ + d₂ − 2) / (d₁ (d₂ − 2)² (d₂ − 4))
/// ```
///
/// and writing `A` for the `d₁`-free factor `2d₂²/((d₂−2)²(d₂−4))` leaves
/// `v = A(1 + (d₂ − 2)/d₁)`, hence `d₁ = (d₂ − 2)/(v/A − 1)`.
///
/// Each precondition is a fact about the family rather than about the
/// arithmetic. An F has no mean unless `d₂ > 2` and no variance unless
/// `d₂ > 4`, so a sample mean at or below 1, or a variance at or below `A`,
/// means no F distribution has these moments.
impl crate::Estimate for FisherSnedecor {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::with_spread(data)?;
        if s.min <= 0.0 || s.mean <= 1.0 {
            return None;
        }
        let d2 = 2.0 * s.mean / (s.mean - 1.0);
        if d2 <= 4.0 {
            return None;
        }
        let free = 2.0 * d2 * d2 / ((d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0));
        let ratio = s.variance / free;
        if ratio <= 1.0 {
            return None;
        }
        FisherSnedecor::new((d2 - 2.0) / (ratio - 1.0), d2).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;

    #[test]
    fn cdf_quantile_and_moments() {
        let f = FisherSnedecor::new(5.0, 10.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            assert_close!(f.cdf(f.inverse_cdf(p)), p, 1e-6);
        }
        assert_close!(f.mean(), 10.0 / 8.0, 1e-12);
        assert!(FisherSnedecor::new(5.0, 2.0).unwrap().mean().is_nan());
        assert_close!(f.inverse_cdf(0.95), 3.325_835, 1e-3);
    }
}
