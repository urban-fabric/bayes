use std::f64::consts::PI;

use crate::{Continuous, ContinuousDensityKernel};
use rand::Rng;
use rand::distr::Distribution;

use super::chi_squared::ChiSquared;
use crate::estimation::sample::{Summary, excess_kurtosis};
use crate::normal::standard_normal_polar;
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::Real;

/// Student's t-distribution with `ν > 0` degrees of freedom and an optional
/// **location** `μ` and **scale** `σ > 0`: `X = μ + σ·T`, where `T` is the standard
/// (location 0, scale 1) Student-t. The location-scale form is the posterior /
/// predictive distribution of the Normal–Inverse-Gamma conjugate model.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Cauchy, Continuous, Moments, Normal, StudentT};
///
/// // Heavier tailed than a normal, and converging to it as the degrees of
/// // freedom grow.
/// let t = StudentT::new(10.0)?;
/// assert!((t.variance() - 10.0 / 8.0).abs() < 1e-12);   // nu / (nu - 2)
/// assert!(t.pdf(3.0_f64) > Normal::new(0.0, 1.0)?.pdf(3.0_f64));
///
/// // Both ends of the family are named distributions: one degree of freedom is
/// // the Cauchy, and the limit is the standard normal.
/// assert!((StudentT::new(1.0)?.pdf(0.5_f64) - Cauchy::new(0.0, 1.0)?.pdf(0.5_f64)).abs() < 1e-12);
/// assert!((StudentT::new(1e7)?.pdf(0.5_f64) - Normal::new(0.0, 1.0)?.pdf(0.5_f64)).abs() < 1e-6);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudentT {
    freedom: f64,
    location: f64,
    scale: f64,
    chi: ChiSquared,
}

impl StudentT {
    /// The standard Student-t (location 0, scale 1) with `freedom` degrees.
    pub fn new(freedom: f64) -> Result<Self, BayesError> {
        Self::new_location_scale(freedom, 0.0, 1.0)
    }

    /// The location-scale Student-t `μ + σ·T_ν`.
    pub fn new_location_scale(freedom: f64, location: f64, scale: f64) -> Result<Self, BayesError> {
        validate::positive(&[("freedom", freedom)])?;
        if !location.is_finite() {
            return Err(BayesError::NonFinite {
                name: "location",
                value: location,
            });
        }
        validate::positive(&[("scale", scale)])?;
        Ok(Self {
            freedom,
            location,
            scale,
            chi: ChiSquared::new(freedom)?,
        })
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }
    pub fn location(&self) -> f64 {
        self.location
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Affine image `aX + b`.
    ///
    /// A zero scale coefficient would collapse the law to a point mass, which is
    /// outside this non-degenerate Student-t type.
    pub fn affine(self, scale: f64, shift: f64) -> Result<Self, BayesError> {
        if !scale.is_finite() {
            return Err(BayesError::NonFinite {
                name: "scale",
                value: scale,
            });
        }
        if !shift.is_finite() {
            return Err(BayesError::NonFinite {
                name: "shift",
                value: shift,
            });
        }
        if scale == 0.0 {
            return Err(BayesError::Malformed {
                what: "student-t affine scale must be non-zero",
            });
        }
        Self::new_location_scale(
            self.freedom,
            scale * self.location + shift,
            scale.abs() * self.scale,
        )
    }
}

impl Distribution<f64> for StudentT {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let z = standard_normal_polar(rng);
        let v = self.chi.sample(rng);
        self.location + self.scale * z / (v / self.freedom).sqrt()
    }
}

impl<T: Real> Support<T> for StudentT {
    fn min(&self) -> T {
        T::neg_infinity()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for StudentT {
    fn cdf(&self, x: T) -> T {
        kernels::students_t_cdf(
            x,
            T::scalar(self.location),
            T::scalar(self.scale),
            T::scalar(self.freedom),
        )
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(|x| self.cdf(x), T::neg_infinity(), T::infinity(), p)
    }

    /// Inverse survival by **reflection**, not by `inverse_cdf(1 − p)`.
    ///
    /// A Student's t is symmetric about its location, so `isf(p) = 2·location − Q(p)`
    /// holds exactly. That keeps `p` itself as the argument, where the generic
    /// default forms `1 − p` and loses every `p` below about `1e-16`.
    fn isf(&self, p: T) -> T {
        T::two() * T::scalar(self.location()) - self.inverse_cdf(p)
    }

    /// `ln(½·I_w(ν/2, ½))` above the location — the survival read straight off
    /// the incomplete beta, which keeps the `z^{−ν}` tail.
    fn ln_sf(&self, x: T) -> T {
        kernels::students_t_ln_sf(
            x,
            T::scalar(self.location),
            T::scalar(self.scale),
            T::scalar(self.freedom),
        )
    }
}

impl Moments for StudentT {
    fn mean(&self) -> f64 {
        if self.freedom > 1.0 {
            self.location
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        if self.freedom > 2.0 {
            self.scale * self.scale * self.freedom / (self.freedom - 2.0)
        } else {
            f64::NAN
        }
    }
    fn skewness(&self) -> f64 {
        if self.freedom > 3.0 { 0.0 } else { f64::NAN }
    }
    fn entropy(&self) -> f64 {
        let nu = self.freedom;
        let half_nu = 0.5 * nu;
        let half_nu_plus_one = 0.5 * (nu + 1.0);

        self.scale.ln() + 0.5 * (nu * PI).ln() + half_nu.ln_gamma() - half_nu_plus_one.ln_gamma()
            + half_nu_plus_one * (half_nu_plus_one.digamma() - half_nu.digamma())
    }
    fn median(&self) -> f64 {
        self.location
    }
    fn mode(&self) -> f64 {
        self.location
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn kurtosis(&self) -> f64 {
        f64::NAN
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for StudentT {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let freedom = T::scalar(self.freedom());
        let log_normalizer = T::half() * (freedom.clone() * T::pi()).ln()
            + (freedom.clone() * T::half()).ln_gamma()
            - ((freedom.clone() + T::one()) * T::half()).ln_gamma();
        kernels::student_t_log_pdf(
            x,
            T::scalar(self.location()),
            T::scalar(self.scale()),
            freedom,
            log_normalizer,
        )
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T: Real> Continuous<T> for StudentT {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

/// Method of moments — which for this family has to reach the **fourth**
/// moment, because the degrees of freedom are a tail index and the variance
/// alone cannot separate them from the scale.
///
/// The excess kurtosis of a Student's t is `6/(ν − 4)` regardless of location
/// and scale, so `ν̂ = 4 + 6/g₂`, and the scale then follows from the variance
/// as `scale² = v(ν − 2)/ν`.
///
/// Two caveats come with that, and both are properties of the family rather
/// than of this code. `None` is returned unless `g₂ > 0`, since a t is strictly
/// leptokurtic and a platykurtic sample argues against it. And `ν ≤ 4` is
/// unreachable — that is precisely where the kurtosis stops existing — so a
/// genuinely heavy-tailed sample will be fitted with too *thin* a tail. A
/// fourth moment is also far noisier to estimate than a mean, which makes this
/// the least precise estimator in the crate: expect it to need thousands of
/// observations to be worth as much as a log-normal fit gets from dozens.
impl crate::Estimate for StudentT {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::with_spread(data)?;
        let excess = excess_kurtosis(data)?;
        if excess <= 0.0 {
            return None;
        }
        let freedom = 4.0 + 6.0 / excess;
        let scale = (s.variance * (freedom - 2.0) / freedom).sqrt();
        StudentT::new_location_scale(freedom, s.mean, scale).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;

    #[test]
    fn symmetric_cdf_and_quantile() {
        let t = StudentT::new(10.0).unwrap();
        assert_close!(t.cdf(0.0), 0.5, 1e-12);
        assert_close!(t.cdf(-1.3), 1.0 - t.cdf(1.3), 1e-9);
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            assert_close!(t.cdf(t.inverse_cdf(p)), p, 1e-6);
        }
        assert_close!(t.inverse_cdf(0.975), 2.228_138_852, 1e-4);
    }

    #[test]
    fn heavy_tail_moments() {
        assert!(StudentT::new(1.0).unwrap().mean().is_nan());
        assert!(StudentT::new(2.0).unwrap().variance().is_nan());
        assert_close!(StudentT::new(5.0).unwrap().variance(), 5.0 / 3.0, 1e-12);
    }

    #[test]
    fn entropy_matches_cauchy_special_case() {
        let cauchy = StudentT::new_location_scale(1.0, 3.0, 2.0).unwrap();
        assert_close!(cauchy.entropy(), (8.0 * PI).ln(), 1e-10);
    }

    #[test]
    fn location_scale_shifts_and_scales() {
        let std = StudentT::new(8.0).unwrap();
        let ls = StudentT::new_location_scale(8.0, 3.0, 2.0).unwrap();
        let z = 0.7;
        assert_close!(ls.pdf(3.0 + 2.0 * z), std.pdf(z) / 2.0, 1e-12);
        assert_close!(ls.cdf(3.0 + 2.0 * z), std.cdf(z), 1e-12);
        assert_close!(ls.mean(), 3.0, 1e-12);
        assert_close!(ls.variance(), 4.0 * 8.0 / 6.0, 1e-12);
        assert_close!(ls.median(), 3.0, 1e-12);
        for &p in &[0.1, 0.5, 0.9] {
            assert_close!(ls.cdf(ls.inverse_cdf(p)), p, 1e-6);
        }

        let affine = ls.affine(-1.5, 4.0).unwrap();
        assert_close!(affine.freedom(), 8.0, 1e-12);
        assert_close!(affine.location(), -0.5, 1e-12);
        assert_close!(affine.scale(), 3.0, 1e-12);
        assert!(ls.affine(0.0, 1.0).is_err());
    }
}
