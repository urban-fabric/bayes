use std::f64::consts::PI;

use rand::Rng;
use rand::distr::Distribution;

use super::chi_squared::ChiSquared;
use crate::distribution::normal::standard_normal_polar;
use crate::distribution::numeric::inverse_cdf_bisect;
use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::function::beta::beta_reg;
use crate::function::gamma::{digamma, ln_gamma};

/// Student's t-distribution with `ν > 0` degrees of freedom and an optional
/// **location** `μ` and **scale** `σ > 0`: `X = μ + σ·T`, where `T` is the standard
/// (location 0, scale 1) Student-t. The location-scale form is the posterior /
/// predictive distribution of the Normal–Inverse-Gamma conjugate model.
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
        if !location.is_finite() {
            return Err(BayesError::NonFinite {
                name: "location",
                value: location,
            });
        }
        if !scale.is_finite() {
            return Err(BayesError::NonFinite {
                name: "scale",
                value: scale,
            });
        }
        if scale <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "scale",
                value: scale,
            });
        }
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

impl Continuous for StudentT {
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        let nu = self.freedom;
        let z = (x - self.location) / self.scale;
        ln_gamma((nu + 1.0) / 2.0)
            - ln_gamma(nu / 2.0)
            - 0.5 * (nu * PI).ln()
            - (nu + 1.0) / 2.0 * (1.0 + z * z / nu).ln()
            - self.scale.ln()
    }
}

impl Support for StudentT {
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for StudentT {
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        if z == 0.0 {
            return 0.5;
        }
        let nu = self.freedom;
        let ix = beta_reg(nu / 2.0, 0.5, nu / (nu + z * z));
        if z > 0.0 { 1.0 - 0.5 * ix } else { 0.5 * ix }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        inverse_cdf_bisect(|x| self.cdf(x), f64::NEG_INFINITY, f64::INFINITY, p)
    }
}

impl Moments for StudentT {
    fn mean(&self) -> Option<f64> {
        (self.freedom > 1.0).then_some(self.location)
    }
    fn variance(&self) -> Option<f64> {
        if self.freedom > 2.0 {
            Some(self.scale * self.scale * self.freedom / (self.freedom - 2.0))
        } else {
            None
        }
    }
    fn skewness(&self) -> Option<f64> {
        (self.freedom > 3.0).then_some(0.0)
    }
    fn entropy(&self) -> Option<f64> {
        let nu = self.freedom;
        let half_nu = 0.5 * nu;
        let half_nu_plus_one = 0.5 * (nu + 1.0);
        Some(
            self.scale.ln() + 0.5 * (nu * PI).ln() + ln_gamma(half_nu) - ln_gamma(half_nu_plus_one)
                + half_nu_plus_one * (digamma(half_nu_plus_one) - digamma(half_nu)),
        )
    }
    fn median(&self) -> Option<f64> {
        Some(self.location)
    }
    fn mode(&self) -> Option<f64> {
        Some(self.location)
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
    fn symmetric_cdf_and_quantile() {
        let t = StudentT::new(10.0).unwrap();
        close(t.cdf(0.0), 0.5, 1e-12);
        close(t.cdf(-1.3), 1.0 - t.cdf(1.3), 1e-9);
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            close(t.cdf(t.inverse_cdf(p)), p, 1e-6);
        }
        close(t.inverse_cdf(0.975), 2.228_138_852, 1e-4);
    }

    #[test]
    fn heavy_tail_moments() {
        assert_eq!(StudentT::new(1.0).unwrap().mean(), None);
        assert_eq!(StudentT::new(2.0).unwrap().variance(), None);
        close(
            StudentT::new(5.0).unwrap().variance().unwrap(),
            5.0 / 3.0,
            1e-12,
        );
    }

    #[test]
    fn entropy_matches_cauchy_special_case() {
        let cauchy = StudentT::new_location_scale(1.0, 3.0, 2.0).unwrap();
        close(cauchy.entropy().unwrap(), (8.0 * PI).ln(), 1e-10);
    }

    #[test]
    fn location_scale_shifts_and_scales() {
        let std = StudentT::new(8.0).unwrap();
        let ls = StudentT::new_location_scale(8.0, 3.0, 2.0).unwrap();
        let z = 0.7;
        close(ls.pdf(3.0 + 2.0 * z), std.pdf(z) / 2.0, 1e-12);
        close(ls.cdf(3.0 + 2.0 * z), std.cdf(z), 1e-12);
        close(ls.mean().unwrap(), 3.0, 1e-12);
        close(ls.variance().unwrap(), 4.0 * 8.0 / 6.0, 1e-12);
        close(ls.median().unwrap(), 3.0, 1e-12);
        for &p in &[0.1, 0.5, 0.9] {
            close(ls.cdf(ls.inverse_cdf(p)), p, 1e-6);
        }

        let affine = ls.affine(-1.5, 4.0).unwrap();
        close(affine.freedom(), 8.0, 1e-12);
        close(affine.location(), -0.5, 1e-12);
        close(affine.scale(), 3.0, 1e-12);
        assert!(ls.affine(0.0, 1.0).is_err());
    }
}
