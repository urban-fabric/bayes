//! Closure-backed **continuous** adapters: lift a PDF (and optional log-PDF, CDF,
//! and quantile) into the [`Continuous`] / [`ContinuousCdf`] spine.

use crate::distribution::{Continuous, ContinuousCdf, Support};
use crate::error::BayesError;
use std::fmt;

fn valid_scalar_support(min: f64, max: f64) -> bool {
    !min.is_nan() && !max.is_nan() && min < max
}

fn invalid_scalar_support() -> BayesError {
    BayesError::Malformed {
        what: "support bounds must be ordered and non-NaN",
    }
}

/// A closure-backed continuous density over scalar support bounds.
///
/// This is useful for custom likelihoods and generic code that only needs
/// [`Continuous`] and [`Support`]. The density is not normalized or validated by
/// construction; callers are responsible for supplying a valid PDF/log-PDF pair.
#[derive(Clone, Copy)]
pub struct ContinuousDensity<P, L = fn(f64) -> f64> {
    min: f64,
    max: f64,
    pdf: P,
    ln_pdf: Option<L>,
}

impl<P, L> fmt::Debug for ContinuousDensity<P, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContinuousDensity")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

impl<P> ContinuousDensity<P, fn(f64) -> f64> {
    /// Build from a PDF and support bounds.
    pub fn from_pdf(min: f64, max: f64, pdf: P) -> Result<Self, BayesError> {
        if !valid_scalar_support(min, max) {
            return Err(invalid_scalar_support());
        }
        Ok(Self {
            min,
            max,
            pdf,
            ln_pdf: None,
        })
    }
}

impl<P, L> ContinuousDensity<P, L> {
    /// Build from a PDF plus a numerically stable log-PDF.
    pub fn from_pdf_and_ln_pdf(min: f64, max: f64, pdf: P, ln_pdf: L) -> Result<Self, BayesError> {
        if !valid_scalar_support(min, max) {
            return Err(invalid_scalar_support());
        }
        Ok(Self {
            min,
            max,
            pdf,
            ln_pdf: Some(ln_pdf),
        })
    }

    pub fn support_bounds(&self) -> (f64, f64) {
        (self.min, self.max)
    }
}

impl<P, L> Support for ContinuousDensity<P, L> {
    fn min(&self) -> f64 {
        self.min
    }
    fn max(&self) -> f64 {
        self.max
    }
}

impl<P, L> Continuous for ContinuousDensity<P, L>
where
    P: Fn(f64) -> f64,
    L: Fn(f64) -> f64,
{
    fn pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            0.0
        } else {
            (self.pdf)(x)
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            f64::NEG_INFINITY
        } else {
            match &self.ln_pdf {
                Some(ln_pdf) => ln_pdf(x),
                None => self.pdf(x).ln(),
            }
        }
    }
}

/// A closure-backed continuous distribution with PDF, CDF, and quantile.
#[derive(Clone, Copy)]
pub struct ContinuousFn<P, C, Q, L = fn(f64) -> f64> {
    density: ContinuousDensity<P, L>,
    cdf: C,
    inverse_cdf: Q,
}

impl<P, C, Q, L> fmt::Debug for ContinuousFn<P, C, Q, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContinuousFn")
            .field("min", &self.density.min)
            .field("max", &self.density.max)
            .finish()
    }
}

impl<P, C, Q> ContinuousFn<P, C, Q, fn(f64) -> f64> {
    /// Build from PDF, CDF, quantile, and support bounds.
    pub fn new(min: f64, max: f64, pdf: P, cdf: C, inverse_cdf: Q) -> Result<Self, BayesError> {
        Ok(Self {
            density: ContinuousDensity::from_pdf(min, max, pdf)?,
            cdf,
            inverse_cdf,
        })
    }
}

impl<P, C, Q, L> ContinuousFn<P, C, Q, L> {
    /// Build from PDF, stable log-PDF, CDF, quantile, and support bounds.
    pub fn with_ln_pdf(
        min: f64,
        max: f64,
        pdf: P,
        ln_pdf: L,
        cdf: C,
        inverse_cdf: Q,
    ) -> Result<Self, BayesError> {
        Ok(Self {
            density: ContinuousDensity::from_pdf_and_ln_pdf(min, max, pdf, ln_pdf)?,
            cdf,
            inverse_cdf,
        })
    }
}

impl<P, C, Q, L> Support for ContinuousFn<P, C, Q, L> {
    fn min(&self) -> f64 {
        self.density.min()
    }
    fn max(&self) -> f64 {
        self.density.max()
    }
}

impl<P, C, Q, L> Continuous for ContinuousFn<P, C, Q, L>
where
    P: Fn(f64) -> f64,
    L: Fn(f64) -> f64,
{
    fn pdf(&self, x: f64) -> f64 {
        self.density.pdf(x)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.density.ln_pdf(x)
    }
}

impl<P, C, Q, L> ContinuousCdf for ContinuousFn<P, C, Q, L>
where
    C: Fn(f64) -> f64,
    Q: Fn(f64) -> f64,
{
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.min() {
            0.0
        } else if x >= self.max() {
            1.0
        } else {
            (self.cdf)(x)
        }
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        (self.inverse_cdf)(p)
    }
}
