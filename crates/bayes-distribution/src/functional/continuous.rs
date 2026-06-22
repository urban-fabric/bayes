//! Closure-backed continuous adapters.

use crate::{BayesError, Continuous, ContinuousCdf, Support};
use euclid::{Boolean, Real};
use std::fmt;

/// A closure-backed continuous density over generic support bounds.
///
/// The density is not normalized or validated by construction. In the lazy
/// crate, support validity is a value-level condition rather than a host `bool`.
#[derive(Clone, Copy)]
pub struct ContinuousDensity<R: Real = f64, P = fn(R) -> R, L = fn(R) -> R> {
    min: R,
    max: R,
    pdf: P,
    ln_pdf: Option<L>,
}

impl<R, P, L> fmt::Debug for ContinuousDensity<R, P, L>
where
    R: Real + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContinuousDensity")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

impl<R, P> ContinuousDensity<R, P, fn(R) -> R>
where
    R: Real,
{
    /// Build from a PDF and support bounds.
    pub fn from_pdf(min: R, max: R, pdf: P) -> Result<Self, BayesError> {
        Ok(Self {
            min,
            max,
            pdf,
            ln_pdf: None,
        })
    }
}

impl<R, P, L> ContinuousDensity<R, P, L>
where
    R: Real,
{
    /// Build from a PDF plus a numerically stable log-PDF.
    pub fn from_pdf_and_ln_pdf(min: R, max: R, pdf: P, ln_pdf: L) -> Result<Self, BayesError> {
        Ok(Self {
            min,
            max,
            pdf,
            ln_pdf: Some(ln_pdf),
        })
    }

    pub fn support_bounds(&self) -> (R, R) {
        (self.min.clone(), self.max.clone())
    }
}

impl<R, P, L> Support<R> for ContinuousDensity<R, P, L>
where
    R: Real,
{
    fn min(&self) -> R {
        self.min.clone()
    }

    fn max(&self) -> R {
        self.max.clone()
    }
}

impl<R, P, L> Continuous<R> for ContinuousDensity<R, P, L>
where
    R: Real,
    P: Fn(R) -> R,
    L: Fn(R) -> R,
{
    fn pdf(&self, x: R) -> R {
        let outside = x.lt(&self.min).or(&x.gt(&self.max));
        R::select(&outside, R::zero(), (self.pdf)(x))
    }

    fn ln_pdf(&self, x: R) -> R {
        let outside = x.lt(&self.min).or(&x.gt(&self.max));
        let body = match &self.ln_pdf {
            Some(ln_pdf) => ln_pdf(x),
            None => (self.pdf)(x).ln(),
        };
        R::select(&outside, R::neg_infinity(), body)
    }
}

/// A closure-backed continuous distribution with PDF, CDF, and quantile.
#[derive(Clone, Copy)]
pub struct ContinuousFn<
    R: Real = f64,
    P = fn(R) -> R,
    C = fn(R) -> R,
    Q = fn(R) -> R,
    L = fn(R) -> R,
> {
    density: ContinuousDensity<R, P, L>,
    cdf: C,
    inverse_cdf: Q,
}

impl<R, P, C, Q, L> fmt::Debug for ContinuousFn<R, P, C, Q, L>
where
    R: Real + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContinuousFn")
            .field("min", &self.density.min)
            .field("max", &self.density.max)
            .finish()
    }
}

impl<R, P, C, Q> ContinuousFn<R, P, C, Q, fn(R) -> R>
where
    R: Real,
{
    /// Build from PDF, CDF, quantile, and support bounds.
    pub fn new(min: R, max: R, pdf: P, cdf: C, inverse_cdf: Q) -> Result<Self, BayesError> {
        Ok(Self {
            density: ContinuousDensity::from_pdf(min, max, pdf)?,
            cdf,
            inverse_cdf,
        })
    }
}

impl<R, P, C, Q, L> ContinuousFn<R, P, C, Q, L>
where
    R: Real,
{
    /// Build from PDF, stable log-PDF, CDF, quantile, and support bounds.
    pub fn with_ln_pdf(
        min: R,
        max: R,
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

impl<R, P, C, Q, L> Support<R> for ContinuousFn<R, P, C, Q, L>
where
    R: Real,
{
    fn min(&self) -> R {
        self.density.min()
    }

    fn max(&self) -> R {
        self.density.max()
    }
}

impl<R, P, C, Q, L> Continuous<R> for ContinuousFn<R, P, C, Q, L>
where
    R: Real,
    P: Fn(R) -> R,
    L: Fn(R) -> R,
{
    fn pdf(&self, x: R) -> R {
        self.density.pdf(x)
    }

    fn ln_pdf(&self, x: R) -> R {
        self.density.ln_pdf(x)
    }
}

impl<R, P, C, Q, L> ContinuousCdf<R> for ContinuousFn<R, P, C, Q, L>
where
    R: Real,
    C: Fn(R) -> R,
    Q: Fn(R) -> R,
{
    fn cdf(&self, x: R) -> R {
        let below = x.le(&self.min());
        let above = x.ge(&self.max());
        R::select(
            &below,
            R::zero(),
            R::select(&above, R::one(), (self.cdf)(x)),
        )
    }

    fn inverse_cdf(&self, p: R) -> R {
        (self.inverse_cdf)(p)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn continuous_adapter_accepts_f32_real_surface() {
        let unit = ContinuousFn::new(
            0.0_f32,
            1.0_f32,
            |x: f32| {
                if (0.0..=1.0).contains(&x) { 1.0 } else { 0.0 }
            },
            |x: f32| x.clamp(0.0, 1.0),
            |p: f32| p.clamp(0.0, 1.0),
        )
        .unwrap();

        assert_eq!(unit.pdf(0.25_f32), 1.0);
        assert_eq!(unit.cdf(2.0_f32), 1.0);
        assert_eq!(unit.inverse_cdf(0.75_f32), 0.75);
    }
}
