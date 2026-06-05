//! Functional adapters and combinators for the distribution trait spine.
//!
//! The concrete distribution types cover common named families. This module covers
//! the other functional use case: lift pure functions into the same traits, then
//! compose them with log-likelihood, interval probability, truncation, and
//! bijective transforms.

use crate::distribution::{
    Affine, Bijector, Continuous, ContinuousCdf, Discrete, DiscreteCdf, Exp, Sigmoid, Support,
    Transformed, Truncated,
};
use crate::error::BayesError;
use crate::function::stable::ln_sub_exp;
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

/// A closure-backed discrete mass function over explicit support bounds.
#[derive(Clone, Copy)]
pub struct DiscreteMass<X, P, L = fn(X) -> f64> {
    min: X,
    max: X,
    pmf: P,
    ln_pmf: Option<L>,
}

impl<X, P, L> fmt::Debug for DiscreteMass<X, P, L>
where
    X: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscreteMass")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

impl<X, P> DiscreteMass<X, P, fn(X) -> f64> {
    /// Build from a PMF and support bounds.
    pub fn from_pmf(min: X, max: X, pmf: P) -> Self {
        Self {
            min,
            max,
            pmf,
            ln_pmf: None,
        }
    }
}

impl<X, P, L> DiscreteMass<X, P, L> {
    /// Build from a PMF plus a numerically stable log-PMF.
    pub fn from_pmf_and_ln_pmf(min: X, max: X, pmf: P, ln_pmf: L) -> Self {
        Self {
            min,
            max,
            pmf,
            ln_pmf: Some(ln_pmf),
        }
    }
}

impl<X, P, L> Support<X> for DiscreteMass<X, P, L>
where
    X: Copy,
{
    fn min(&self) -> X {
        self.min
    }
    fn max(&self) -> X {
        self.max
    }
}

impl<X, P, L> Discrete<X> for DiscreteMass<X, P, L>
where
    X: Copy,
    P: Fn(X) -> f64,
    L: Fn(X) -> f64,
{
    fn pmf(&self, k: X) -> f64 {
        (self.pmf)(k)
    }

    fn ln_pmf(&self, k: X) -> f64 {
        match &self.ln_pmf {
            Some(ln_pmf) => ln_pmf(k),
            None => self.pmf(k).ln(),
        }
    }
}

/// A closure-backed discrete distribution with PMF, CDF, and quantile.
#[derive(Clone, Copy)]
pub struct DiscreteFn<X, P, C, Q, L = fn(X) -> f64> {
    mass: DiscreteMass<X, P, L>,
    cdf: C,
    inverse_cdf: Q,
}

impl<X, P, C, Q, L> fmt::Debug for DiscreteFn<X, P, C, Q, L>
where
    X: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscreteFn")
            .field("min", &self.mass.min)
            .field("max", &self.mass.max)
            .finish()
    }
}

impl<X, P, C, Q> DiscreteFn<X, P, C, Q, fn(X) -> f64> {
    /// Build from PMF, CDF, quantile, and support bounds.
    pub fn new(min: X, max: X, pmf: P, cdf: C, inverse_cdf: Q) -> Self {
        Self {
            mass: DiscreteMass::from_pmf(min, max, pmf),
            cdf,
            inverse_cdf,
        }
    }
}

impl<X, P, C, Q, L> DiscreteFn<X, P, C, Q, L> {
    /// Build from PMF, stable log-PMF, CDF, quantile, and support bounds.
    pub fn with_ln_pmf(min: X, max: X, pmf: P, ln_pmf: L, cdf: C, inverse_cdf: Q) -> Self {
        Self {
            mass: DiscreteMass::from_pmf_and_ln_pmf(min, max, pmf, ln_pmf),
            cdf,
            inverse_cdf,
        }
    }
}

impl<X, P, C, Q, L> Support<X> for DiscreteFn<X, P, C, Q, L>
where
    X: Copy,
{
    fn min(&self) -> X {
        self.mass.min()
    }
    fn max(&self) -> X {
        self.mass.max()
    }
}

impl<X, P, C, Q, L> Discrete<X> for DiscreteFn<X, P, C, Q, L>
where
    X: Copy,
    P: Fn(X) -> f64,
    L: Fn(X) -> f64,
{
    fn pmf(&self, k: X) -> f64 {
        self.mass.pmf(k)
    }

    fn ln_pmf(&self, k: X) -> f64 {
        self.mass.ln_pmf(k)
    }
}

impl<X, P, C, Q, L> DiscreteCdf<X> for DiscreteFn<X, P, C, Q, L>
where
    X: Copy + PartialOrd,
    C: Fn(X) -> f64,
    Q: Fn(f64) -> X,
{
    fn cdf(&self, k: X) -> f64 {
        if k < self.min() {
            0.0
        } else if k >= self.max() {
            1.0
        } else {
            (self.cdf)(k)
        }
    }

    fn inverse_cdf(&self, p: f64) -> X {
        (self.inverse_cdf)(p)
    }
}

/// Functional operations for any continuous distribution.
pub trait ContinuousDistributionExt: Continuous + Sized {
    /// Log-likelihood `Σ ln f(xᵢ)`. Empty samples have log-likelihood `0`.
    fn ln_likelihood(&self, data: &[f64]) -> f64 {
        data.iter().map(|&x| self.ln_pdf(x)).sum()
    }

    /// Likelihood `Π f(xᵢ)`, computed as `exp(log-likelihood)`.
    fn likelihood(&self, data: &[f64]) -> f64 {
        self.ln_likelihood(data).exp()
    }

    /// Push the distribution through a bijector.
    fn transformed<B: Bijector>(self, bijector: B) -> Transformed<Self, B> {
        Transformed::new(self, bijector)
    }

    /// Shift and scale the distribution by `scale * x + shift`.
    fn affine(self, scale: f64, shift: f64) -> Result<Transformed<Self, Affine>, BayesError> {
        Ok(self.transformed(Affine::new(scale, shift)?))
    }

    /// Push through `exp(x)`.
    fn exp(self) -> Transformed<Self, Exp> {
        self.transformed(Exp)
    }

    /// Push through the logistic sigmoid.
    fn sigmoid(self) -> Transformed<Self, Sigmoid> {
        self.transformed(Sigmoid)
    }
}

impl<D> ContinuousDistributionExt for D where D: Continuous + Sized {}

/// Functional operations for any continuous distribution with a CDF.
pub trait ContinuousCdfExt: ContinuousCdf + Sized {
    /// Probability mass in `(lo, hi]`.
    fn probability_between(&self, lo: f64, hi: f64) -> f64 {
        if lo > hi || lo.is_nan() || hi.is_nan() {
            return f64::NAN;
        }
        let p = self.cdf(hi) - self.cdf(lo);
        if p.is_nan() { f64::NAN } else { p.max(0.0) }
    }

    /// Log probability mass in `(lo, hi]`, using CDF or survival logs to reduce
    /// cancellation in tails.
    fn ln_probability_between(&self, lo: f64, hi: f64) -> f64 {
        if lo > hi || lo.is_nan() || hi.is_nan() {
            return f64::NAN;
        }
        let cdf_hi = self.cdf(hi);
        if cdf_hi.is_nan() {
            return f64::NAN;
        }
        if cdf_hi <= 0.5 {
            ln_sub_exp(self.ln_cdf(hi), self.ln_cdf(lo))
        } else {
            ln_sub_exp(self.ln_sf(lo), self.ln_sf(hi))
        }
    }

    /// Truncate and renormalize to `[lo, hi]`.
    fn truncated(self, lo: f64, hi: f64) -> Result<Truncated<Self>, BayesError> {
        Truncated::new(self, lo, hi)
    }
}

impl<D> ContinuousCdfExt for D where D: ContinuousCdf + Sized {}

/// Functional operations for any discrete distribution.
pub trait DiscreteDistributionExt<X>: Discrete<X>
where
    X: Copy,
{
    /// Log-likelihood `Σ ln P(X = xᵢ)`. Empty samples have log-likelihood `0`.
    fn ln_likelihood(&self, data: &[X]) -> f64 {
        data.iter().map(|&x| self.ln_pmf(x)).sum()
    }

    /// Likelihood `Π P(X = xᵢ)`, computed as `exp(log-likelihood)`.
    fn likelihood(&self, data: &[X]) -> f64 {
        self.ln_likelihood(data).exp()
    }
}

impl<D, X> DiscreteDistributionExt<X> for D
where
    D: Discrete<X>,
    X: Copy,
{
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn closure_backed_continuous_distribution_implements_the_spine() {
        let unit = ContinuousFn::new(
            0.0,
            1.0,
            |x| if (0.0..=1.0).contains(&x) { 1.0 } else { 0.0 },
            |x: f64| x.clamp(0.0, 1.0),
            |p: f64| p.clamp(0.0, 1.0),
        )
        .unwrap();

        close(unit.pdf(0.25), 1.0, 1e-12);
        close(unit.cdf(-1.0), 0.0, 1e-12);
        close(unit.cdf(0.25), 0.25, 1e-12);
        close(unit.cdf(2.0), 1.0, 1e-12);
        close(unit.inverse_cdf(0.75), 0.75, 1e-12);
        close(unit.ln_likelihood(&[0.2, 0.4, 0.8]), 0.0, 1e-12);
        close(unit.probability_between(0.2, 0.7), 0.5, 1e-12);
        close(unit.ln_probability_between(0.2, 0.7).exp(), 0.5, 1e-12);
    }

    #[test]
    fn interval_probability_propagates_malformed_cdfs() {
        let bad = ContinuousFn::new(0.0, 1.0, |_: f64| 1.0, |_: f64| f64::NAN, |p: f64| p).unwrap();

        assert!(bad.probability_between(0.2, 0.7).is_nan());
        assert!(bad.ln_probability_between(0.2, 0.7).is_nan());
    }

    #[test]
    fn extension_combinators_apply_existing_distribution_adapters() {
        let unit = ContinuousFn::new(
            0.0,
            1.0,
            |_| 1.0,
            |x: f64| x.clamp(0.0, 1.0),
            |p: f64| p.clamp(0.0, 1.0),
        )
        .unwrap();

        let shifted = unit.affine(2.0, -1.0).unwrap();
        close(shifted.pdf(0.0), 0.5, 1e-12);

        let truncated = unit.truncated(0.25, 0.75).unwrap();
        close(truncated.pdf(0.5), 2.0, 1e-12);
        close(truncated.cdf(0.5), 0.5, 1e-12);
    }

    #[test]
    fn closure_backed_discrete_distribution_implements_the_spine() {
        let coin = DiscreteFn::new(
            0u64,
            1u64,
            |k: u64| match k {
                0 => 0.25,
                1 => 0.75,
                _ => 0.0,
            },
            |k: u64| if k == 0 { 0.25 } else { 1.0 },
            |p: f64| if p <= 0.25 { 0 } else { 1 },
        );

        close(coin.pmf(1), 0.75, 1e-12);
        close(coin.cdf(0), 0.25, 1e-12);
        close(coin.cdf(7), 1.0, 1e-12);
        close(coin.inverse_cdf(0.9) as f64, 1.0, 1e-12);
        close(
            coin.ln_likelihood(&[1, 1, 0]).exp(),
            0.75 * 0.75 * 0.25,
            1e-12,
        );
    }
}
