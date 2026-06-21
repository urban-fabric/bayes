//! Lazy numeric probability traits.
//!
//! This crate is the low-level contract for write-once probability formulas over
//! [`euclid`] lazy numeric substrates. It deliberately excludes eager statistical
//! conveniences such as CDFs, moments, host-side support metadata, fitting, and
//! sampling.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

use euclid::{LazyInteger, LazyNatural, LazyReal};

/// Continuous probability density, the PDF `f(x)`.
pub trait Continuous<T: LazyReal = f64> {
    /// The density at `x`.
    fn pdf(&self, x: T) -> T;

    /// `ln f(x)`.
    fn ln_pdf(&self, x: T) -> T;
}

/// Lazy-real continuous density formula on the distribution's support.
///
/// This companion to [`Continuous`] intentionally does not perform support
/// checks or host-`bool` branching. It is the closed-form algebra for points
/// already known to lie on the continuous support.
pub trait ContinuousDensityKernel<T: LazyReal = f64> {
    /// `ln f(x)` over a [`LazyReal`] substrate.
    fn ln_pdf_kernel(&self, x: T) -> T;

    /// `f(x) = exp(ln f(x))` over the same [`LazyReal`] substrate.
    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

/// Discrete probability mass, the PMF `P(X = k)` over natural counts.
pub trait Discrete<K: LazyNatural = u64> {
    /// The mass at `k`.
    fn pmf(&self, k: K) -> K::Real;

    /// `ln P(X = k)`.
    fn ln_pmf(&self, k: K) -> K::Real;
}

/// Signed-integer discrete probability mass for distributions whose support
/// lives on `Z` rather than `N`.
pub trait IntegerDiscrete<K: LazyInteger = i64> {
    /// The mass at `k`.
    fn pmf(&self, k: K) -> K::Real;

    /// `ln P(X = k)`.
    fn ln_pmf(&self, k: K) -> K::Real;
}

/// Lazy-real discrete mass formula on a natural-count support.
///
/// Counts are represented by [`LazyNatural`] so formulas can use exact
/// combinatorial helpers before crossing into the associated [`LazyReal`]
/// analysis surface.
pub trait DiscreteMassKernel<K: LazyNatural = u64> {
    /// `ln P(X = k)` over a lazy natural count substrate.
    fn ln_pmf_kernel(&self, k: K) -> K::Real;

    /// `P(X = k) = exp(ln P(X = k))` over the associated lazy real substrate.
    fn pmf_kernel(&self, k: K) -> K::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

/// Lazy-real signed-integer mass formula on an integer support.
pub trait IntegerMassKernel<K: LazyInteger = i64> {
    /// `ln P(X = k)` over a lazy integer substrate.
    fn ln_pmf_kernel(&self, k: K) -> K::Real;

    /// `P(X = k) = exp(ln P(X = k))` over the associated lazy real substrate.
    fn pmf_kernel(&self, k: K) -> K::Real {
        self.ln_pmf_kernel(k).exp()
    }
}
