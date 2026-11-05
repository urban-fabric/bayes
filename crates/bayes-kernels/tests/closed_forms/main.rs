//! **Closed forms**: the kernels on their own terms.
//!
//! `bayes-kernels` holds every density, CDF, quantile and divergence in the
//! workspace, written once against `euclid_numeric::Real` so one formula serves
//! every carrier. It is the crate where a wrong constant does the most damage
//! and, until this suite, the only large one with no tests of its own — its
//! coverage came entirely from `bayes-distribution-ext/tests/kernels/`, which
//! asks a different question: *do the kernels agree with the eager families
//! that wrap them?* Two layers reading the same wrong formula agree perfectly.
//!
//! So nothing here goes through a family.
//!
//! | module | asks |
//! |--------|------|
//! | [`reductions`] | do the families that coincide actually coincide? |
//! | [`consistency`] | does each kernel agree with its own logarithm and complement? |
//! | [`stability`] | does anything produce a `NaN`, or an overflow, where a finite answer exists? |
//!
//! [`reductions`] is the load-bearing one. Every other check here is internal:
//! a formula can satisfy `ln_cdf == ln(cdf)` while being the wrong
//! distribution. But `Gamma(1, λ)` *is* `Exponential(λ)`, and `ChiSquared(2)`
//! *is* `Exponential(½)`, and those are identities between **separately
//! written** formulas — one built from an incomplete gamma, the other from an
//! exponential. They agree only if both are right.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod consistency;
mod reductions;
mod stability;
mod table;

/// Relative error, floored so a comparison against zero does not divide by it.
pub(crate) fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}
