//! **Conformance**: each family against its own contract, under hostile input.
//!
//! The question here is not "do two families agree" — that is
//! `tests/relationships/` — but "is one family internally consistent, and does
//! it stay consistent when the inputs get hostile". The checks are properties
//! every distribution must satisfy whatever its parameters:
//!
//! ```text
//!   0 <= cdf <= 1,  non-decreasing,  saturating at the support bounds
//!   pdf >= 0,  ln_pdf == pdf.ln(),  sf == 1 - cdf,  ln_sf == sf.ln()
//!   cdf(inverse_cdf(p)) == p,  inverse_cdf monotone in p
//!   pdf == d(cdf)/dx                      (independent code paths)
//!   E[X], Var, H  ==  quantile-space integrals  (independent code paths)
//!   cdf(+-inf) saturates;  a NaN argument never yields a confident probability
//!   a constructor refuses NaN, infinity, and out-of-domain parameters
//! ```
//!
//! Parameters come from [`bayes_utils::continuous_cases`] and
//! [`bayes_utils::discrete_cases`], the canonical sweep — scales from `1e-9`
//! to `1e10`, shapes from `1e-3` to `1e4`, degenerate `p = 0` and `p = 1`, and a
//! location so far from the origin that the family's own quantiles stop being
//! representable. Adding a family there adds it here.
//!
//! Three things this suite detects **exactly** rather than papering over,
//! because they are limits of `f64` and not defects: an unrepresentable
//! quantile, a scale below one ulp at the distribution's location, and a
//! variance integral whose fourth moment is beyond the probe.
//!
//! | module | covers |
//! |--------|--------|
//! | [`continuous`] | the 23 continuous families, plus `VonMises` separately |
//! | [`discrete`] | the 9 discrete families, plus `DiscreteUniform` separately |
//! | [`spine`] | the trait spine itself: sampling, fitting, `Estimate` round-trips |
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod continuous;
mod discrete;
mod spine;
