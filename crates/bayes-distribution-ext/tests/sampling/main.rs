//! **Samplers against their own distributions.**
//!
//! Every other suite in this directory rests on identities: the density, CDF,
//! quantile and moments all constrain each other, so a wrong constant anywhere
//! has to be wrong consistently in several places to survive. The sampler has no
//! such protection. It is a *separate algorithm* — Marsaglia–Tsang for a gamma,
//! inverse-CDF for a Weibull, a ziggurat for a normal — and nothing in the
//! closed-form surface notices if it draws from the wrong law. A sampler that
//! quietly returns `Exponential(1/rate)` where `Exponential(rate)` was asked for
//! passes every test in `conformance/` and `relationships/`.
//!
//! So this suite is the only one that is **statistical**, and that makes the
//! tolerances the delicate part. A threshold tuned until the suite passes is a
//! threshold that asserts the sampler's present bias. Every bound here is
//! instead derived from theory, in [`bayes_utils::gof`]:
//!
//! * the Kolmogorov–Smirnov p-value is a self-contained series, sharing no code
//!   with what it judges
//! * a sample-mean bound is `sigmas·√(Var/n)` — the central-limit standard
//!   error, so the chance of a correct sampler failing is stated, not hoped for
//! * a sample-variance bound uses the fourth moment, and declines to exist when
//!   the kurtosis does not
//!
//! Seeds are fixed, so a failure reproduces exactly rather than appearing once a
//! week in someone else's CI.
//!
//! | module | asks |
//! |--------|------|
//! | [`continuous`] | do draws follow the family's own CDF? (KS, moments, support) |
//! | [`discrete`] | do draws follow the family's own PMF? (chi-squared, moments) |
//! | [`moments`] | analytic moments against large seeded draws, family by family |
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod continuous;
mod discrete;
mod moments;

/// The seed every sweep starts from. One constant so a reproduction is one
/// number, and an obviously arbitrary one so nobody mistakes it for tuned.
pub const SEED: u64 = 20_260_105;
