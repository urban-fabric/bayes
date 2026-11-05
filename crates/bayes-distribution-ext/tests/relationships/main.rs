//! The **relationship network**: every family checked against the other families
//! it is provably related to.
//!
//! This suite is organized by the *kind of relation* under test rather than by
//! the family under test, so one family is constrained from several directions at
//! once. The point is that the two sides of each relation share no code — a
//! `ChiSquared` CDF is a regularized incomplete gamma, a `Poisson` CDF is a
//! finite sum of log-factorials — so an identity between them is an independent
//! check of both. One edge in this network is worth more than a dozen
//! hand-transcribed reference values, because a reference value can be copied
//! wrongly and an identity cannot.
//!
//! The modules:
//!
//! | module | asks |
//! |--------|------|
//! | [`families`] | exact reparameterizations: is `ChiSquared(k)` really `Gamma(k/2, ½)`? |
//! | [`bridges`] | the discrete↔continuous CDF identities (`Poisson` tail = `ChiSquared` tail) |
//! | [`limits`] | limiting cases: `Binomial(n, λ/n) → Poisson(λ)`, `StudentT(∞) → Normal` |
//! | [`generating`] | `mgf`/`cgf`/`cf`/`pgf`/L-moments against the moments they generate |
//! | [`conjugate`] | closed-form posteriors against Bayes' rule applied numerically |
//!
//! Growing the network is the way to raise confidence: a new function should gain
//! at least one edge here — a limit, a reparameterization, a derivative, or a
//! discrete↔continuous bridge.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod bridges;
mod conjugate;
mod families;
mod generating;
mod limits;

pub use bayes_utils::probe::{
    DensityCase, Failures, derivative, integrate_unit, quantile_grid, rel_error, second_derivative,
    simpson,
};
