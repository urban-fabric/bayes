//! Shared test probes for the bayes workspace.
//!
//! **Dev-only.** `publish = false`, so nothing here reaches a consumer. It exists
//! because the suites had accumulated five copies of the same failure
//! accumulator and three of the same Simpson rule, and because a *canonical*
//! family sweep is worth more than per-suite lists: add a family to
//! [`families::continuous_cases`] once and the conformance, sampling, and
//! relationship suites all pick it up.
//!
//! Two halves:
//!
//! * [`probe`] — numerical machinery with no knowledge of bayes: failure
//!   accumulation, quadrature (including the endpoint-clustering substitution
//!   that singular quantile functions need), and Richardson-extrapolated
//!   derivatives.
//! * [`families`] — one uniform view over every family plus the canonical
//!   parameter sweep, so a property check is written once rather than per family.
//! * [`gof`] — goodness-of-fit probes for judging a *sampler*, the one part of a
//!   distribution no identity can pin down. Every threshold there is derived
//!   (a Kolmogorov series, a central-limit standard error) rather than tuned,
//!   because a hand-tuned sampler tolerance ends up asserting the current bias.
//!
//! This crate depends on `bayes-distribution-ext` while that crate dev-depends on
//! this one. The cycle is only through dev edges, which cargo permits.

// This crate is dev-only scaffolding, so a panic *is* the right failure mode: a
// sweep whose parameters have stopped being valid should fail loudly on the first
// call rather than quietly skip a family and report success over a shorter list.
// Every constructor below takes literal parameters that are valid by construction.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

pub mod families;
pub mod gof;
pub mod probe;

pub use families::{
    ContView, DiscSampleView, DiscView, continuous_cases, discrete_cases, discrete_sampler_cases,
};
pub use gof::{
    chi_squared_gof_pvalue, ks_pvalue, ks_statistic, mean_tolerance, variance_tolerance,
};
pub use probe::{
    DensityCase, Failures, derivative, integrate_unit, quantile_grid, rel_error, second_derivative,
    simpson,
};
