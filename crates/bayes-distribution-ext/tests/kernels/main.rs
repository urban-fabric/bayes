//! **Kernels against families**: the carrier-generic closed forms in
//! `bayes-kernels` against the eager families that are supposed to implement
//! them.
//!
//! This is the load-bearing suite for the library's central claim. `bayes-kernels`
//! writes each density, CDF and divergence once against `euclid_numeric::Real`
//! so it runs on any carrier; `bayes-distribution-ext` wraps those with
//! validation, support handling and host branching. If the two ever disagree,
//! the claim that they are "the same closed form reached through two doors" is
//! false — and a model whose log-likelihood is staged through the kernels would
//! silently stop matching the library's own density.
//!
//! The suite lives at this level because of the dependency graph: the kernels sit
//! *below* the families, so nothing in `bayes-kernels` can see a `Normal` to
//! compare against. Only a test above both can.
//!
//! | module | covers |
//! |--------|--------|
//! | [`density`] | `*_log_pdf` / `*_log_pmf` kernels vs `Continuous` / `Discrete` |
//! | [`cdf_entropy`] | `*_cdf`, `*_quantile` and entropy kernels vs the families |
//! | [`conjugate`] | conjugate-update kernels vs the `Conjugate` models |
//! | [`divergence`] | `kl_*`, `hellinger_*`, `chi_squared_*` kernels vs the traits |
//! | [`misc`] | the remainder: Bessel-backed, stable-tail and transform kernels |
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod cdf_entropy;
mod conjugate;
mod density;
mod divergence;
mod misc;
