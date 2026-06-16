//! Special functions underpinning the distributions' densities, CDFs, and
//! quantiles: the gamma family (Lanczos `ln_gamma`, regularized incomplete
//! gamma), the error function (`erf`/`erfc`/`erf_inv`, used by the Normal CDF and
//! quantile), the beta family (`ln_beta`, regularized incomplete beta),
//! multivariate gamma/beta, and combinatorics
//! (`factorial`/`ln_factorial`/`binomial`).
//!
//! Ported from the standard algorithms (Lanczos for gamma; Numerical-Recipes
//! continued fractions for the incomplete gamma/beta). `f64`-only for now.
//!
//! [`stable`] adds numerically stable log-domain helpers (`ln_sum_exp`,
//! `ln_mean_exp`, `softmax`, `log_softmax`, `ln_sub_exp`, `ln_1p_exp`,
//! `ln_1m_exp`, `xlogy`, `erfcx`) used for mixture/categorical log-densities,
//! normalized log-weights, and accurate distribution tails; [`bessel`] provides
//! the modified Bessel functions `I₀`/`I₁` (plus the stable ratio `I₁/I₀` and
//! `ln I₀`) for directional distributions.

pub mod bessel;
pub mod beta;
pub mod erf;
pub mod factorial;
pub mod gamma;
pub mod multivariate;
pub mod stable;

pub use bessel::{i0, i1, i1_over_i0, ln_i0};
pub use beta::{beta, beta_reg, ln_beta};
pub use erf::{erf, erf_inv, erfc};
pub use factorial::{binomial, factorial, ln_binomial, ln_factorial};
pub use gamma::{digamma, gamma, gamma_lr, gamma_ur, ln_gamma, trigamma};
pub use multivariate::{
    ln_multivariate_beta, ln_multivariate_gamma, multivariate_beta, multivariate_gamma,
};
pub use stable::{
    erfcx, ln_1m_exp, ln_1p_exp, ln_add_exp, ln_mean_exp, ln_sub_exp, ln_sum_exp, log_softmax,
    softmax, xlogy,
};
