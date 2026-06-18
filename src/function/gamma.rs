//! The gamma function family — now owned by [`euclid::special`] and re-exported
//! here so the `f64` distribution spine and the `LazyReal` lazy kernels share one
//! implementation.
//!
//! The Lanczos `ln_gamma`/`gamma`, the regularized incomplete gamma
//! `gamma_lr`/`gamma_ur` (the Gamma / chi-squared / Poisson CDF kernels), and the
//! polygammas `digamma`/`trigamma` are the concrete reference behind the
//! [`euclid::numeric::LazyReal`] special-function methods.

pub use euclid::special::gamma::{digamma, gamma, gamma_lr, gamma_ur, ln_gamma, trigamma};
