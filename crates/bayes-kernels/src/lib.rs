//! Euclid-generic closed-form probability kernels.
//!
//! This module is the `LazyReal`/`LazyComplex` companion to the concrete `f64`
//! distribution spine. The functions here are pure closed-form kernels: they do
//! not validate parameters, check support membership, sample, branch on CDFs, or
//! perform eager special-function work. Scalar constants are lifted through
//! `LazyReal::scalar`; callers provide precomputed special-function values such
//! as `ln Gamma(shape)` or `ln B(alpha, beta)`.
//!
//! That shape is intentional. Euclid's lazy traits are for tensor/expression
//! backends such as `lebesgue`, where comparisons and container construction may
//! not exist, but algebraic log-density formulas should still be reusable.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod primitives;

use euclid::numeric::{LazyInteger, LazyReal};

pub use crate::primitives::{
    complex_normal_log_prob_with_normalizer, complex_residual_norm_sqr, normal_log_kernel,
    normal_log_prob, normal_log_prob_with_constants, normal_log_prob_with_normalizer,
};

/// Evaluate `exp(ln_density)`.
pub fn density_from_log<T: LazyReal>(ln_density: T) -> T {
    ln_density.exp()
}

/// Square a lazy scalar.
pub fn square<T: LazyReal>(x: T) -> T {
    x.clone() * x
}

/// `ln I₀(x) = ln I₀ᵉ(x) + |x|` — the overflow-free Bessel log shared by the von Mises
/// kernels, spelled over the [`LazyReal::ln_bessel_i0`] method.
fn ln_i0<T: LazyReal>(x: T) -> T {
    x.ln_bessel_i0()
}

/// `H² = (1 − BC).max(0)` — squared Hellinger from a Bhattacharyya coefficient.
/// Shared by the per-family Hellinger kernels and the categorical reduction.
fn hellinger_sq_from_bc<T: LazyReal>(bc: T) -> T {
    (T::one() - bc).max(T::zero())
}

// The kernel library is partitioned by kind into the submodules below; every
// item is re-exported flat, so callers reference `bayes_kernels::<fn>` regardless
// of which file a kernel lives in.
mod cdf;
mod correlation;
mod density;
mod divergence_chi_squared;
mod divergence_hellinger;
mod divergence_kl;
mod divergence_renyi;
mod entropy;
mod glm;
mod marginal_likelihood;
mod moments;
mod quantile;
mod transform;

pub use cdf::*;
pub use correlation::*;
pub use density::*;
pub use divergence_chi_squared::*;
pub use divergence_hellinger::*;
pub use divergence_kl::*;
pub use divergence_renyi::*;
pub use entropy::*;
pub use glm::*;
pub use marginal_likelihood::*;
pub use moments::*;
pub use quantile::*;
pub use transform::*;
