//! Euclid-generic closed-form probability kernels.
//!
//! This module is the `Real`/`Complex` companion to the concrete `f64`
//! distribution spine. The functions here are pure closed-form kernels: they do
//! not validate parameters, check support membership, sample, branch on CDFs, or
//! perform eager special-function work. Scalar constants are lifted through
//! `Real::scalar`; callers provide precomputed special-function values such
//! as `ln Gamma(shape)` or `ln B(alpha, beta)`.
//!
//! That shape is intentional. Euclid's lazy traits are for tensor/expression
//! backends such as `lebesgue`, where comparisons and container construction may
//! not exist, but algebraic log-density formulas should still be reusable.
//!
//! # How to read this crate
//!
//! The root remains a flat compatibility surface:
//! `bayes_kernels::normal_log_pdf(...)` and the existing function names keep
//! working. For browsing, use the public operation modules:
//! [`density`], [`cdf`], [`quantile`], [`entropy`], [`moments`],
//! [`divergence`], [`marginal_likelihood`], [`transform`], [`correlation`],
//! and [`glm`].
//!
//! If you are thinking in distributions instead of operations, start in
//! [`families`]. The implementation files live under
//! `families/continuous/*.rs` and `families/discrete/*.rs`, with modules such
//! as `families::normal::*` and `families::poisson::*`. The operation modules
//! are now short re-export views over those family implementations.
//!
//! Naming convention: `_on_support` and `_on_unit` kernels assume the caller has
//! already checked support membership. Full-domain wrappers encode support
//! behavior when that can be expressed with Euclid's generic scalar operations.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod primitives;

use euclid::numeric::Real;

pub use crate::primitives::{
    complex_normal_log_prob_with_normalizer, complex_residual_norm_sqr, normal_log_kernel,
    normal_log_prob, normal_log_prob_with_constants, normal_log_prob_with_normalizer,
};

/// `ln I₀(x) = ln I₀ᵉ(x) + |x|` — the overflow-free Bessel log shared by the von Mises
/// kernels, spelled over the [`Real::ln_bessel_i0`] method.
fn ln_i0<T: Real>(x: T) -> T {
    x.ln_bessel_i0()
}

/// `H² = (1 − BC).max(0)` — squared Hellinger from a Bhattacharyya coefficient.
/// Shared by the per-family Hellinger kernels and the categorical reduction.
fn hellinger_sq_from_bc<T: Real>(bc: T) -> T {
    (T::one() - bc).max(T::zero())
}

// Families are the implementation home. Operation modules are public browsing
// views, and the flat root re-exports below preserve the older
// `bayes_kernels::<fn>` surface.
pub mod cdf;
pub mod correlation;
pub mod density;
mod divergence_chi_squared;
mod divergence_hellinger;
mod divergence_kl;
mod divergence_renyi;
pub mod entropy;
pub mod glm;
pub mod marginal_likelihood;
pub mod moments;
pub mod quantile;
pub mod transform;

pub mod families;

/// Divergence kernels grouped by divergence family.
///
/// The root still exports each divergence function directly. This module gives
/// callers a narrower path when they want all KL, Hellinger, chi-squared, or
/// Renyi kernels together.
pub mod divergence {
    /// Chi-squared divergence kernels.
    pub mod chi_squared {
        pub use crate::divergence_chi_squared::*;
    }

    /// Hellinger distance kernels.
    pub mod hellinger {
        pub use crate::divergence_hellinger::*;
    }

    /// Kullback-Leibler divergence kernels.
    pub mod kl {
        pub use crate::divergence_kl::*;
    }

    /// Renyi divergence kernels.
    pub mod renyi {
        pub use crate::divergence_renyi::*;
    }

    pub use chi_squared::*;
    pub use hellinger::*;
    pub use kl::*;
    pub use renyi::*;
}

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
