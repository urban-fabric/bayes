//! Scalar-generic probability primitives.
//!
//! The kernel-level closed forms are written over Euclid's lazy analysis traits:
//! [`LazyReal`] for real-valued formulas and [`LazyComplex`] for complex-valued
//! residuals. Scalar constants come from [`LazyReal::scalar`] and its helper
//! methods, while precomputed special-function normalizers remain caller inputs.
//! Concrete callers such as `f64`/`f32` satisfy the same lazy formulas through
//! [`euclid::numeric::Real`].
//!
//! These mirror selected hot closed forms; the owning distribution types (e.g.
//! [`Normal`](crate::Normal)) remain the semantic authority, and their scalar
//! methods are parity-tested against these generic primitives.

use euclid::numeric::{LazyComplex, LazyReal};

/// Gaussian quadratic log-kernel `-½·((x-µ)/σ)²`, written over the lazy real
/// analysis surface.
pub fn normal_log_kernel<T: LazyReal>(x: T, mu: T, sigma: T) -> T {
    let z = (x - mu) / sigma;
    -(T::half() * z.clone() * z)
}

/// Gaussian log-density `ln N(x | µ, σ) = -½·((x-µ)/σ)² − ln σ − ½·ln(2π)`,
/// written over the lazy real analysis surface with a supplied normalizer.
///
/// `half_log_2pi` is `0.5 * ln(2π)`.
pub fn normal_log_prob_with_normalizer<T: LazyReal>(x: T, mu: T, sigma: T, half_log_2pi: T) -> T {
    let log_sigma = sigma.ln();
    normal_log_kernel(x, mu, sigma) - log_sigma - half_log_2pi
}

/// Gaussian log-density `ln N(x | µ, σ) = -½·((x-µ)/σ)² − ln σ − ½·ln(2π)`,
/// with constants supplied from the lazy real tier.
pub fn normal_log_prob<T: LazyReal>(x: T, mu: T, sigma: T) -> T {
    normal_log_prob_with_normalizer(x, mu, sigma, T::half_ln_tau())
}

/// Backwards-compatible spelling for callers that already precompute both
/// constants.
pub fn normal_log_prob_with_constants<T: LazyReal>(
    x: T,
    mu: T,
    sigma: T,
    half: T,
    half_log_2pi: T,
) -> T {
    let log_sigma = sigma.clone().ln();
    let z = (x - mu) / sigma;
    -(half * z.clone() * z) - log_sigma - half_log_2pi
}

/// Squared complex residual `|z - µ|²`, written over the lazy complex analysis
/// surface.
pub fn complex_residual_norm_sqr<T: LazyComplex>(z: T, mu: T) -> T::RealPart {
    (z - mu).norm_sqr()
}

/// Circular complex-normal log-density kernel
/// `-|z-µ|² / variance - log(π·variance)`, written over [`LazyComplex`].
///
/// `log_pi_variance` is supplied by the caller because it is usually cached with
/// the complex variance parameter.
pub fn complex_normal_log_prob_with_normalizer<T: LazyComplex>(
    z: T,
    mu: T,
    variance: T::RealPart,
    log_pi_variance: T::RealPart,
) -> T::RealPart {
    -(complex_residual_norm_sqr(z, mu) / variance) - log_pi_variance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Continuous, Normal};

    #[test]
    fn generic_normal_log_prob_matches_scalar_distribution() {
        // The generic primitive equals the owning distribution's closed-form
        // ln_pdf to machine precision — one formula, one meaning.
        for (x, mu, sigma) in [(-2.0, 0.0, 1.0), (0.5, 1.0, 2.0), (3.0, -1.0, 0.7)] {
            let reference = Normal::new(mu, sigma)
                .expect("valid normal parameters")
                .ln_pdf(x);
            let generic = normal_log_prob::<f64>(x, mu, sigma);
            assert!(
                (reference - generic).abs() <= 1e-12 + 1e-12 * reference.abs(),
                "reference {reference} vs generic {generic} for ({x}, {mu}, {sigma})"
            );
        }
    }
}
