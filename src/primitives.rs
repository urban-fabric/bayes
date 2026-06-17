//! Scalar-generic probability primitives (plans/sci-comp/supercomputer.md §3).
//!
//! The closed forms are written **once**, generic over the concrete real scalar
//! [`euclid::numeric::Real`], so the *same* formula is the `f64` reference here in
//! `bayes` and instantiates at any other concrete real (`f32`, a decimal). It
//! carries literal constants (`½`, `ln 2π`) via [`Real::from_f64`], so it lives on
//! the concrete tier rather than the lazy/deferred one.
//!
//! These mirror selected hot closed forms; the owning distribution types (e.g.
//! [`Normal`](crate::Normal)) remain the semantic authority, and their scalar
//! methods are parity-tested against these generic primitives.

use euclid::numeric::Real;

/// Gaussian log-density `ln N(x | µ, σ) = -½·((x-µ)/σ)² − ln σ − ½·ln(2π)`,
/// written once over the reals.
pub fn normal_log_prob<T: Real>(x: T, mu: T, sigma: T) -> T {
    let z = (x - mu) / sigma;
    let half = T::from_f64(0.5);
    let log_2pi = T::from_f64((2.0 * core::f64::consts::PI).ln());
    -(half * z * z) - sigma.ln() - half * log_2pi
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
