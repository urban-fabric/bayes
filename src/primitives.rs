//! Field-generic probability primitives (plans/sci-comp/supercomputer.md §3).
//!
//! The closed forms are written **once**, generic over [`euclid::field`], so the
//! *same* formula is the scalar reference here in `bayes` and — when instantiated
//! at `lebesgue::Tensor` — a differentiable tensor op, with no per-backend copy.
//! `bayes` stays tensor-free: it only names the trait.
//!
//! These mirror selected hot closed forms; the owning distribution types (e.g.
//! [`Normal`](crate::Normal)) remain the semantic authority, and their scalar
//! methods are parity-tested against these generic primitives.

use euclid::field::RealField;

/// Gaussian log-density `ln N(x | µ, σ) = -½·((x-µ)/σ)² − ln σ − ½·ln(2π)`,
/// written once over the real field.
pub fn normal_log_prob<T: RealField>(x: T, mu: T, sigma: T) -> T {
    let z = (x - mu) / sigma.clone();
    let half = T::from_f64(0.5);
    let log_2pi = T::from_f64((2.0 * core::f64::consts::PI).ln());
    -(half.clone() * z.clone() * z) - sigma.ln() - half * log_2pi
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
