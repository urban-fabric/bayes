//! **Squared Hellinger distance** `H²(p, q) = 1 − BC`, a bounded symmetric
//! divergence whose square root is a metric. Closed-form per family via the
//! Bhattacharyya coefficient `BC = ∫√(pq)`.

use crate::distribution::{Bernoulli, Beta, Exponential, Gamma, Normal, Poisson};
use crate::function::beta::ln_beta;
use crate::function::gamma::ln_gamma;

/// `H² = 1 − BC` clamped to `[0, ∞)` — the `max` only absorbs `O(ε)` rounding,
/// since `BC ≤ 1` is proved (`bhattacharyya_le_one`).
fn hellinger_sq_from_bc(bc: f64) -> f64 {
    (1.0 - bc).max(0.0)
}

/// The **squared Hellinger distance** `H²(p, q) = ½ ∫ (√p − √q)² = 1 − BC`, where
/// `BC = ∫ √(pq)` is the Bhattacharyya coefficient. A *bounded* divergence:
/// `0 ≤ H² ≤ 1`, symmetric, zero iff `p = q`, and `√H²` is a metric. Closed-form
/// per family (no Monte-Carlo).
///
/// Proved in `riemann` (`RiemannLean/Bayes/Divergence.lean`): `hellingerSq_nonneg`,
/// `hellingerSq_symm`, `hellingerSq_le_one`, and the comparisons
/// `hellingerSq_le_tvDist` (`H² ≤ TV`) and `hellingerSq_le_half_klDiv` (`H² ≤ ½ KL`).
pub trait HellingerDistance {
    /// The squared Hellinger distance `H² ∈ [0, 1]`.
    fn hellinger_sq(&self, other: &Self) -> f64;
    /// The Hellinger distance `H = √(H²) ∈ [0, 1]` — a metric.
    fn hellinger(&self, other: &Self) -> f64 {
        self.hellinger_sq(other).sqrt()
    }
}

impl HellingerDistance for Normal {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        let (m1, s1) = self.params();
        let (m2, s2) = other.params();
        let v = s1 * s1 + s2 * s2;
        let bc = (2.0 * s1 * s2 / v).sqrt() * (-0.25 * (m1 - m2) * (m1 - m2) / v).exp();
        hellinger_sq_from_bc(bc)
    }
}

impl HellingerDistance for Exponential {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.rate(), other.rate());
        hellinger_sq_from_bc(2.0 * (l1 * l2).sqrt() / (l1 + l2))
    }
}

impl HellingerDistance for Gamma {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        let (a1, b1) = (self.shape(), self.rate());
        let (a2, b2) = (other.shape(), other.rate());
        let a = 0.5 * (a1 + a2);
        let b = 0.5 * (b1 + b2);
        let ln_bc = ln_gamma(a) + 0.5 * (a1 * b1.ln() + a2 * b2.ln())
            - 0.5 * (ln_gamma(a1) + ln_gamma(a2))
            - a * b.ln();
        hellinger_sq_from_bc(ln_bc.exp())
    }
}

impl HellingerDistance for Beta {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        let (a1, b1) = (self.alpha(), self.beta());
        let (a2, b2) = (other.alpha(), other.beta());
        let ln_bc =
            ln_beta(0.5 * (a1 + a2), 0.5 * (b1 + b2)) - 0.5 * (ln_beta(a1, b1) + ln_beta(a2, b2));
        hellinger_sq_from_bc(ln_bc.exp())
    }
}

impl HellingerDistance for Poisson {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.lambda(), other.lambda());
        hellinger_sq_from_bc((-0.5 * (l1.sqrt() - l2.sqrt()).powi(2)).exp())
    }
}

impl HellingerDistance for Bernoulli {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        let (p, q) = (self.p(), other.p());
        hellinger_sq_from_bc((p * q).sqrt() + ((1.0 - p) * (1.0 - q)).sqrt())
    }
}
