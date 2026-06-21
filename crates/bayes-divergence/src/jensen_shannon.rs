//! **Jensen–Shannon divergence** — the symmetric, bounded (`0 ≤ JS ≤ ln 2`)
//! symmetrization of KL, closed-form on the finite-support discretes where the
//! mixture `m = (p+q)/2` stays in family.

use super::kl_term;
use crate::{Bernoulli, Categorical};

/// The **Jensen–Shannon divergence** `JS(p, q) = ½ D_KL(p ‖ m) + ½ D_KL(q ‖ m)`,
/// `m = (p+q)/2`, in nats. Unlike KL it is **symmetric** and **bounded**:
/// `0 ≤ JS ≤ ln 2` (its square root is a metric).
///
/// It is implemented only where the mixture `m` stays in the same family and the
/// sum is therefore a closed form — the finite-support discretes. The bounds are
/// proved in `riemann` (`RiemannLean/Bayes/Divergence.lean`: `jsDiv_nonneg`,
/// `jsDiv_le_log2`).
pub trait JensenShannon {
    /// `JS(self ‖ other)`, in nats, in `[0, ln 2]`.
    fn js_divergence(&self, other: &Self) -> f64;
}

impl JensenShannon for Bernoulli {
    fn js_divergence(&self, other: &Self) -> f64 {
        let (p, q) = (self.p(), other.p());
        let m1 = 0.5 * (p + q); // mixture P(success)
        let m0 = 0.5 * ((1.0 - p) + (1.0 - q)); // mixture P(failure)
        0.5 * (kl_term(p, m1) + kl_term(1.0 - p, m0))
            + 0.5 * (kl_term(q, m1) + kl_term(1.0 - q, m0))
    }
}

impl JensenShannon for Categorical {
    /// Categories present in only one distribution (unequal lengths) are treated
    /// as having probability `0` in the other.
    fn js_divergence(&self, other: &Self) -> f64 {
        let (pa, pb) = (self.probabilities(), other.probabilities());
        let n = pa.len().max(pb.len());
        (0..n)
            .map(|i| {
                let a = pa.get(i).copied().unwrap_or(0.0);
                let b = pb.get(i).copied().unwrap_or(0.0);
                let m = 0.5 * (a + b);
                0.5 * (kl_term(a, m) + kl_term(b, m))
            })
            .sum()
    }
}
