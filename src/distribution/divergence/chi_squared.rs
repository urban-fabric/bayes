//! **χ² (Pearson) divergence** `χ²(p ‖ q) = ∫(p − q)²/q`, the f-divergence for
//! `f(t) = (t − 1)²` and an upper bound on KL.

use crate::distribution::{Bernoulli, Categorical, Exponential, Poisson};

/// The **χ² (Pearson) divergence** `χ²(p ‖ q) = ∫ (p − q)²/q = ∫ p²/q − 1`, in
/// `[0, ∞]`. The `f`-divergence for `f(t) = (t − 1)²`, and an upper bound on KL:
/// `D_KL(p ‖ q) ≤ χ²(p ‖ q)`. Implemented where it is a closed form (and, for
/// continuous families, where the integral converges — `χ²` can be `+∞` when the
/// second distribution has the lighter tail).
///
/// Proved in `riemann` (`RiemannLean/Bayes/Divergence.lean`): `chiSqDiv_nonneg`,
/// `klDiv_le_chiSqDiv`, and the umbrella `fDiv_nonneg` (of which both KL and χ²
/// are instances).
pub trait ChiSquaredDivergence {
    /// `χ²(self ‖ other) ∈ [0, ∞]` (`f64::INFINITY` when the integral diverges).
    fn chi_squared_divergence(&self, other: &Self) -> f64;
}

impl ChiSquaredDivergence for Bernoulli {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (p, q) = (self.p(), other.p());
        // (p−q)²/q + (p−q)²/(1−q) = (p−q)² / (q(1−q)).
        (p - q).powi(2) / (q * (1.0 - q))
    }
}

impl ChiSquaredDivergence for Poisson {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.lambda(), other.lambda());
        ((l1 - l2).powi(2) / l2).exp() - 1.0
    }
}

impl ChiSquaredDivergence for Exponential {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.rate(), other.rate());
        // ∫ λ1²e^{−2λ1x}/(λ2 e^{−λ2x}) − 1 converges only when 2λ1 > λ2.
        if 2.0 * l1 > l2 {
            l1 * l1 / (l2 * (2.0 * l1 - l2)) - 1.0
        } else {
            f64::INFINITY
        }
    }
}

impl ChiSquaredDivergence for Categorical {
    /// Categories present in only one distribution are treated as `0` in the other;
    /// a category with `q = 0` but `p > 0` makes the divergence `+∞`.
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (pa, pb) = (self.probabilities(), other.probabilities());
        let n = pa.len().max(pb.len());
        (0..n)
            .map(|i| {
                let a = pa.get(i).copied().unwrap_or(0.0);
                let b = pb.get(i).copied().unwrap_or(0.0);
                if b > 0.0 {
                    (a - b).powi(2) / b
                } else if a > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                }
            })
            .sum()
    }
}
