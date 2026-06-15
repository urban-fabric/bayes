//! **Total-variation distance** `TV(p, q) = ½ Σ|pᵢ − qᵢ|`, a bounded metric,
//! closed-form on the finite-support discretes.

use crate::distribution::{Bernoulli, Categorical};

/// The **total-variation distance** `TV(p, q) = ½ Σ |pᵢ − qᵢ|` (`= sup_A |P(A) − Q(A)|`),
/// in `[0, 1]`. Symmetric and a metric. Implemented only where it is a closed-form
/// finite sum — the finite-support discretes ([`Bernoulli`], [`Categorical`]).
///
/// Proved in `riemann` (`RiemannLean/Bayes/Divergence.lean`): `tvDist_nonneg`,
/// `tvDist_symm`, `tvDist_le_one`, and **Pinsker's inequality** `klDiv_pinsker`
/// (`TV² ≤ KL`), which controls TV by the KL divergence.
pub trait TotalVariation {
    /// The total-variation distance `TV ∈ [0, 1]`.
    fn total_variation(&self, other: &Self) -> f64;
}

impl TotalVariation for Bernoulli {
    fn total_variation(&self, other: &Self) -> f64 {
        // ½(|p−q| + |(1−p)−(1−q)|) = |p − q|.
        (self.p() - other.p()).abs()
    }
}

impl TotalVariation for Categorical {
    /// Categories present in only one distribution are treated as `0` in the other.
    fn total_variation(&self, other: &Self) -> f64 {
        let (pa, pb) = (self.probabilities(), other.probabilities());
        let n = pa.len().max(pb.len());
        0.5 * (0..n)
            .map(|i| (pa.get(i).copied().unwrap_or(0.0) - pb.get(i).copied().unwrap_or(0.0)).abs())
            .sum::<f64>()
    }
}
