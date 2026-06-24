//! **Total-variation distance** `TV(p, q) = 1/2 sum |p_i - q_i|`, a bounded
//! metric, closed-form on the finite-support discretes.

use crate::{Bernoulli, Categorical};
use bayes_kernels as kernels;
use euclid::Real;

/// The **total-variation distance** `TV(p, q) = 1/2 sum |p_i - q_i|`,
/// in `[0, 1]`. Implemented only where it is a closed-form finite sum.
pub trait TotalVariation {
    type Output: Real;

    /// The total-variation distance `TV in [0, 1]`.
    fn total_variation(&self, other: &Self) -> Self::Output;
}

impl<R: Real> TotalVariation for Bernoulli<R> {
    type Output = R;

    fn total_variation(&self, other: &Self) -> R {
        kernels::total_variation_bernoulli(self.p(), other.p())
    }
}

impl<R: Real> TotalVariation for Categorical<R> {
    type Output = R;

    /// Categories present in only one distribution are treated as `0` in the other.
    fn total_variation(&self, other: &Self) -> R {
        let (pa, pb) = (self.probabilities(), other.probabilities());
        let n = pa.len().max(pb.len());
        R::half()
            * (0..n)
                .map(|i| {
                    (pa.get(i).cloned().unwrap_or_else(R::zero)
                        - pb.get(i).cloned().unwrap_or_else(R::zero))
                    .abs()
                })
                .fold(R::zero(), |acc, x| acc + x)
    }
}
