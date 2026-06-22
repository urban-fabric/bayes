//! **Jensen-Shannon divergence** - the symmetric, bounded (`0 <= JS <= ln 2`)
//! symmetrization of KL, closed-form on the finite-support discretes where the
//! mixture `m = (p + q)/2` stays in family.

use super::kl_term;
use crate::{Bernoulli, Categorical};
use euclid::Real;

/// The **Jensen-Shannon divergence** `JS(p, q) = 1/2 D_KL(p || m) + 1/2 D_KL(q || m)`,
/// `m = (p + q)/2`, in nats.
pub trait JensenShannon {
    type Output: Real;

    /// `JS(self || other)`, in nats, in `[0, ln 2]`.
    fn js_divergence(&self, other: &Self) -> Self::Output;
}

impl<R: Real> JensenShannon for Bernoulli<R> {
    type Output = R;

    fn js_divergence(&self, other: &Self) -> R {
        let p = self.p();
        let q = other.p();
        let one = R::one();
        let p0 = one.clone() - p.clone();
        let q0 = one - q.clone();
        let m1 = R::half() * (p.clone() + q.clone());
        let m0 = R::half() * (p0.clone() + q0.clone());
        R::half() * (kl_term(p, m1.clone()) + kl_term(p0, m0.clone()))
            + R::half() * (kl_term(q, m1) + kl_term(q0, m0))
    }
}

impl<R: Real> JensenShannon for Categorical<R> {
    type Output = R;

    /// Categories present in only one distribution (unequal lengths) are treated
    /// as having probability `0` in the other.
    fn js_divergence(&self, other: &Self) -> R {
        let (pa, pb) = (self.probabilities(), other.probabilities());
        let n = pa.len().max(pb.len());
        (0..n)
            .map(|i| {
                let a = pa.get(i).cloned().unwrap_or_else(R::zero);
                let b = pb.get(i).cloned().unwrap_or_else(R::zero);
                let m = R::half() * (a.clone() + b.clone());
                R::half() * (kl_term(a, m.clone()) + kl_term(b, m))
            })
            .fold(R::zero(), |acc, x| acc + x)
    }
}
