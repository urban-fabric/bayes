//! **Renyi divergence** of order `alpha` - the one-parameter family interpolating
//! KL (`alpha -> 1`), Hellinger (`alpha = 1/2`), and chi-squared (`alpha = 2`).

use super::KlDivergence;
use crate::{
    Bernoulli, Binomial, Categorical, ChiSquared, DiscreteUniform, Gamma, HalfNormal, InverseGamma,
    LogNormal, NegativeBinomial, Normal, Uniform, VonMises, Wald,
};
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Element, Integer, Natural, Real};

/// Whether `alpha` is close enough to 1 that the KL limit is the better answer.
///
/// The threshold is a **crossover**, not a nominal "equals one" test. Two
/// errors compete as `alpha → 1`:
///
/// - Returning `D_KL` instead of `D_alpha` is wrong by `O(|alpha − 1|)`, since
///   KL is the limit.
/// - Evaluating `ln(...)/(alpha − 1)` is wrong by `O(eps/|alpha − 1|)`, because
///   the logarithm of a quantity near 1 carries an absolute error of about
///   `eps` and the division amplifies it.
///
/// They cross at `|alpha − 1| ≈ √eps ≈ 1.5e-8`. The threshold was `1e-12`,
/// four orders of magnitude inside the region where the formula is the worse
/// of the two: `renyi_divergence(p, q, 1 − 1e-9)` came back with a relative
/// error of `2.6e-7` against a KL limit that was accurate to `1e-9`.
/// A Renyi divergence is nonnegative analytically, so clamp the rounding noise
/// the `1/(alpha − 1)` factor introduces.
///
/// At `|alpha − 1| = 1e-7` that factor amplifies the logarithm's `eps`-level
/// error to about `2e-9`, which for two *identical* distributions comes out
/// **negative** — and `D_f ≥ 0` is one of the guarantees this crate documents
/// as machine-checked. A caller taking `sqrt` of it would get a `NaN`.
///
/// `clamp_negative_to_zero` rather than `max(0)`, so a genuine `NaN` is not
/// turned into a confident zero.
fn clamp<R: Real>(value: R) -> R {
    bayes_kernels::clamp_negative_to_zero(value)
}

fn alpha_is_kl<R: Real>(alpha: &R) -> R::Mask {
    (alpha.clone() - R::one()).abs().lt(&R::scalar(1.5e-8))
}

/// The **Renyi divergence** of order `alpha`,
/// `D_alpha(p || q) = ln integral p^alpha q^(1-alpha) / (alpha - 1)`.
pub trait RenyiDivergence {
    type Output: Real;

    /// `D_alpha(self || other)`, in nats. `alpha > 0`; `alpha = 1` returns KL.
    fn renyi_divergence(&self, other: &Self, alpha: Self::Output) -> Self::Output;
}

impl<R: Real> RenyiDivergence for Bernoulli<R> {
    type Output = R;

    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let is_kl = alpha_is_kl(&alpha);
        let value = kernels::renyi_bernoulli(self.p(), other.p(), alpha);
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}

impl<R> RenyiDivergence for Categorical<R>
where
    R: Real + Compare,
{
    type Output = R;

    /// Categories present in only one distribution are treated as `0` in the other.
    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let (pa, pb) = (self.probabilities(), other.probabilities());
        let n = pa.len().max(pb.len());
        let is_kl = alpha_is_kl(&alpha);
        let s = (0..n)
            .map(|i| {
                let a = pa.get(i).cloned().unwrap_or_else(R::zero);
                let b = pb.get(i).cloned().unwrap_or_else(R::zero);
                let a_positive = a.gt(&R::zero());
                let base_a = R::select(&a_positive, a, R::one());
                R::select(
                    &a_positive,
                    base_a.powf(alpha.clone()) * b.powf(R::one() - alpha.clone()),
                    R::zero(),
                )
            })
            .fold(R::zero(), |acc, x| acc + x);
        let value = s.ln() / (alpha - R::one());
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}

impl<R: Real> RenyiDivergence for Normal<R> {
    type Output = R;

    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let is_kl = alpha_is_kl(&alpha);
        let (m1, s1) = self.params();
        let (m2, s2) = other.params();
        let one_minus_alpha = R::one() - alpha.clone();
        let var_alpha =
            alpha.clone() * s2.clone() * s2.clone() + one_minus_alpha * s1.clone() * s1.clone();
        let witness = var_alpha.clone();
        let finite = var_alpha.gt(&R::zero());
        let value = kernels::renyi_normal_on_support(m1, s1, m2, s2, alpha);
        let value = crate::guarded(&finite, value, witness);
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}

impl<R: Real> RenyiDivergence for LogNormal<R> {
    type Output = R;

    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let is_kl = alpha_is_kl(&alpha);
        let (s1, s2) = (self.sigma(), other.sigma());
        let one_minus_alpha = R::one() - alpha.clone();
        let var_alpha =
            alpha.clone() * s2.clone() * s2.clone() + one_minus_alpha * s1.clone() * s1.clone();
        let witness = var_alpha.clone();
        let finite = var_alpha.gt(&R::zero());
        let value = kernels::renyi_normal_on_support(self.mu(), s1, other.mu(), s2, alpha);
        let value = crate::guarded(&finite, value, witness);
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}

impl<R: Real> RenyiDivergence for HalfNormal<R> {
    type Output = R;

    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let is_kl = alpha_is_kl(&alpha);
        let (s1, s2) = (self.scale(), other.scale());
        let one_minus_alpha = R::one() - alpha.clone();
        let var_alpha =
            alpha.clone() * s2.clone() * s2.clone() + one_minus_alpha * s1.clone() * s1.clone();
        let witness = var_alpha.clone();
        let finite = var_alpha.gt(&R::zero());
        let value = kernels::renyi_normal_on_support(R::zero(), s1, R::zero(), s2, alpha);
        let value = crate::guarded(&finite, value, witness);
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}

impl<R: Real> RenyiDivergence for Gamma<R> {
    type Output = R;

    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let is_kl = alpha_is_kl(&alpha);
        let (a1, b1) = (self.shape(), self.rate());
        let (a2, b2) = (other.shape(), other.rate());
        let one_minus_alpha = R::one() - alpha.clone();
        let a_alpha = alpha.clone() * a1.clone() + one_minus_alpha.clone() * a2.clone();
        let b_alpha = alpha.clone() * b1.clone() + one_minus_alpha * b2.clone();
        let witness = a_alpha.clone() + b_alpha.clone();
        let finite = a_alpha.gt(&R::zero()).and(&b_alpha.gt(&R::zero()));
        let value = kernels::renyi_gamma_on_support(a1, b1, a2, b2, alpha);
        let value = crate::guarded(&finite, value, witness);
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}

impl<R: Real> RenyiDivergence for ChiSquared<R> {
    type Output = R;

    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let is_kl = alpha_is_kl(&alpha);
        let a1 = R::half() * self.freedom();
        let a2 = R::half() * other.freedom();
        let one_minus_alpha = R::one() - alpha.clone();
        let a_alpha = alpha.clone() * a1.clone() + one_minus_alpha * a2.clone();
        let witness = a_alpha.clone();
        let finite = a_alpha.gt(&R::zero());
        let value = kernels::renyi_gamma_on_support(a1, R::half(), a2, R::half(), alpha);
        let value = crate::guarded(&finite, value, witness);
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}

impl<R: Real> RenyiDivergence for InverseGamma<R> {
    type Output = R;

    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let is_kl = alpha_is_kl(&alpha);
        let (a1, b1) = (self.shape(), self.scale());
        let (a2, b2) = (other.shape(), other.scale());
        let one_minus_alpha = R::one() - alpha.clone();
        let a_alpha = alpha.clone() * a1.clone() + one_minus_alpha.clone() * a2.clone();
        let b_alpha = alpha.clone() * b1.clone() + one_minus_alpha * b2.clone();
        let witness = a_alpha.clone() + b_alpha.clone();
        let finite = a_alpha.gt(&R::zero()).and(&b_alpha.gt(&R::zero()));
        let value = kernels::renyi_gamma_on_support(a1, b1, a2, b2, alpha);
        let value = crate::guarded(&finite, value, witness);
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}

impl<R: Real> RenyiDivergence for VonMises<R> {
    type Output = R;

    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let is_kl = alpha_is_kl(&alpha);
        let value =
            kernels::renyi_von_mises(self.mu(), self.kappa(), other.mu(), other.kappa(), alpha);
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}

impl<R: Real> RenyiDivergence for Wald<R> {
    type Output = R;

    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let is_kl = alpha_is_kl(&alpha);
        let (m1, l1) = (self.mu(), self.shape());
        let (m2, l2) = (other.mu(), other.shape());
        let one_minus_alpha = R::one() - alpha.clone();
        let la = alpha.clone() * l1.clone() + one_minus_alpha.clone() * l2.clone();
        let aa = alpha.clone() * l1.clone() / (m1.clone() * m1.clone())
            + one_minus_alpha * l2.clone() / (m2.clone() * m2.clone());
        let witness = la.clone() + aa.clone();
        let finite = la.gt(&R::zero()).and(&aa.gt(&R::zero()));
        let value = kernels::renyi_wald_on_support(m1, l1, m2, l2, alpha);
        let value = crate::guarded(&finite, value, witness);
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}

impl<R: Real> RenyiDivergence for Uniform<R> {
    type Output = R;

    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let is_kl = alpha_is_kl(&alpha);
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        let contained = a1.ge(&a2).and(&b1.le(&b2));
        let support_ok = alpha.le(&R::one()).or(&contained);
        let overlap = (b1.clone().min(b2.clone()) - a1.clone().max(a2.clone())).max(R::zero());
        let witness = overlap.clone() + alpha.clone();
        let finite = support_ok.and(&overlap.gt(&R::zero()));
        let value = kernels::renyi_uniform_on_support(a1, b1, a2, b2, alpha);
        let value = crate::guarded(&finite, value, witness);
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}

impl<I> RenyiDivergence for DiscreteUniform<I>
where
    I: Integer + Compare,
    I::Real: Compare<Mask = <I as Compare>::Mask>,
{
    type Output = I::Real;

    fn renyi_divergence(&self, other: &Self, alpha: I::Real) -> I::Real {
        let is_kl = alpha_is_kl(&alpha);
        let (a1, b1) = (self.lo().to_real(), self.hi().to_real());
        let (a2, b2) = (other.lo().to_real(), other.hi().to_real());
        let contained = a1.ge(&a2).and(&b1.le(&b2));
        let support_ok = alpha.le(&I::Real::one()).or(&contained);
        let overlap = (b1.clone().min(b2.clone()) - a1.clone().max(a2.clone()) + I::Real::one())
            .max(I::Real::zero());
        let finite = support_ok.and(&overlap.gt(&I::Real::zero()));
        let value = kernels::renyi_discrete_uniform_on_support(a1, b1, a2, b2, alpha);
        let value = <I::Real as Compare>::select(&finite, value, I::Real::infinity());
        <I::Real as Compare>::select(&is_kl, self.kl_divergence(other), value)
    }
}

impl<N> RenyiDivergence for Binomial<N>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    type Output = N::Real;

    fn renyi_divergence(&self, other: &Self, alpha: N::Real) -> N::Real {
        let is_kl = alpha_is_kl(&alpha);
        let same_trials = self.trials().eq(&other.trials());
        let value = kernels::renyi_binomial(self.trials().to_real(), self.p(), other.p(), alpha);
        let value = <N::Real as Compare>::select(&same_trials, value, N::Real::infinity());
        <N::Real as Compare>::select(&is_kl, self.kl_divergence(other), value)
    }
}

impl<R: Real> RenyiDivergence for NegativeBinomial<R> {
    type Output = R;

    fn renyi_divergence(&self, other: &Self, alpha: R) -> R {
        let is_kl = alpha_is_kl(&alpha);
        let successes = self.successes();
        let same_successes = successes.eq(&other.successes());
        let (p1, p2) = (self.p(), other.p());
        let one_minus_alpha = R::one() - alpha.clone();
        let w = (alpha.clone() * (R::one() - p1.clone()).ln()).exp()
            * (one_minus_alpha * (R::one() - p2.clone()).ln()).exp();
        let witness = w.clone();
        let finite = same_successes.and(&w.lt(&R::one()));
        let value = kernels::renyi_negative_binomial_on_support(successes, p1, p2, alpha);
        let value = crate::guarded(&finite, value, witness);
        clamp(R::select(&is_kl, self.kl_divergence(other), value))
    }
}
