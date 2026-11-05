//! **Chi-squared (Pearson) divergence** `chi2(p || q) = integral (p - q)^2/q`,
//! the f-divergence for `f(t) = (t - 1)^2` and an upper bound on KL.

use crate::{
    Bernoulli, Binomial, Categorical, ChiSquared, DiscreteUniform, Exponential, Gamma, HalfNormal,
    InverseGamma, LogNormal, NegativeBinomial, Normal, Poisson, Uniform, VonMises, Wald,
};
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Integer, Natural, Real};

/// The **chi-squared (Pearson) divergence** `chi2(p || q) = integral (p - q)^2/q`,
/// in `[0, infinity]`. Implemented where it is a closed form.
pub trait ChiSquaredDivergence {
    type Output: Real;

    /// `chi2(self || other)`; `+infinity` when the integral diverges.
    fn chi_squared_divergence(&self, other: &Self) -> Self::Output;
}

impl<R: Real> ChiSquaredDivergence for Bernoulli<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        kernels::chi_squared_bernoulli(self.p(), other.p())
    }
}

impl<R: Real> ChiSquaredDivergence for Poisson<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        kernels::chi_squared_poisson(self.lambda(), other.lambda())
    }
}

impl<R: Real> ChiSquaredDivergence for Exponential<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        let (l1, l2) = (self.rate(), other.rate());
        let witness = l1.clone() + l2.clone();
        let finite = (R::two() * l1.clone()).gt(&l2);
        let value = kernels::chi_squared_exponential_on_support(l1, l2);
        crate::guarded(&finite, value, witness)
    }
}

impl<R> ChiSquaredDivergence for Categorical<R>
where
    R: Real + Compare,
{
    type Output = R;

    /// Categories present in only one distribution are treated as `0` in the other;
    /// a category with `q = 0` but `p > 0` makes the divergence `+infinity`.
    fn chi_squared_divergence(&self, other: &Self) -> R {
        let (pa, pb) = (self.probabilities(), other.probabilities());
        let n = pa.len().max(pb.len());
        (0..n)
            .map(|i| {
                let a = pa.get(i).cloned().unwrap_or_else(R::zero);
                let b = pb.get(i).cloned().unwrap_or_else(R::zero);
                let b_positive = b.gt(&R::zero());
                let a_positive = a.gt(&R::zero());
                let denominator = R::select(&b_positive, b.clone(), R::one());
                let diff = a - b;
                let finite = diff.clone() * diff / denominator;
                R::select(
                    &b_positive,
                    finite,
                    R::select(&a_positive, R::infinity(), R::zero()),
                )
            })
            .fold(R::zero(), |acc, x| acc + x)
    }
}

impl<R: Real> ChiSquaredDivergence for Normal<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        let (m1, s1) = self.params();
        let (m2, s2) = other.params();
        let denom = R::two() * s2.clone() * s2.clone() - s1.clone() * s1.clone();
        let witness = denom.clone();
        let finite = denom.gt(&R::zero());
        let value = kernels::chi_squared_normal_on_support(m1, s1, m2, s2);
        crate::guarded(&finite, value, witness)
    }
}

impl<R: Real> ChiSquaredDivergence for LogNormal<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        let (s1, s2) = (self.sigma(), other.sigma());
        let denom = R::two() * s2.clone() * s2.clone() - s1.clone() * s1.clone();
        let witness = denom.clone();
        let finite = denom.gt(&R::zero());
        let value = kernels::chi_squared_normal_on_support(self.mu(), s1, other.mu(), s2);
        crate::guarded(&finite, value, witness)
    }
}

impl<R: Real> ChiSquaredDivergence for HalfNormal<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        let (s1, s2) = (self.scale(), other.scale());
        let denom = R::two() * s2.clone() * s2.clone() - s1.clone() * s1.clone();
        let witness = denom.clone();
        let finite = denom.gt(&R::zero());
        let value = kernels::chi_squared_normal_on_support(R::zero(), s1, R::zero(), s2);
        crate::guarded(&finite, value, witness)
    }
}

impl<R: Real> ChiSquaredDivergence for Gamma<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        let (a1, b1) = (self.shape(), self.rate());
        let (a2, b2) = (other.shape(), other.rate());
        let a = R::two() * a1.clone() - a2.clone();
        let b = R::two() * b1.clone() - b2.clone();
        let witness = a.clone() + b.clone();
        let finite = a.gt(&R::zero()).and(&b.gt(&R::zero()));
        let value = kernels::chi_squared_gamma_on_support(a1, b1, a2, b2);
        crate::guarded(&finite, value, witness)
    }
}

impl<R: Real> ChiSquaredDivergence for ChiSquared<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        let a1 = R::half() * self.freedom();
        let a2 = R::half() * other.freedom();
        let a = R::two() * a1.clone() - a2.clone();
        let witness = a.clone();
        let finite = a.gt(&R::zero());
        let value = kernels::chi_squared_gamma_on_support(a1, R::half(), a2, R::half());
        crate::guarded(&finite, value, witness)
    }
}

impl<R: Real> ChiSquaredDivergence for InverseGamma<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        let (a1, b1) = (self.shape(), self.scale());
        let (a2, b2) = (other.shape(), other.scale());
        let a = R::two() * a1.clone() - a2.clone();
        let b = R::two() * b1.clone() - b2.clone();
        let witness = a.clone() + b.clone();
        let finite = a.gt(&R::zero()).and(&b.gt(&R::zero()));
        let value = kernels::chi_squared_gamma_on_support(a1, b1, a2, b2);
        crate::guarded(&finite, value, witness)
    }
}

impl<R: Real> ChiSquaredDivergence for Wald<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        let (m1, l1) = (self.mu(), self.shape());
        let (m2, l2) = (other.mu(), other.shape());
        let la = R::two() * l1.clone() - l2.clone();
        let aa = R::two() * l1.clone() / (m1.clone() * m1.clone())
            - l2.clone() / (m2.clone() * m2.clone());
        let witness = la.clone() + aa.clone();
        let finite = la.gt(&R::zero()).and(&aa.gt(&R::zero()));
        let value = kernels::chi_squared_wald_on_support(m1, l1, m2, l2);
        crate::guarded(&finite, value, witness)
    }
}

impl<R: Real> ChiSquaredDivergence for VonMises<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        kernels::chi_squared_von_mises(self.mu(), self.kappa(), other.mu(), other.kappa())
    }
}

impl<R: Real> ChiSquaredDivergence for Uniform<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        let contained = a1.ge(&a2).and(&b1.le(&b2));
        let value = kernels::chi_squared_uniform_on_support(b1 - a1, b2 - a2);
        R::select(&contained, value, R::infinity())
    }
}

impl<I> ChiSquaredDivergence for DiscreteUniform<I>
where
    I: Integer + Compare,
    I::Real: Compare<Mask = <I as Compare>::Mask>,
{
    type Output = I::Real;

    fn chi_squared_divergence(&self, other: &Self) -> I::Real {
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        let contained = a1.ge(&a2).and(&b1.le(&b2));
        let n1 = (b1 - a1 + I::from_i64(1)).to_real();
        let n2 = (b2 - a2 + I::from_i64(1)).to_real();
        let value = kernels::chi_squared_discrete_uniform_on_support(n1, n2);
        <I::Real as Compare>::select(&contained, value, I::Real::infinity())
    }
}

impl<N> ChiSquaredDivergence for Binomial<N>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    type Output = N::Real;

    fn chi_squared_divergence(&self, other: &Self) -> N::Real {
        let same_trials = self.trials().eq(&other.trials());
        let value = kernels::chi_squared_binomial(self.trials().to_real(), self.p(), other.p());
        <N::Real as Compare>::select(&same_trials, value, N::Real::infinity())
    }
}

impl<R: Real> ChiSquaredDivergence for NegativeBinomial<R> {
    type Output = R;

    fn chi_squared_divergence(&self, other: &Self) -> R {
        let successes = self.successes();
        let same_successes = successes.eq(&other.successes());
        let (p1, p2) = (self.p(), other.p());
        let q1 = R::one() - p1.clone();
        let z = q1.clone() * q1 / (R::one() - p2.clone());
        let witness = z.clone();
        let finite = same_successes.and(&z.lt(&R::one()));
        let value = kernels::chi_squared_negative_binomial_on_support(successes, p1, p2);
        crate::guarded(&finite, value, witness)
    }
}
