//! **Squared Hellinger distance** `H^2(p, q) = 1 - BC`, a bounded symmetric
//! divergence whose square root is a metric. Closed-form per family via the
//! Bhattacharyya coefficient `BC = integral sqrt(pq)`.

use crate::{
    Bernoulli, Beta, Binomial, ChiSquared, DiscreteUniform, Exponential, Gamma, HalfNormal,
    InverseGamma, LogNormal, NegativeBinomial, Normal, Poisson, Uniform, VonMises, Wald,
};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Element, Integer, Natural, Real};

/// The **squared Hellinger distance** `H^2(p, q) = 1 - BC`, where
/// `BC = integral sqrt(pq)` is the Bhattacharyya coefficient.
pub trait HellingerDistance {
    type Output: Real;

    /// The squared Hellinger distance `H^2 in [0, 1]`.
    fn hellinger_sq(&self, other: &Self) -> Self::Output;

    /// The Hellinger distance `H = sqrt(H^2) in [0, 1]`.
    fn hellinger(&self, other: &Self) -> Self::Output {
        self.hellinger_sq(other).sqrt()
    }
}

impl<R: Real> HellingerDistance for Normal<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        let (m1, s1) = self.params();
        let (m2, s2) = other.params();
        kernels::hellinger_sq_normal(m1, s1, m2, s2)
    }
}

impl<R: Real> HellingerDistance for Exponential<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_exponential(self.rate(), other.rate())
    }
}

impl<R: Real> HellingerDistance for Gamma<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_gamma(self.shape(), self.rate(), other.shape(), other.rate())
    }
}

impl<R: Real> HellingerDistance for Beta<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_beta(self.alpha(), self.beta(), other.alpha(), other.beta())
    }
}

impl<R: Real> HellingerDistance for Poisson<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_poisson(self.lambda(), other.lambda())
    }
}

impl<R: Real> HellingerDistance for Bernoulli<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_bernoulli(self.p(), other.p())
    }
}

impl<R: Real> HellingerDistance for LogNormal<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_normal(self.mu(), self.sigma(), other.mu(), other.sigma())
    }
}

impl<R: Real> HellingerDistance for HalfNormal<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_normal(R::zero(), self.scale(), R::zero(), other.scale())
    }
}

impl<R: Real> HellingerDistance for ChiSquared<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_gamma(
            R::half() * self.freedom(),
            R::half(),
            R::half() * other.freedom(),
            R::half(),
        )
    }
}

impl<R: Real> HellingerDistance for InverseGamma<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_gamma(self.shape(), self.scale(), other.shape(), other.scale())
    }
}

impl<R: Real> HellingerDistance for Wald<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_wald(self.mu(), self.shape(), other.mu(), other.shape())
    }
}

impl<R: Real> HellingerDistance for VonMises<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_von_mises(self.mu(), self.kappa(), other.mu(), other.kappa())
    }
}

impl<R: Real> HellingerDistance for Uniform<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        kernels::hellinger_sq_uniform(self.lo(), self.hi(), other.lo(), other.hi())
    }
}

impl<I: Integer> HellingerDistance for DiscreteUniform<I> {
    type Output = I::Real;

    fn hellinger_sq(&self, other: &Self) -> I::Real {
        kernels::hellinger_sq_discrete_uniform(
            self.lo().to_real(),
            self.hi().to_real(),
            other.lo().to_real(),
            other.hi().to_real(),
        )
    }
}

impl<N> HellingerDistance for Binomial<N>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    type Output = N::Real;

    fn hellinger_sq(&self, other: &Self) -> N::Real {
        let same_trials = self.trials().eq(&other.trials());
        let value = kernels::hellinger_sq_binomial(self.trials().to_real(), self.p(), other.p());
        <N::Real as Compare>::select(&same_trials, value, N::Real::one())
    }
}

impl<R: Real> HellingerDistance for NegativeBinomial<R> {
    type Output = R;

    fn hellinger_sq(&self, other: &Self) -> R {
        let successes = self.successes();
        let same_successes = successes.eq(&other.successes());
        let value = kernels::hellinger_sq_negative_binomial(successes, self.p(), other.p());
        R::select(&same_successes, value, R::one())
    }
}
