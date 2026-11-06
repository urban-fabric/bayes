//! **Kullback-Leibler divergence** and its free symmetrizations.
//!
//! [`KlDivergence`] carries the per-family closed-form `D_KL(p || q)`; [`Divergence`]
//! is a blanket trait giving every KL family the reverse direction, the Jeffreys
//! divergence, and cross-entropy for free.

use super::{discrete_kl, kl_term};
use crate::{
    Bernoulli, Beta, Binomial, Categorical, Cauchy, ChiSquared, DiscreteUniform, Exponential,
    Frechet, Gamma, Geometric, Gumbel, HalfNormal, InverseGamma, Laplace, LogNormal, Moments,
    NegativeBinomial, Normal, Pareto, Poisson, Rayleigh, Uniform, VonMises, Wald, Weibull,
};
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Integer, Natural, Real};

/// Closed-form Kullback-Leibler divergence `D_KL(self || other)`, in nats.
///
/// `D_KL >= 0` for any same-family pair (**Gibbs' inequality**), with equality iff
/// the distributions are identical - proved in `riemann`
/// (`RiemannLean/Bayes/Divergence.lean`: `klDiv_nonneg`, `klDiv_self`).
pub trait KlDivergence<Rhs = Self> {
    type Output: Real;

    fn kl_divergence(&self, other: &Rhs) -> Self::Output;
}

impl<R: Real> KlDivergence for Normal<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        let (m1, s1) = self.params();
        let (m2, s2) = other.params();
        kernels::kl_normal(m1, s1, m2, s2)
    }
}

impl<R: Real> KlDivergence for Exponential<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_exponential(self.rate(), other.rate())
    }
}

impl<R: Real> KlDivergence for Gamma<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_gamma(self.shape(), self.rate(), other.shape(), other.rate())
    }
}

impl<R: Real> KlDivergence for Beta<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_beta(self.alpha(), self.beta(), other.alpha(), other.beta())
    }
}

impl<R: Real> KlDivergence for Bernoulli<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        let (p, q) = (self.p(), other.p());
        kl_term(p.clone(), q.clone()) + kl_term(R::one() - p, R::one() - q)
    }
}

impl<R: Real> KlDivergence for Poisson<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_poisson(self.lambda(), other.lambda())
    }
}

impl<R: Real> KlDivergence for LogNormal<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_normal(self.mu(), self.sigma(), other.mu(), other.sigma())
    }
}

impl<R: Real> KlDivergence for Laplace<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_laplace(
            self.location(),
            self.scale(),
            other.location(),
            other.scale(),
        )
    }
}

impl<R: Real> KlDivergence for Geometric<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_geometric(self.p(), other.p())
    }
}

impl<R: Real> KlDivergence for Categorical<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        discrete_kl(self.probabilities(), other.probabilities())
    }
}

impl<R: Real> KlDivergence for Cauchy<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_cauchy(
            self.location(),
            self.scale(),
            other.location(),
            other.scale(),
        )
    }
}

impl<R: Real> KlDivergence for Rayleigh<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_rayleigh(self.scale(), other.scale())
    }
}

impl<R: Real> KlDivergence for Weibull<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_weibull(self.shape(), self.scale(), other.shape(), other.scale())
    }
}

impl<R: Real> KlDivergence for Pareto<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        let (xm1, a1) = (self.scale(), self.shape());
        let (xm2, a2) = (other.scale(), other.shape());
        let finite = xm1.ge(&xm2);
        let value = kernels::kl_pareto_on_support(xm1, a1, xm2, a2);
        R::select(&finite, value, R::infinity())
    }
}

impl<R: Real> KlDivergence for ChiSquared<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_gamma(
            R::half() * self.freedom(),
            R::half(),
            R::half() * other.freedom(),
            R::half(),
        )
    }
}

impl<R: Real> KlDivergence for InverseGamma<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_gamma(self.shape(), self.scale(), other.shape(), other.scale())
    }
}

impl<R: Real> KlDivergence for HalfNormal<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_normal(R::zero(), self.scale(), R::zero(), other.scale())
    }
}

impl<R: Real> KlDivergence for Frechet<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_weibull(
            self.shape(),
            R::one() / self.scale(),
            other.shape(),
            R::one() / other.scale(),
        )
    }
}

impl<R: Real> KlDivergence for Gumbel<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_gumbel(
            self.location(),
            self.scale(),
            other.location(),
            other.scale(),
        )
    }
}

impl<R: Real> KlDivergence for Wald<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_wald(self.mu(), self.shape(), other.mu(), other.shape())
    }
}

impl<R: Real> KlDivergence for VonMises<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        kernels::kl_von_mises(self.mu(), self.kappa(), other.mu(), other.kappa())
    }
}

impl<R: Real> KlDivergence for Uniform<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        let contained = a1.ge(&a2).and(&b1.le(&b2));
        let value = kernels::kl_uniform_on_support(b1 - a1, b2 - a2);
        R::select(&contained, value, R::infinity())
    }
}

impl<N> KlDivergence for Binomial<N>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    type Output = N::Real;

    fn kl_divergence(&self, other: &Self) -> N::Real {
        let same_trials = self.trials().eq(&other.trials());
        let value = kernels::kl_binomial(self.trials().to_real(), self.p(), other.p());
        <N::Real as Compare>::select(&same_trials, value, N::Real::infinity())
    }
}

impl<R: Real> KlDivergence for NegativeBinomial<R> {
    type Output = R;

    fn kl_divergence(&self, other: &Self) -> R {
        let successes = self.successes();
        let same_successes = successes.eq(&other.successes());
        let value = kernels::kl_negative_binomial(successes, self.p(), other.p());
        R::select(&same_successes, value, R::infinity())
    }
}

impl<I> KlDivergence for DiscreteUniform<I>
where
    I: Integer + Compare,
    I::Real: Compare<Mask = <I as Compare>::Mask>,
{
    type Output = I::Real;

    fn kl_divergence(&self, other: &Self) -> I::Real {
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        let contained = a1.ge(&a2).and(&b1.le(&b2));
        let n1 = (b1 - a1 + I::from_i64(1)).to_real();
        let n2 = (b2 - a2 + I::from_i64(1)).to_real();
        let value = kernels::kl_discrete_uniform_on_support(n1, n2);
        <I::Real as Compare>::select(&contained, value, I::Real::infinity())
    }
}

/// Symmetrizations of [`KlDivergence`] available to every KL family for free: the
/// reverse direction and the **Jeffreys** divergence.
pub trait Divergence: KlDivergence + Sized {
    /// `D_KL(other || self)` - the divergence in the reverse direction.
    fn reverse_kl(&self, other: &Self) -> Self::Output {
        other.kl_divergence(self)
    }

    /// **Jeffreys divergence** `J(self, other) = D_KL(self || other) + D_KL(other || self)`.
    fn jeffreys(&self, other: &Self) -> Self::Output {
        self.kl_divergence(other) + other.kl_divergence(self)
    }

    /// Cross-entropy `H(self, other) = H(self) + D_KL(self || other)`, in nats.
    fn cross_entropy(&self, other: &Self) -> Self::Output
    where
        Self: Moments<Self::Output>,
    {
        self.entropy() + self.kl_divergence(other)
    }
}

impl<T: KlDivergence> Divergence for T {}
