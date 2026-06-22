//! Backend-generic support, CDF, and quantile impls for lazy distribution families.

use crate::continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel, HalfNormal,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh, StudentT, Uniform,
    VonMises, Wald, Weibull,
};
use crate::discrete::{Bernoulli, Binomial, DiscreteUniform, Geometric, NegativeBinomial, Poisson};
use crate::{ContinuousCdf, DiscreteCdf, IntegerDiscreteCdf, Support};
use bayes_kernels as kernels;
use euclid::{Boolean, Compare, Element, Integer, Natural, Real};

impl<T: Real> Support<T> for Normal<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Normal<T> {
    fn cdf(&self, x: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_cdf(x, mean, std_dev)
    }

    fn inverse_cdf(&self, p: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_quantile(p.clamp(T::zero(), T::one()), mean, std_dev)
    }

    fn ln_cdf(&self, x: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_ln_cdf(x, mean, std_dev)
    }

    fn ln_sf(&self, x: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_ln_sf(x, mean, std_dev)
    }
}

impl<T: Real> Support<T> for Cauchy<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Cauchy<T> {
    fn cdf(&self, x: T) -> T {
        kernels::cauchy_cdf(x, self.location(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::cauchy_quantile(p.clamp(T::zero(), T::one()), self.location(), self.scale())
    }
}

impl<T: Real> Support<T> for Laplace<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Laplace<T> {
    fn cdf(&self, x: T) -> T {
        kernels::laplace_cdf(x, self.location(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::laplace_quantile(p.clamp(T::zero(), T::one()), self.location(), self.scale())
    }
}

impl<T: Real> Support<T> for Logistic<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Logistic<T> {
    fn cdf(&self, x: T) -> T {
        kernels::logistic_cdf(x, self.location(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::logistic_quantile(p.clamp(T::zero(), T::one()), self.location(), self.scale())
    }

    fn ln_cdf(&self, x: T) -> T {
        kernels::logistic_ln_cdf(x, self.location(), self.scale())
    }

    fn ln_sf(&self, x: T) -> T {
        kernels::logistic_ln_sf(x, self.location(), self.scale())
    }
}

impl<T: Real> Support<T> for Gumbel<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Gumbel<T> {
    fn cdf(&self, x: T) -> T {
        kernels::gumbel_cdf(x, self.location(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::gumbel_quantile(p.clamp(T::zero(), T::one()), self.location(), self.scale())
    }

    fn ln_cdf(&self, x: T) -> T {
        kernels::gumbel_ln_cdf(x, self.location(), self.scale())
    }
}

impl<T: Real> Support<T> for StudentT<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for StudentT<T> {
    fn cdf(&self, x: T) -> T {
        kernels::students_t_cdf(x, self.location(), self.scale(), self.freedom())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::neg_infinity(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }
}

impl<T: Real> Support<T> for Uniform<T> {
    fn min(&self) -> T {
        self.lo()
    }

    fn max(&self) -> T {
        self.hi()
    }
}

impl<T: Real> ContinuousCdf<T> for Uniform<T> {
    fn cdf(&self, x: T) -> T {
        kernels::uniform_cdf(x, self.lo(), self.hi() - self.lo())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::uniform_quantile(
            p.clamp(T::zero(), T::one()),
            self.lo(),
            self.hi() - self.lo(),
        )
    }
}

impl<T: Real> Support<T> for Beta<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::one()
    }
}

impl<T: Real> ContinuousCdf<T> for Beta<T> {
    fn cdf(&self, x: T) -> T {
        kernels::beta_cdf_on_unit(x.clamp(T::zero(), T::one()), self.alpha(), self.beta())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::one(),
            p.clamp(T::zero(), T::one()),
        )
    }
}

impl<T: Real> Support<T> for Exponential<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Exponential<T> {
    fn cdf(&self, x: T) -> T {
        kernels::exponential_cdf(x, self.rate())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::exponential_quantile(p.clamp(T::zero(), T::one()), self.rate())
    }

    fn ln_cdf(&self, x: T) -> T {
        kernels::exponential_ln_cdf(x, self.rate())
    }

    fn ln_sf(&self, x: T) -> T {
        kernels::exponential_ln_sf(x, self.rate())
    }
}

impl<T: Real> Support<T> for Gamma<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Gamma<T> {
    fn cdf(&self, x: T) -> T {
        kernels::gamma_cdf(x, self.shape(), self.rate())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }
}

impl<T: Real> Support<T> for ChiSquared<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for ChiSquared<T> {
    fn cdf(&self, x: T) -> T {
        kernels::chi_squared_cdf(x, self.freedom() * T::half())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }
}

impl<T: Real> Support<T> for FisherSnedecor<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for FisherSnedecor<T> {
    fn cdf(&self, x: T) -> T {
        kernels::fisher_snedecor_cdf(x, self.d1(), self.d2())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }
}

impl<T: Real> Support<T> for InverseGamma<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for InverseGamma<T> {
    fn cdf(&self, x: T) -> T {
        kernels::inverse_gamma_cdf(x, self.shape(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }
}

impl<T: Real> Support<T> for Frechet<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Frechet<T> {
    fn cdf(&self, x: T) -> T {
        kernels::frechet_cdf(x, self.scale(), self.shape())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::frechet_quantile(p.clamp(T::zero(), T::one()), self.scale(), self.shape())
    }

    fn ln_cdf(&self, x: T) -> T {
        kernels::frechet_ln_cdf(x, self.scale(), self.shape())
    }
}

impl<T: Real> Support<T> for HalfNormal<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for HalfNormal<T> {
    fn cdf(&self, x: T) -> T {
        kernels::half_normal_cdf(x, self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::half_normal_quantile(p.clamp(T::zero(), T::one()), self.scale())
    }
}

impl<T: Real> Support<T> for LogNormal<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for LogNormal<T> {
    fn cdf(&self, x: T) -> T {
        kernels::lognormal_cdf(x, self.mu(), self.sigma())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::lognormal_quantile(p.clamp(T::zero(), T::one()), self.mu(), self.sigma())
    }
}

impl<T: Real> Support<T> for Lomax<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Lomax<T> {
    fn cdf(&self, x: T) -> T {
        kernels::lomax_cdf(x, self.shape(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::lomax_quantile(p.clamp(T::zero(), T::one()), self.scale(), self.shape())
    }

    fn ln_sf(&self, x: T) -> T {
        kernels::lomax_ln_sf(x, self.shape(), self.scale())
    }
}

impl<T: Real> Support<T> for Pareto<T> {
    fn min(&self) -> T {
        self.scale()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Pareto<T> {
    fn cdf(&self, x: T) -> T {
        kernels::pareto_cdf(x, self.scale(), self.shape())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::pareto_quantile(p.clamp(T::zero(), T::one()), self.scale(), self.shape())
    }
}

impl<T: Real> Support<T> for Rayleigh<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Rayleigh<T> {
    fn cdf(&self, x: T) -> T {
        kernels::rayleigh_cdf(x, self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::rayleigh_quantile(p.clamp(T::zero(), T::one()), self.scale())
    }
}

impl<T: Real> Support<T> for Wald<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Wald<T> {
    fn cdf(&self, x: T) -> T {
        kernels::wald_cdf(x, self.mu(), self.shape())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }
}

impl<T: Real> Support<T> for Weibull<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Weibull<T> {
    fn cdf(&self, x: T) -> T {
        kernels::weibull_cdf(x, self.scale(), self.shape())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::weibull_quantile(p.clamp(T::zero(), T::one()), self.scale(), self.shape())
    }
}

impl<T: Real> Support<T> for VonMises<T> {
    fn min(&self) -> T {
        -T::pi()
    }

    fn max(&self) -> T {
        T::pi()
    }
}

impl<N: Natural> Support<N> for Bernoulli<N::Real> {
    fn min(&self) -> N {
        N::zero()
    }

    fn max(&self) -> N {
        N::one()
    }
}

impl<N> DiscreteCdf<N> for Bernoulli<N::Real>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn cdf(&self, k: N) -> N::Real {
        let is_zero = k.eq(&N::zero());
        <N::Real as Compare>::select(&is_zero, N::Real::one() - self.p(), N::Real::one())
    }

    fn inverse_cdf(&self, p: N::Real) -> N {
        let threshold = N::Real::one() - self.p();
        N::select(&p.gt(&threshold), N::one(), N::zero())
    }
}

impl<N: Natural> Support<N> for Binomial<N> {
    fn min(&self) -> N {
        N::zero()
    }

    fn max(&self) -> N {
        self.trials()
    }
}

impl<N> DiscreteCdf<N> for Binomial<N>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn cdf(&self, k: N) -> N::Real {
        let at_or_above = k.ge(&self.trials());
        let one = N::Real::one();
        let k_real = k.to_real();
        let trials = self.trials().to_real();
        let interior =
            (one.clone() - self.p()).beta_reg(trials - k_real.clone(), k_real + one.clone());
        <N::Real as Compare>::select(&at_or_above, one, interior)
    }

    fn inverse_cdf(&self, p: N::Real) -> N {
        let n = self.trials();
        N::while_loop(
            N::zero(),
            |k| k.lt(&n).and(&self.cdf(k.clone()).lt(&p)),
            |k| k + N::one(),
        )
    }
}

impl<N: Natural> Support<N> for Geometric<N::Real> {
    fn min(&self) -> N {
        N::one()
    }

    fn max(&self) -> N {
        N::from_u64(u64::MAX)
    }
}

impl<N> DiscreteCdf<N> for Geometric<N::Real>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn cdf(&self, k: N) -> N::Real {
        let one = N::Real::one();
        one.clone() - (k.to_real() * (one - self.p()).ln()).exp()
    }

    fn inverse_cdf(&self, p: N::Real) -> N {
        N::while_loop(N::one(), |k| self.cdf(k.clone()).lt(&p), |k| k + N::one())
    }
}

impl<N: Natural> Support<N> for NegativeBinomial<N::Real> {
    fn min(&self) -> N {
        N::zero()
    }

    fn max(&self) -> N {
        N::from_u64(u64::MAX)
    }
}

impl<N> DiscreteCdf<N> for NegativeBinomial<N::Real>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn cdf(&self, k: N) -> N::Real {
        self.p()
            .beta_reg(self.successes(), k.to_real() + N::Real::one())
    }

    fn inverse_cdf(&self, p: N::Real) -> N {
        N::while_loop(N::zero(), |k| self.cdf(k.clone()).lt(&p), |k| k + N::one())
    }
}

impl<N: Natural> Support<N> for Poisson<N::Real> {
    fn min(&self) -> N {
        N::zero()
    }

    fn max(&self) -> N {
        N::from_u64(u64::MAX)
    }
}

impl<N> DiscreteCdf<N> for Poisson<N::Real>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn cdf(&self, k: N) -> N::Real {
        self.lambda().gamma_ur(k.to_real() + N::Real::one())
    }

    fn inverse_cdf(&self, p: N::Real) -> N {
        N::while_loop(N::zero(), |k| self.cdf(k.clone()).lt(&p), |k| k + N::one())
    }
}

impl<I: Integer> Support<I> for DiscreteUniform<I> {
    fn min(&self) -> I {
        self.lo()
    }

    fn max(&self) -> I {
        self.hi() - I::one()
    }
}

impl<I> IntegerDiscreteCdf<I> for DiscreteUniform<I>
where
    I: Integer + Compare,
    I::Real: Compare<Mask = <I as Compare>::Mask>,
{
    fn cdf(&self, k: I) -> I::Real {
        let below = k.lt(&self.lo());
        let at_or_above_last = k.ge(&(self.hi() - I::one()));
        let count = (self.hi() - self.lo()).to_real();
        let interior = (k - self.lo() + I::one()).to_real() / count;
        let clamped_above =
            <I::Real as Compare>::select(&at_or_above_last, I::Real::one(), interior);
        <I::Real as Compare>::select(&below, I::Real::zero(), clamped_above)
    }

    fn inverse_cdf(&self, p: I::Real) -> I {
        let hi = self.hi() - I::one();
        I::while_loop(
            self.lo(),
            |k| k.lt(&hi).and(&self.cdf(k.clone()).lt(&p)),
            |k| k + I::one(),
        )
    }
}
