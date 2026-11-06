//! CDFs and quantiles for the discrete families.
//!
//! The quantile here is `discrete_inverse_cdf`: the smallest count whose CDF
//! reaches `p`, found by integer bisection carried out in the *real* surface
//! because `Natural` has no subtraction and `u64::MAX` would overflow a
//! midpoint.
use crate::discrete::{Bernoulli, Binomial, DiscreteUniform, Geometric, NegativeBinomial, Poisson};
use crate::{DiscreteCdf, IntegerDiscreteCdf, Support};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Element, Integer, Natural, Real};

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
        kernels::discrete_inverse_cdf(N::zero(), self.trials(), p, |k| self.cdf(k))
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
        kernels::discrete_inverse_cdf(N::one(), N::from_u64(u64::MAX), p, |k| self.cdf(k))
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
        kernels::discrete_inverse_cdf(N::zero(), N::from_u64(u64::MAX), p, |k| self.cdf(k))
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
        kernels::discrete_inverse_cdf(N::zero(), N::from_u64(u64::MAX), p, |k| self.cdf(k))
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
        // Closed form (a uniform needs no search): the smallest integer offset `j` with
        // `(j+1)/count ≥ p` is `ceil(p·count) − 1`, clamped to `[0, count−1]`; then `k = lo + j`.
        let count = (self.hi() - self.lo()).to_real();
        let one = I::Real::one();
        let j = ((p * count.clone()).ceil() - one.clone())
            .max(I::Real::zero())
            .min(count - one);
        self.lo() + I::from_real(j)
    }
}
