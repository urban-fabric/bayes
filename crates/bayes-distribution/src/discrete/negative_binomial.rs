use crate::{Discrete, DiscreteMassKernel, Moments};
use bayes_kernels as kernels;
use euclid::{Compare, Element, Natural, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NegativeBinomial<R: Real = f64> {
    r: R,
    p: R,
}

impl<R: Real> NegativeBinomial<R> {
    pub fn new(successes: R, p: R) -> Self {
        Self { r: successes, p }
    }

    pub fn successes(&self) -> R {
        self.r.clone()
    }

    pub fn p(&self) -> R {
        self.p.clone()
    }
}

impl<N> DiscreteMassKernel<N> for NegativeBinomial<N::Real>
where
    N: Natural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let r = self.successes();
        let p = self.p();
        let k_real = k.to_real();
        let ln_coefficient =
            (k_real.clone() + r.clone()).ln_gamma() - r.clone().ln_gamma() - k.ln_factorial();
        ln_coefficient + r * p.clone().ln() + k_real * (N::Real::one() - p).ln()
    }

    fn pmf_kernel(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<N> Discrete<N> for NegativeBinomial<N::Real>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let zero_count = N::zero();
        let is_zero = k.eq(&zero_count);
        let one_case =
            <N::Real as Compare>::select(&is_zero, N::Real::zero(), N::Real::neg_infinity());
        let p_is_one = self.p().ge(&N::Real::one());
        <N::Real as Compare>::select(&p_is_one, one_case, self.ln_pmf_kernel(k))
    }
}

impl<T: Real> Moments<T> for NegativeBinomial<T> {
    fn mean(&self) -> T {
        self.successes() * (T::one() - self.p()) / self.p()
    }

    fn variance(&self) -> T {
        self.successes() * (T::one() - self.p()) / (self.p() * self.p())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::negative_binomial_skewness(self.successes(), self.p())
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        T::nan()
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        let successes = self.successes();
        let body = ((successes.clone() - T::one()) * (T::one() - self.p()) / self.p()).floor();
        T::select(&successes.gt(&T::one()), body, T::zero())
    }
}
