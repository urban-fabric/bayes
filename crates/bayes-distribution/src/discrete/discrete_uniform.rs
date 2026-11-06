use crate::{IntegerDiscrete, IntegerMassKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Element, Integer, Real};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiscreteUniform<I: Integer = i64> {
    lo: I,
    hi: I,
}

impl<I: Integer> DiscreteUniform<I> {
    pub fn new(lo: I, hi: I) -> Self {
        Self { lo, hi }
    }

    pub fn lo(&self) -> I {
        self.lo.clone()
    }

    pub fn hi(&self) -> I {
        self.hi.clone()
    }
}

impl<I> IntegerMassKernel<I> for DiscreteUniform<I>
where
    I: Integer,
{
    fn ln_pmf_kernel(&self, _k: I) -> I::Real {
        kernels::discrete_uniform_log_pmf_on_support((self.hi() - self.lo()).to_real())
    }

    fn pmf_kernel(&self, k: I) -> I::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<I> IntegerDiscrete<I> for DiscreteUniform<I>
where
    I: Integer + Compare,
    I::Real: Compare<Mask = <I as Compare>::Mask>,
{
    fn pmf(&self, k: I) -> I::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: I) -> I::Real {
        let lo = self.lo();
        let hi = self.hi();
        let valid = k.ge(&lo).and(&k.lt(&hi));
        <I::Real as Compare>::select(&valid, self.ln_pmf_kernel(k), I::Real::neg_infinity())
    }
}

impl<I: Integer> Moments<I::Real> for DiscreteUniform<I> {
    fn mean(&self) -> I::Real {
        (self.lo().to_real() + self.hi().to_real() - I::Real::one()) * I::Real::half()
    }

    fn variance(&self) -> I::Real {
        let count = (self.hi() - self.lo()).to_real();
        (count.clone() * count - I::Real::one()) / I::Real::scalar(12.0)
    }

    fn std_dev(&self) -> I::Real {
        self.variance().sqrt()
    }

    fn skewness(&self) -> I::Real {
        I::Real::zero()
    }

    fn kurtosis(&self) -> I::Real {
        I::Real::nan()
    }

    fn entropy(&self) -> I::Real {
        kernels::discrete_uniform_entropy((self.hi() - self.lo()).to_real())
    }

    fn median(&self) -> I::Real {
        self.mean()
    }

    fn mode(&self) -> I::Real {
        I::Real::nan()
    }
}
