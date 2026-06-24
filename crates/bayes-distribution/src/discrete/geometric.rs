use crate::{Discrete, DiscreteMassKernel, Moments};
use bayes_kernels as kernels;
use euclid::{Compare, Element, Natural, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometric<R: Real = f64> {
    p: R,
}

impl<R: Real> Geometric<R> {
    pub fn new(p: R) -> Self {
        Self { p }
    }

    pub fn p(&self) -> R {
        self.p.clone()
    }
}

impl<N> DiscreteMassKernel<N> for Geometric<N::Real>
where
    N: Natural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let p = self.p();
        (k.to_real() - N::Real::one()) * (N::Real::one() - p.clone()).ln() + p.ln()
    }

    fn pmf_kernel(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<N> Discrete<N> for Geometric<N::Real>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let one_count = N::from_u64(1);
        let zero = N::Real::zero();
        let p = self.p();

        let is_one = k.eq(&one_count);
        let valid = k.ge(&one_count);
        let one_case = <N::Real as Compare>::select(&is_one, zero, N::Real::neg_infinity());
        let regular =
            <N::Real as Compare>::select(&valid, self.ln_pmf_kernel(k), N::Real::neg_infinity());

        let p_is_one = p.ge(&N::Real::one());
        <N::Real as Compare>::select(&p_is_one, one_case, regular)
    }
}

impl<T: Real> Moments<T> for Geometric<T> {
    fn mean(&self) -> T {
        self.p().recip()
    }

    fn variance(&self) -> T {
        (T::one() - self.p()) / (self.p() * self.p())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::geometric_skewness(self.p())
    }

    fn kurtosis(&self) -> T {
        kernels::geometric_excess_kurtosis(self.p())
    }

    fn entropy(&self) -> T {
        let p = self.p();
        let q = T::one() - p.clone();
        let body = (-q.clone() * q.ln() - p.clone() * p.clone().ln()) / p;
        T::select(&self.p().ge(&T::one()), T::zero(), body)
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        T::one()
    }
}
