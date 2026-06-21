use crate::{Discrete, DiscreteMassKernel, Moments};
use bayes_kernels as kernels;
use euclid::{Boolean, Compare, Element, Natural, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bernoulli<R: Real = f64> {
    p: R,
}

impl<R: Real> Bernoulli<R> {
    pub fn new(p: R) -> Self {
        Self { p }
    }

    pub fn p(&self) -> R {
        self.p.clone()
    }
}

impl<N> DiscreteMassKernel<N> for Bernoulli<N::Real>
where
    N: Natural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        kernels::bernoulli_log_pmf(k.to_real(), self.p())
    }

    fn pmf_kernel(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<N> Discrete<N> for Bernoulli<N::Real>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let zero_count = N::zero();
        let one_count = N::one();
        let zero = N::Real::zero();
        let p = self.p();

        let is_zero = k.eq(&zero_count);
        let is_one = k.eq(&one_count);
        let valid = is_zero.or(&is_one);

        let zero_case =
            <N::Real as Compare>::select(&is_zero, zero.clone(), N::Real::neg_infinity());
        let one_case = <N::Real as Compare>::select(&is_one, zero, N::Real::neg_infinity());
        let regular =
            <N::Real as Compare>::select(&valid, self.ln_pmf_kernel(k), N::Real::neg_infinity());

        let p_is_zero = p.le(&N::Real::zero());
        let p_is_one = p.ge(&N::Real::one());
        <N::Real as Compare>::select(
            &p_is_zero,
            zero_case,
            <N::Real as Compare>::select(&p_is_one, one_case, regular),
        )
    }
}

impl<T: Real> Moments<T> for Bernoulli<T> {
    fn mean(&self) -> T {
        self.p()
    }

    fn variance(&self) -> T {
        self.p() * (T::one() - self.p())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::bernoulli_skewness(self.p())
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        let p = self.p();
        let q = T::one() - p.clone();
        -p.clone().xlogy(p) - q.clone().xlogy(q)
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        let p = self.p();
        T::select(&p.gt(&T::half()), T::one(), T::zero())
    }
}
