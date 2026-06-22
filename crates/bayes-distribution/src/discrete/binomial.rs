use crate::{Discrete, DiscreteMassKernel, Moments};
use bayes_kernels as kernels;
use euclid::{Compare, Element, Natural, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binomial<N: Natural = u64> {
    n: N,
    p: N::Real,
}

impl<N: Natural> Binomial<N> {
    pub fn new(n: N, p: N::Real) -> Self {
        Self { n, p }
    }

    pub fn trials(&self) -> N {
        self.n.clone()
    }

    pub fn p(&self) -> N::Real {
        self.p.clone()
    }
}

impl<N> DiscreteMassKernel<N> for Binomial<N>
where
    N: Natural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let n = self.trials();
        let p = self.p();
        let k_real = k.to_real();
        let failures = n.to_real() - k_real.clone();
        n.ln_binomial(&k) + k_real * p.clone().ln() + failures * (N::Real::one() - p).ln()
    }

    fn pmf_kernel(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<N> Discrete<N> for Binomial<N>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let zero_count = N::from_u64(0);
        let n = self.trials();
        let zero = N::Real::zero();
        let p = self.p();

        let is_zero = k.eq(&zero_count);
        let is_n = k.eq(&n);
        let valid = k.le(&n);

        let zero_case =
            <N::Real as Compare>::select(&is_zero, zero.clone(), N::Real::neg_infinity());
        let one_case = <N::Real as Compare>::select(&is_n, zero, N::Real::neg_infinity());
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

impl<N: Natural> Moments<N::Real> for Binomial<N> {
    fn mean(&self) -> N::Real {
        self.trials().to_real() * self.p()
    }

    fn variance(&self) -> N::Real {
        self.trials().to_real() * self.p() * (N::Real::one() - self.p())
    }

    fn std_dev(&self) -> N::Real {
        self.variance().sqrt()
    }

    fn skewness(&self) -> N::Real {
        kernels::binomial_skewness(self.trials().to_real(), self.p())
    }

    fn kurtosis(&self) -> N::Real {
        N::Real::nan()
    }

    fn entropy(&self) -> N::Real {
        N::Real::nan()
    }

    fn median(&self) -> N::Real {
        N::Real::nan()
    }

    fn mode(&self) -> N::Real {
        ((self.trials().to_real() + N::Real::one()) * self.p()).floor()
    }
}
