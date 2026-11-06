use crate::{Discrete, DiscreteMassKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Natural, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poisson<R: Real = f64> {
    lambda: R,
}

impl<R: Real> Poisson<R> {
    pub fn new(lambda: R) -> Self {
        Self { lambda }
    }

    pub fn lambda(&self) -> R {
        self.lambda.clone()
    }
}

impl<N> DiscreteMassKernel<N> for Poisson<N::Real>
where
    N: Natural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let lambda = self.lambda();
        k.to_real() * lambda.clone().ln() - lambda - k.ln_factorial()
    }

    fn pmf_kernel(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<N: Natural> Discrete<N> for Poisson<N::Real> {
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k)
    }
}

impl<T: Real> Moments<T> for Poisson<T> {
    fn mean(&self) -> T {
        self.lambda()
    }

    fn variance(&self) -> T {
        self.lambda()
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::poisson_skewness(self.lambda())
    }

    fn kurtosis(&self) -> T {
        kernels::poisson_excess_kurtosis(self.lambda())
    }

    fn entropy(&self) -> T {
        T::nan()
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        self.lambda().floor()
    }
}
