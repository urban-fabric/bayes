use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wald<R: Real = f64> {
    mu: R,
    shape: R,
}

impl<R: Real> Wald<R> {
    pub fn new(mu: R, shape: R) -> Self {
        Self { mu, shape }
    }

    pub fn mu(&self) -> R {
        self.mu.clone()
    }

    pub fn shape(&self) -> R {
        self.shape.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for Wald<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::wald_log_pdf_on_support(x, self.mu(), self.shape())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for Wald<T>
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

impl<T: Real> Moments<T> for Wald<T> {
    fn mean(&self) -> T {
        self.mu()
    }

    fn variance(&self) -> T {
        self.mu() * self.mu() * self.mu() / self.shape()
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::wald_skewness(self.mu(), self.shape())
    }

    fn kurtosis(&self) -> T {
        kernels::wald_excess_kurtosis(self.mu(), self.shape())
    }

    fn entropy(&self) -> T {
        T::nan()
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        let ratio = self.mu() / self.shape();
        self.mu()
            * ((T::one() + T::scalar(9.0) * ratio.clone() * ratio / T::scalar(4.0)).sqrt()
                - T::scalar(1.5) * self.mu() / self.shape())
    }
}
