use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogNormal<R: Real = f64> {
    mu: R,
    sigma: R,
}

impl<R: Real> LogNormal<R> {
    pub fn new(mu: R, sigma: R) -> Self {
        Self { mu, sigma }
    }

    pub fn mu(&self) -> R {
        self.mu.clone()
    }

    pub fn sigma(&self) -> R {
        self.sigma.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for LogNormal<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::lognormal_log_pdf_on_support(x, self.mu(), self.sigma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for LogNormal<T>
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

impl<T: Real> Moments<T> for LogNormal<T> {
    fn mean(&self) -> T {
        kernels::lognormal_mean(self.mu(), self.sigma())
    }

    fn variance(&self) -> T {
        kernels::lognormal_variance(self.mu(), self.sigma())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::lognormal_skewness(self.sigma())
    }

    fn kurtosis(&self) -> T {
        kernels::lognormal_excess_kurtosis(self.sigma())
    }

    fn entropy(&self) -> T {
        kernels::lognormal_entropy(self.mu(), self.sigma())
    }

    fn median(&self) -> T {
        self.mu().exp()
    }

    fn mode(&self) -> T {
        let sigma = self.sigma();
        (self.mu() - sigma.clone() * sigma).exp()
    }
}
