use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid::Real;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Logistic<R: Real = f64> {
    location: R,
    scale: R,
}

impl<R: Real> Logistic<R> {
    pub fn new(location: R, scale: R) -> Self {
        Self { location, scale }
    }

    pub fn location(&self) -> R {
        self.location.clone()
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for Logistic<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::logistic_log_pdf(x, self.location(), self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: Real> Continuous<T> for Logistic<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T: Real> Moments<T> for Logistic<T> {
    fn mean(&self) -> T {
        self.location()
    }

    fn variance(&self) -> T {
        self.scale() * self.scale() * T::pi() * T::pi() / T::scalar(3.0)
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::zero()
    }

    fn kurtosis(&self) -> T {
        T::scalar(1.2)
    }

    fn entropy(&self) -> T {
        kernels::logistic_entropy(self.scale())
    }

    fn median(&self) -> T {
        self.location()
    }

    fn mode(&self) -> T {
        self.location()
    }
}
