use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid::Real;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Laplace<R: Real = f64> {
    location: R,
    scale: R,
}

impl<R: Real> Laplace<R> {
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

impl<T: Real> ContinuousDensityKernel<T> for Laplace<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::laplace_log_pdf(x, self.location(), self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: Real> Continuous<T> for Laplace<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T: Real> Moments<T> for Laplace<T> {
    fn mean(&self) -> T {
        self.location()
    }

    fn variance(&self) -> T {
        T::two() * self.scale() * self.scale()
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::zero()
    }

    fn kurtosis(&self) -> T {
        T::scalar(3.0)
    }

    fn entropy(&self) -> T {
        kernels::laplace_entropy(self.scale())
    }

    fn median(&self) -> T {
        self.location()
    }

    fn mode(&self) -> T {
        self.location()
    }
}
