use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::Real;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cauchy<R: Real = f64> {
    location: R,
    scale: R,
}

impl<R: Real> Cauchy<R> {
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

impl<T: Real> ContinuousDensityKernel<T> for Cauchy<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::cauchy_log_pdf(x, self.location(), self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: Real> Continuous<T> for Cauchy<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T: Real> Moments<T> for Cauchy<T> {
    fn mean(&self) -> T {
        T::nan()
    }

    fn variance(&self) -> T {
        T::nan()
    }

    fn std_dev(&self) -> T {
        T::nan()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::cauchy_entropy(self.scale())
    }

    fn median(&self) -> T {
        self.location()
    }

    fn mode(&self) -> T {
        self.location()
    }
}
