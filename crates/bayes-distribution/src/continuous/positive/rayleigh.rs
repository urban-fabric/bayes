use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rayleigh<R: Real = f64> {
    scale: R,
}

impl<R: Real> Rayleigh<R> {
    pub fn new(scale: R) -> Self {
        Self { scale }
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for Rayleigh<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::rayleigh_log_pdf_on_support(x, self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for Rayleigh<T>
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.ge(&zero);
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

impl<T: Real> Moments<T> for Rayleigh<T> {
    fn mean(&self) -> T {
        self.scale() * (T::pi() / T::two()).sqrt()
    }

    fn variance(&self) -> T {
        (T::scalar(4.0) - T::pi()) * self.scale() * self.scale() / T::two()
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::rayleigh_skewness()
    }

    fn kurtosis(&self) -> T {
        kernels::rayleigh_excess_kurtosis()
    }

    fn entropy(&self) -> T {
        kernels::rayleigh_entropy(self.scale())
    }

    fn median(&self) -> T {
        self.scale() * (T::two() * T::ln_2()).sqrt()
    }

    fn mode(&self) -> T {
        self.scale()
    }
}
