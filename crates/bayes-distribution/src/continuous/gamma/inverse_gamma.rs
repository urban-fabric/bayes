use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseGamma<R: Real = f64> {
    shape: R,
    scale: R,
}

impl<R: Real> InverseGamma<R> {
    pub fn new(shape: R, scale: R) -> Self {
        Self { shape, scale }
    }

    pub fn shape(&self) -> R {
        self.shape.clone()
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for InverseGamma<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let shape = self.shape();
        kernels::inverse_gamma_log_pdf_on_support(x, shape.clone(), self.scale(), shape.ln_gamma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for InverseGamma<T>
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

impl<T: Real> Moments<T> for InverseGamma<T> {
    fn mean(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body = self.scale() / (shape - T::one());
        T::select(&valid, body, T::nan())
    }

    fn variance(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::two());
        let body = self.scale() * self.scale()
            / ((shape.clone() - T::one()) * (shape.clone() - T::one()) * (shape - T::two()));
        T::select(&valid, body, T::nan())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::inverse_gamma_entropy(self.shape(), self.scale())
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        self.scale() / (self.shape() + T::one())
    }
}
