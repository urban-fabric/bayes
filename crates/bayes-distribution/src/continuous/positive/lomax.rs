use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lomax<R: Real = f64> {
    shape: R,
    scale: R,
}

impl<R: Real> Lomax<R> {
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

impl<T: Real> ContinuousDensityKernel<T> for Lomax<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::lomax_log_pdf_on_support(x, self.shape(), self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for Lomax<T>
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

impl<T: Real> Moments<T> for Lomax<T> {
    fn mean(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body = self.scale() / (shape - T::one());
        T::select(&valid, body, T::nan())
    }

    fn variance(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::two());
        let body = self.scale() * self.scale() * shape.clone()
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
        T::nan()
    }

    fn median(&self) -> T {
        self.scale() * (T::two().powf(T::one() / self.shape()) - T::one())
    }

    fn mode(&self) -> T {
        T::zero()
    }
}
