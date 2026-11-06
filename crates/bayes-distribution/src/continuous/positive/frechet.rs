use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frechet<R: Real = f64> {
    shape: R,
    scale: R,
}

impl<R: Real> Frechet<R> {
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

impl<T: Real> ContinuousDensityKernel<T> for Frechet<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::frechet_log_pdf_on_support(x, self.scale(), self.shape())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for Frechet<T>
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

impl<T: Real> Moments<T> for Frechet<T> {
    fn mean(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body = self.scale() * (T::one() - T::one() / shape).gamma();
        T::select(&valid, body, T::nan())
    }

    fn variance(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::two());
        let g1 = (T::one() - T::one() / shape.clone()).gamma();
        let g2 = (T::one() - T::two() / shape).gamma();
        let body = self.scale() * self.scale() * (g2 - g1.clone() * g1);
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
        self.scale() * T::ln_2().powf(-T::one() / self.shape())
    }

    fn mode(&self) -> T {
        let shape = self.shape();
        self.scale() * (shape.clone() / (shape.clone() + T::one())).powf(T::one() / shape)
    }
}
