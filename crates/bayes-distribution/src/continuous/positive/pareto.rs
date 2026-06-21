use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pareto<R: Real = f64> {
    scale: R,
    shape: R,
}

impl<R: Real> Pareto<R> {
    pub fn new(scale: R, shape: R) -> Self {
        Self { scale, shape }
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }

    pub fn shape(&self) -> R {
        self.shape.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for Pareto<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::pareto_log_pdf_on_support(x, self.scale(), self.shape())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for Pareto<T>
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let min = self.scale();
        let valid = x.ge(&min);
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

impl<T: Real> Moments<T> for Pareto<T> {
    fn mean(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body = shape.clone() * self.scale() / (shape - T::one());
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
        kernels::pareto_entropy(self.scale(), self.shape())
    }

    fn median(&self) -> T {
        self.scale() * T::two().powf(T::one() / self.shape())
    }

    fn mode(&self) -> T {
        self.scale()
    }
}
