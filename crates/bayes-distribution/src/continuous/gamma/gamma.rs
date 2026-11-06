use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gamma<R: Real = f64> {
    shape: R,
    rate: R,
}

impl<R: Real> Gamma<R> {
    pub fn new(shape: R, rate: R) -> Self {
        Self { shape, rate }
    }

    pub fn shape(&self) -> R {
        self.shape.clone()
    }

    pub fn rate(&self) -> R {
        self.rate.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for Gamma<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let shape = self.shape();
        kernels::gamma_log_pdf_on_support(x, shape.clone(), self.rate(), shape.ln_gamma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for Gamma<T>
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

impl<T: Real> Moments<T> for Gamma<T> {
    fn mean(&self) -> T {
        self.shape() / self.rate()
    }

    fn variance(&self) -> T {
        kernels::gamma_variance(self.shape(), self.rate())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::gamma_skewness(self.shape())
    }

    fn kurtosis(&self) -> T {
        kernels::gamma_excess_kurtosis(self.shape())
    }

    fn entropy(&self) -> T {
        kernels::gamma_entropy(self.shape(), self.rate())
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body = (shape - T::one()) / self.rate();
        T::select(&valid, body, T::zero())
    }
}
