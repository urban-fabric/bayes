use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exponential<R: Real = f64> {
    rate: R,
}

impl<R: Real> Exponential<R> {
    pub fn new(rate: R) -> Self {
        Self { rate }
    }

    pub fn rate(&self) -> R {
        self.rate.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for Exponential<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::exponential_log_pdf_on_support(x, self.rate())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for Exponential<T>
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

impl<T: Real> Moments<T> for Exponential<T> {
    fn mean(&self) -> T {
        self.rate().recip()
    }

    fn variance(&self) -> T {
        let rate = self.rate();
        T::one() / (rate.clone() * rate)
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::two()
    }

    fn kurtosis(&self) -> T {
        T::scalar(6.0)
    }

    fn entropy(&self) -> T {
        kernels::exponential_entropy(self.rate())
    }

    fn median(&self) -> T {
        T::ln_2() / self.rate()
    }

    fn mode(&self) -> T {
        T::zero()
    }
}
