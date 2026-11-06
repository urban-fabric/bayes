use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfNormal<R: Real = f64> {
    scale: R,
}

impl<R: Real> HalfNormal<R> {
    pub fn new(scale: R) -> Self {
        Self { scale }
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for HalfNormal<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::half_normal_log_pdf_on_support(x, self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for HalfNormal<T>
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

impl<T: Real> Moments<T> for HalfNormal<T> {
    fn mean(&self) -> T {
        self.scale() * (T::two() / T::pi()).sqrt()
    }

    fn variance(&self) -> T {
        self.scale() * self.scale() * (T::one() - T::two() / T::pi())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::half_normal_skewness()
    }

    fn kurtosis(&self) -> T {
        kernels::half_normal_excess_kurtosis()
    }

    fn entropy(&self) -> T {
        kernels::half_normal_entropy(self.scale())
    }

    fn median(&self) -> T {
        self.scale() * T::sqrt_2() * T::half().erf_inv()
    }

    fn mode(&self) -> T {
        T::zero()
    }
}
