use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChiSquared<R: Real = f64> {
    freedom: R,
}

impl<R: Real> ChiSquared<R> {
    pub fn new(freedom: R) -> Self {
        Self { freedom }
    }

    pub fn freedom(&self) -> R {
        self.freedom.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for ChiSquared<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let half_freedom = self.freedom() * T::half();
        kernels::chi_squared_log_pdf_on_support(x, half_freedom.clone(), half_freedom.ln_gamma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for ChiSquared<T>
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

impl<T: Real> Moments<T> for ChiSquared<T> {
    fn mean(&self) -> T {
        self.freedom()
    }

    fn variance(&self) -> T {
        T::two() * self.freedom()
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::chi_squared_skewness(self.freedom())
    }

    fn kurtosis(&self) -> T {
        kernels::chi_squared_excess_kurtosis(self.freedom())
    }

    fn entropy(&self) -> T {
        kernels::chi_squared_entropy(self.freedom() * T::half())
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        (self.freedom() - T::two()).max(T::zero())
    }
}
