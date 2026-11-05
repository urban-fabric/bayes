use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FisherSnedecor<R: Real = f64> {
    d1: R,
    d2: R,
}

impl<R: Real> FisherSnedecor<R> {
    pub fn new(d1: R, d2: R) -> Self {
        Self { d1, d2 }
    }

    pub fn d1(&self) -> R {
        self.d1.clone()
    }

    pub fn d2(&self) -> R {
        self.d2.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for FisherSnedecor<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let d1 = self.d1();
        let d2 = self.d2();
        let half_d1 = d1.clone() * T::half();
        let half_d2 = d2.clone() * T::half();
        let ln_beta = half_d1.clone().ln_beta(half_d2.clone());
        kernels::fisher_snedecor_log_pdf_on_support(x, d1, d2, half_d1, half_d2, ln_beta)
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for FisherSnedecor<T>
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

impl<T: Real> Moments<T> for FisherSnedecor<T> {
    fn mean(&self) -> T {
        let d2 = self.d2();
        let valid = d2.gt(&T::two());
        let body = d2.clone() / (d2 - T::two());
        T::select(&valid, body, T::nan())
    }

    fn variance(&self) -> T {
        let d1 = self.d1();
        let d2 = self.d2();
        let valid = d2.gt(&T::scalar(4.0));
        let body = T::two() * d2.clone() * d2.clone() * (d1.clone() + d2.clone() - T::two())
            / (d1 * (d2.clone() - T::two()) * (d2.clone() - T::two()) * (d2 - T::scalar(4.0)));
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
        T::nan()
    }

    fn mode(&self) -> T {
        let d1 = self.d1();
        let d2 = self.d2();
        let valid = d1.gt(&T::two());
        let body = (d1.clone() - T::two()) / d1 * d2.clone() / (d2 + T::two());
        T::select(&valid, body, T::zero())
    }
}
