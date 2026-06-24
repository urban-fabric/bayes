use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid::Real;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudentT<R: Real = f64> {
    freedom: R,
    location: R,
    scale: R,
}

impl<R: Real> StudentT<R> {
    pub fn new(freedom: R) -> Self {
        Self {
            freedom,
            location: R::zero(),
            scale: R::one(),
        }
    }

    pub fn new_location_scale(freedom: R, location: R, scale: R) -> Self {
        Self {
            freedom,
            location,
            scale,
        }
    }

    pub fn freedom(&self) -> R {
        self.freedom.clone()
    }

    pub fn location(&self) -> R {
        self.location.clone()
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for StudentT<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let freedom = self.freedom();
        let log_normalizer = T::half() * (freedom.clone() * T::pi()).ln()
            + (freedom.clone() * T::half()).ln_gamma()
            - ((freedom.clone() + T::one()) * T::half()).ln_gamma();
        kernels::student_t_log_pdf(x, self.location(), self.scale(), freedom, log_normalizer)
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: Real> Continuous<T> for StudentT<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T: Real> Moments<T> for StudentT<T> {
    fn mean(&self) -> T {
        let freedom = self.freedom();
        let valid = freedom.gt(&T::one());
        T::select(&valid, self.location(), T::nan())
    }

    fn variance(&self) -> T {
        let freedom = self.freedom();
        let valid = freedom.gt(&T::two());
        let body = self.scale() * self.scale() * freedom.clone() / (freedom - T::two());
        T::select(&valid, body, T::nan())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        let freedom = self.freedom();
        let valid = freedom.gt(&T::scalar(3.0));
        T::select(&valid, T::zero(), T::nan())
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::students_t_entropy(self.scale(), self.freedom())
    }

    fn median(&self) -> T {
        self.location()
    }

    fn mode(&self) -> T {
        self.location()
    }
}
