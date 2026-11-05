use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weibull<R: Real = f64> {
    scale: R,
    shape: R,
}

impl<R: Real> Weibull<R> {
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

impl<T: Real> ContinuousDensityKernel<T> for Weibull<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::weibull_log_pdf_on_support(x, self.scale(), self.shape())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T> Continuous<T> for Weibull<T>
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

impl<T: Real> Moments<T> for Weibull<T> {
    fn mean(&self) -> T {
        self.scale() * (T::one() + T::one() / self.shape()).gamma()
    }

    fn variance(&self) -> T {
        let g1 = (T::one() + T::one() / self.shape()).gamma();
        let g2 = (T::one() + T::two() / self.shape()).gamma();
        self.scale() * self.scale() * (g2 - g1.clone() * g1)
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
        kernels::weibull_entropy(self.scale(), self.shape())
    }

    fn median(&self) -> T {
        self.scale() * T::ln_2().powf(T::one() / self.shape())
    }

    fn mode(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body =
            self.scale() * ((shape.clone() - T::one()) / shape.clone()).powf(T::one() / shape);
        T::select(&valid, body, T::zero())
    }
}
