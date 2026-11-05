use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::Real;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normal<R: Real = f64> {
    mean: R,
    std_dev: R,
}

impl<R: Real> Normal<R> {
    pub fn new(mean: R, std_dev: R) -> Self {
        Self { mean, std_dev }
    }

    pub fn params(&self) -> (R, R) {
        (self.mean.clone(), self.std_dev.clone())
    }
}

impl<T: Real> ContinuousDensityKernel<T> for Normal<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_log_pdf(x, mean, std_dev)
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: Real> Continuous<T> for Normal<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T: Real> Moments<T> for Normal<T> {
    fn mean(&self) -> T {
        self.params().0
    }

    fn variance(&self) -> T {
        let std_dev = self.params().1;
        std_dev.clone() * std_dev
    }

    fn std_dev(&self) -> T {
        self.params().1
    }

    fn skewness(&self) -> T {
        T::zero()
    }

    fn kurtosis(&self) -> T {
        T::zero()
    }

    fn entropy(&self) -> T {
        kernels::normal_entropy(self.params().1)
    }

    fn median(&self) -> T {
        self.params().0
    }

    fn mode(&self) -> T {
        self.params().0
    }
}
