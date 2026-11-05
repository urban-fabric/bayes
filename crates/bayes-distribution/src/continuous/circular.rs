use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::Real;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VonMises<R: Real = f64> {
    mu: R,
    kappa: R,
}

impl<R: Real> VonMises<R> {
    pub fn new(mu: R, kappa: R) -> Self {
        Self { mu, kappa }
    }

    pub fn mu(&self) -> R {
        self.mu.clone()
    }

    pub fn kappa(&self) -> R {
        self.kappa.clone()
    }
}

// Density kernels.
impl<T: Real> ContinuousDensityKernel<T> for VonMises<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let kappa = self.kappa();
        let log_normalizer = T::ln_tau() + kappa.clone().ln_bessel_i0();
        kernels::von_mises_log_pdf(x, self.mu(), kappa, log_normalizer)
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T: Real> Continuous<T> for VonMises<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

// Moments.
impl<T: Real> Moments<T> for VonMises<T> {
    fn mean(&self) -> T {
        self.mu()
    }

    fn variance(&self) -> T {
        T::nan()
    }

    fn std_dev(&self) -> T {
        T::nan()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::von_mises_entropy(self.kappa())
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        self.mu()
    }
}
