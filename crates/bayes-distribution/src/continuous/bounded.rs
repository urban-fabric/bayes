use crate::{Continuous, ContinuousDensityKernel, Moments};
use bayes_kernels as kernels;
use euclid::{Boolean, Compare, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beta<R: Real = f64> {
    alpha: R,
    beta: R,
}

impl<R: Real> Beta<R> {
    pub fn new(alpha: R, beta: R) -> Self {
        Self { alpha, beta }
    }

    pub fn alpha(&self) -> R {
        self.alpha.clone()
    }

    pub fn beta(&self) -> R {
        self.beta.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniform<R: Real = f64> {
    lo: R,
    hi: R,
}

impl<R: Real> Uniform<R> {
    pub fn new(lo: R, hi: R) -> Self {
        Self { lo, hi }
    }

    pub fn lo(&self) -> R {
        self.lo.clone()
    }

    pub fn hi(&self) -> R {
        self.hi.clone()
    }
}

// Density kernels.
impl<T: Real> ContinuousDensityKernel<T> for Beta<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let alpha = self.alpha();
        let beta = self.beta();
        kernels::beta_log_pdf_on_unit(x, alpha.clone(), beta.clone(), alpha.ln_beta(beta))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: Real> ContinuousDensityKernel<T> for Uniform<T> {
    fn ln_pdf_kernel(&self, _x: T) -> T {
        kernels::uniform_log_pdf_on_support(self.hi() - self.lo())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for Beta<T>
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let one = T::one();
        let valid = x.gt(&zero).and(&x.lt(&one));
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

impl<T> Continuous<T> for Uniform<T>
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let lo = self.lo();
        let hi = self.hi();
        let valid = x.ge(&lo).and(&x.le(&hi));
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

// Moments.
impl<T: Real> Moments<T> for Beta<T> {
    fn mean(&self) -> T {
        let alpha = self.alpha();
        alpha.clone() / (alpha + self.beta())
    }

    fn variance(&self) -> T {
        kernels::beta_variance(self.alpha(), self.beta())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::beta_skewness(self.alpha(), self.beta())
    }

    fn kurtosis(&self) -> T {
        kernels::beta_excess_kurtosis(self.alpha(), self.beta())
    }

    fn entropy(&self) -> T {
        kernels::beta_entropy(self.alpha(), self.beta())
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        let alpha = self.alpha();
        let beta = self.beta();
        let valid = alpha.gt(&T::one()).and(&beta.gt(&T::one()));
        let body = (alpha.clone() - T::one()) / (alpha + beta - T::two());
        T::select(&valid, body, T::nan())
    }
}

impl<T: Real> Moments<T> for Uniform<T> {
    fn mean(&self) -> T {
        (self.lo() + self.hi()) * T::half()
    }

    fn variance(&self) -> T {
        let width = self.hi() - self.lo();
        width.clone() * width / T::scalar(12.0)
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::zero()
    }

    fn kurtosis(&self) -> T {
        T::scalar(-1.2)
    }

    fn entropy(&self) -> T {
        kernels::uniform_entropy(self.hi() - self.lo())
    }

    fn median(&self) -> T {
        self.mean()
    }

    fn mode(&self) -> T {
        T::nan()
    }
}
