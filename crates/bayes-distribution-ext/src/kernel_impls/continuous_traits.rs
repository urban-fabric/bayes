use super::{neg_inf, scalar};
use crate::continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel, HalfNormal,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh, StudentT, Uniform,
    VonMises, Wald, Weibull,
};
use crate::{Continuous, ContinuousDensityKernel};
use euclid::{LazyBool, LazyCompare, LazyReal};

fn select<T>(mask: &T::Mask, on_true: T, on_false: T) -> T
where
    T: LazyCompare,
{
    T::select(mask, on_true, on_false)
}

impl<T> Continuous<T> for Beta
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let one = T::one();
        let valid = x.gt(&zero).and(&x.lt(&one));
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T: LazyReal> Continuous<T> for Cauchy {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for ChiSquared
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T> Continuous<T> for Exponential
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.ge(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T> Continuous<T> for FisherSnedecor
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T> Continuous<T> for Frechet
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T> Continuous<T> for Gamma
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T: LazyReal> Continuous<T> for Gumbel {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for HalfNormal
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.ge(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T> Continuous<T> for InverseGamma
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T: LazyReal> Continuous<T> for Laplace {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T: LazyReal> Continuous<T> for Logistic {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for LogNormal
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T> Continuous<T> for Lomax
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.ge(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T: LazyReal> Continuous<T> for Normal {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for Pareto
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let min = scalar::<T>(self.scale());
        let valid = x.ge(&min);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T> Continuous<T> for Rayleigh
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.ge(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T: LazyReal> Continuous<T> for StudentT {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for Uniform
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let lo = scalar::<T>(self.lo());
        let hi = scalar::<T>(self.hi());
        let valid = x.ge(&lo).and(&x.le(&hi));
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T: LazyReal> Continuous<T> for VonMises {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for Wald
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T> Continuous<T> for Weibull
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.ge(&zero);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}
