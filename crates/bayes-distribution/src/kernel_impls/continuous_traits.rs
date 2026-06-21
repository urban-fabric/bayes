use super::neg_inf;
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

impl<T> Continuous<T> for Beta<T>
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

impl<T: LazyReal> Continuous<T> for Cauchy<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for ChiSquared<T>
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

impl<T> Continuous<T> for Exponential<T>
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

impl<T> Continuous<T> for FisherSnedecor<T>
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

impl<T> Continuous<T> for Frechet<T>
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

impl<T> Continuous<T> for Gamma<T>
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

impl<T: LazyReal> Continuous<T> for Gumbel<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for HalfNormal<T>
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

impl<T> Continuous<T> for InverseGamma<T>
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

impl<T: LazyReal> Continuous<T> for Laplace<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T: LazyReal> Continuous<T> for Logistic<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for LogNormal<T>
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

impl<T> Continuous<T> for Lomax<T>
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

impl<T: LazyReal> Continuous<T> for Normal<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for Pareto<T>
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let min = self.scale();
        let valid = x.ge(&min);
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T> Continuous<T> for Rayleigh<T>
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

impl<T: LazyReal> Continuous<T> for StudentT<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for Uniform<T>
where
    T: LazyReal + LazyCompare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let lo = self.lo();
        let hi = self.hi();
        let valid = x.ge(&lo).and(&x.le(&hi));
        select(&valid, self.ln_pdf_kernel(x), neg_inf())
    }
}

impl<T: LazyReal> Continuous<T> for VonMises<T> {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        self.ln_pdf_kernel(x)
    }
}

impl<T> Continuous<T> for Wald<T>
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

impl<T> Continuous<T> for Weibull<T>
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
