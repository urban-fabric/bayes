//! CDFs and quantiles for the families supported on the whole real line, plus
//! the bounded ones whose CDF is elementary.
//!
//! These are the closed forms that need no special function — a `tanh`, an
//! `erf`, an exponential of an exponential — so each is a handful of lines and
//! an exact inverse.
use crate::continuous::{Cauchy, Gumbel, Laplace, Logistic, Normal, StudentT, Uniform, VonMises};
use crate::{ContinuousCdf, Support};
use bayes_kernels as kernels;
use euclid_numeric::Real;

impl<T: Real> Support<T> for Normal<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Normal<T> {
    fn cdf(&self, x: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_cdf(x, mean, std_dev)
    }

    fn inverse_cdf(&self, p: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_quantile(p.clamp(T::zero(), T::one()), mean, std_dev)
    }

    fn ln_cdf(&self, x: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_ln_cdf(x, mean, std_dev)
    }

    fn ln_sf(&self, x: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_ln_sf(x, mean, std_dev)
    }
}

impl<T: Real> Support<T> for Cauchy<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Cauchy<T> {
    fn cdf(&self, x: T) -> T {
        kernels::cauchy_cdf(x, self.location(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::cauchy_quantile(p.clamp(T::zero(), T::one()), self.location(), self.scale())
    }

    /// `atan(1/z)/π` above the location, which has no subtraction.
    fn ln_sf(&self, x: T) -> T {
        kernels::cauchy_ln_sf(x, self.location(), self.scale())
    }
}

impl<T: Real> Support<T> for Laplace<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Laplace<T> {
    fn cdf(&self, x: T) -> T {
        kernels::laplace_cdf(x, self.location(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::laplace_quantile(p.clamp(T::zero(), T::one()), self.location(), self.scale())
    }

    /// `−ln 2 − (x − μ)/b` above the location — linear, so exact everywhere.
    fn ln_sf(&self, x: T) -> T {
        kernels::laplace_ln_sf(x, self.location(), self.scale())
    }
}

impl<T: Real> Support<T> for Logistic<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Logistic<T> {
    fn cdf(&self, x: T) -> T {
        kernels::logistic_cdf(x, self.location(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::logistic_quantile(p.clamp(T::zero(), T::one()), self.location(), self.scale())
    }

    fn ln_cdf(&self, x: T) -> T {
        kernels::logistic_ln_cdf(x, self.location(), self.scale())
    }

    fn ln_sf(&self, x: T) -> T {
        kernels::logistic_ln_sf(x, self.location(), self.scale())
    }
}

impl<T: Real> Support<T> for Gumbel<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Gumbel<T> {
    fn cdf(&self, x: T) -> T {
        kernels::gumbel_cdf(x, self.location(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::gumbel_quantile(p.clamp(T::zero(), T::one()), self.location(), self.scale())
    }

    fn ln_cdf(&self, x: T) -> T {
        kernels::gumbel_ln_cdf(x, self.location(), self.scale())
    }

    /// `ln(1 − e^{−t})` with `t = e^{−z}`, through `ln_1m_exp`.
    fn ln_sf(&self, x: T) -> T {
        kernels::gumbel_ln_sf(x, self.location(), self.scale())
    }
}

impl<T: Real> Support<T> for StudentT<T> {
    fn min(&self) -> T {
        T::neg_infinity()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for StudentT<T> {
    fn cdf(&self, x: T) -> T {
        kernels::students_t_cdf(x, self.location(), self.scale(), self.freedom())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::neg_infinity(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }

    /// `ln(½·I_w(ν/2, ½))` above the location, keeping the `z^{−ν}` tail.
    fn ln_sf(&self, x: T) -> T {
        kernels::students_t_ln_sf(x, self.location(), self.scale(), self.freedom())
    }
}

impl<T: Real> Support<T> for Uniform<T> {
    fn min(&self) -> T {
        self.lo()
    }

    fn max(&self) -> T {
        self.hi()
    }
}

impl<T: Real> ContinuousCdf<T> for Uniform<T> {
    fn cdf(&self, x: T) -> T {
        kernels::uniform_cdf(x, self.lo(), self.hi() - self.lo())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::uniform_quantile(
            p.clamp(T::zero(), T::one()),
            self.lo(),
            self.hi() - self.lo(),
        )
    }
}

impl<T: Real> Support<T> for VonMises<T> {
    fn min(&self) -> T {
        -T::pi()
    }

    fn max(&self) -> T {
        T::pi()
    }
}
