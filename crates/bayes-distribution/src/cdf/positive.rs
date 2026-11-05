//! CDFs and quantiles for the families on a half-line or the unit interval.
//!
//! The ones that need a regularized incomplete gamma or beta, and whose
//! quantiles therefore come from the generic bisection in `bayes-kernels`
//! rather than from an elementary inverse. That bisection searches an
//! exponential coordinate, so it resolves quantiles many orders of magnitude
//! below the bracket — see `inverse_cdf_bisect`.
use crate::continuous::{
    Beta, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, HalfNormal, InverseGamma,
    LogNormal, Lomax, Pareto, Rayleigh, Wald, Weibull,
};
use crate::{ContinuousCdf, Support};
use bayes_kernels as kernels;
use euclid_numeric::Real;

impl<T: Real> Support<T> for Beta<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::one()
    }
}

impl<T: Real> ContinuousCdf<T> for Beta<T> {
    fn cdf(&self, x: T) -> T {
        kernels::beta_cdf_on_unit(x.clamp(T::zero(), T::one()), self.alpha(), self.beta())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::one(),
            p.clamp(T::zero(), T::one()),
        )
    }
}

impl<T: Real> Support<T> for Exponential<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Exponential<T> {
    fn cdf(&self, x: T) -> T {
        kernels::exponential_cdf(x, self.rate())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::exponential_quantile(p.clamp(T::zero(), T::one()), self.rate())
    }

    fn ln_cdf(&self, x: T) -> T {
        kernels::exponential_ln_cdf(x, self.rate())
    }

    fn ln_sf(&self, x: T) -> T {
        kernels::exponential_ln_sf(x, self.rate())
    }
}

impl<T: Real> Support<T> for Gamma<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Gamma<T> {
    fn cdf(&self, x: T) -> T {
        kernels::gamma_cdf(x, self.shape(), self.rate())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }

    /// `ln Q(k, βx)`, the upper regularized incomplete gamma.
    fn ln_sf(&self, x: T) -> T {
        kernels::gamma_ln_sf(x, self.shape(), self.rate())
    }
}

impl<T: Real> Support<T> for ChiSquared<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for ChiSquared<T> {
    fn cdf(&self, x: T) -> T {
        kernels::chi_squared_cdf(x, self.freedom() * T::half())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }

    /// `ln Q(k/2, x/2)`. The kernel takes **half** the degrees of freedom.
    fn ln_sf(&self, x: T) -> T {
        kernels::chi_squared_ln_sf(x, self.freedom() * T::half())
    }
}

impl<T: Real> Support<T> for FisherSnedecor<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for FisherSnedecor<T> {
    fn cdf(&self, x: T) -> T {
        kernels::fisher_snedecor_cdf(x, self.d1(), self.d2())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }

    /// `ln I_{d2/(d1x+d2)}(d2/2, d1/2)`: the survival, with no subtraction.
    fn ln_sf(&self, x: T) -> T {
        kernels::fisher_snedecor_ln_sf(x, self.d1(), self.d2())
    }
}

impl<T: Real> Support<T> for InverseGamma<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for InverseGamma<T> {
    fn cdf(&self, x: T) -> T {
        kernels::inverse_gamma_cdf(x, self.shape(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }

    /// `ln P(a, β/x)` — the **lower** incomplete gamma, since the CDF is upper.
    fn ln_sf(&self, x: T) -> T {
        kernels::inverse_gamma_ln_sf(x, self.shape(), self.scale())
    }
}

impl<T: Real> Support<T> for Frechet<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Frechet<T> {
    fn cdf(&self, x: T) -> T {
        kernels::frechet_cdf(x, self.scale(), self.shape())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::frechet_quantile(p.clamp(T::zero(), T::one()), self.scale(), self.shape())
    }

    fn ln_cdf(&self, x: T) -> T {
        kernels::frechet_ln_cdf(x, self.scale(), self.shape())
    }

    /// `ln(1 − e^{−t})` with `t = (x/s)^{−α}`.
    fn ln_sf(&self, x: T) -> T {
        kernels::frechet_ln_sf(x, self.scale(), self.shape())
    }
}

impl<T: Real> Support<T> for HalfNormal<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for HalfNormal<T> {
    fn cdf(&self, x: T) -> T {
        kernels::half_normal_cdf(x, self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::half_normal_quantile(p.clamp(T::zero(), T::one()), self.scale())
    }

    /// `ln erfc(x/(σ√2))` through `erfcx`, which does not underflow.
    fn ln_sf(&self, x: T) -> T {
        kernels::half_normal_ln_sf(x, self.scale())
    }
}

impl<T: Real> Support<T> for LogNormal<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for LogNormal<T> {
    fn cdf(&self, x: T) -> T {
        kernels::lognormal_cdf(x, self.mu(), self.sigma())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::lognormal_quantile(p.clamp(T::zero(), T::one()), self.mu(), self.sigma())
    }

    /// The normal's log-survival at `ln x`, inheriting its `erfcx` tail.
    fn ln_sf(&self, x: T) -> T {
        kernels::lognormal_ln_sf(x, self.mu(), self.sigma())
    }
}

impl<T: Real> Support<T> for Lomax<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Lomax<T> {
    fn cdf(&self, x: T) -> T {
        kernels::lomax_cdf(x, self.shape(), self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::lomax_quantile(p.clamp(T::zero(), T::one()), self.shape(), self.scale())
    }

    fn ln_sf(&self, x: T) -> T {
        kernels::lomax_ln_sf(x, self.shape(), self.scale())
    }
}

impl<T: Real> Support<T> for Pareto<T> {
    fn min(&self) -> T {
        self.scale()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Pareto<T> {
    fn cdf(&self, x: T) -> T {
        kernels::pareto_cdf(x, self.scale(), self.shape())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::pareto_quantile(p.clamp(T::zero(), T::one()), self.scale(), self.shape())
    }

    /// `−α·ln_1p((x − x_m)/x_m)`: linear in logs, exact at the minimum.
    fn ln_sf(&self, x: T) -> T {
        kernels::pareto_ln_sf(x, self.scale(), self.shape())
    }
}

impl<T: Real> Support<T> for Rayleigh<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Rayleigh<T> {
    fn cdf(&self, x: T) -> T {
        kernels::rayleigh_cdf(x, self.scale())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::rayleigh_quantile(p.clamp(T::zero(), T::one()), self.scale())
    }

    /// `−x²/(2σ²)`, exact.
    fn ln_sf(&self, x: T) -> T {
        kernels::rayleigh_ln_sf(x, self.scale())
    }
}

impl<T: Real> Support<T> for Wald<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Wald<T> {
    fn cdf(&self, x: T) -> T {
        kernels::wald_cdf(x, self.mu(), self.shape())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(
            |x| self.cdf(x),
            T::zero(),
            T::infinity(),
            p.clamp(T::zero(), T::one()),
        )
    }

    /// Cancels the `exp(2λ/μ)` factor the textbook form cannot evaluate.
    fn ln_sf(&self, x: T) -> T {
        kernels::wald_ln_sf(x, self.mu(), self.shape())
    }
}

impl<T: Real> Support<T> for Weibull<T> {
    fn min(&self) -> T {
        T::zero()
    }

    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Weibull<T> {
    fn cdf(&self, x: T) -> T {
        kernels::weibull_cdf(x, self.scale(), self.shape())
    }

    fn inverse_cdf(&self, p: T) -> T {
        kernels::weibull_quantile(p.clamp(T::zero(), T::one()), self.scale(), self.shape())
    }

    /// `−(x/λ)^k`, exact at every representable argument.
    fn ln_sf(&self, x: T) -> T {
        kernels::weibull_ln_sf(x, self.scale(), self.shape())
    }
}
