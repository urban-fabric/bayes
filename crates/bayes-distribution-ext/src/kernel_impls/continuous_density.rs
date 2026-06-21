use super::scalar;
use crate::ContinuousDensityKernel;
use crate::continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel, HalfNormal,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh, StudentT, Uniform,
    VonMises, Wald, Weibull,
};
use bayes_kernels as kernels;
use euclid::LazyReal;

impl<T: LazyReal> ContinuousDensityKernel<T> for Beta {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let alpha = scalar::<T>(self.alpha());
        let beta = scalar::<T>(self.beta());
        kernels::beta_log_pdf_on_unit(x, alpha.clone(), beta.clone(), alpha.ln_beta(beta))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Cauchy {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::cauchy_log_pdf(x, scalar(self.location()), scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for ChiSquared {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let half_freedom = scalar::<T>(0.5 * self.freedom());
        kernels::chi_squared_log_pdf_on_support(x, half_freedom.clone(), half_freedom.ln_gamma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Exponential {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::exponential_log_pdf_on_support(x, scalar(self.rate()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for FisherSnedecor {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let d1 = scalar::<T>(self.d1());
        let d2 = scalar::<T>(self.d2());
        let half_d1 = d1.clone() * T::half();
        let half_d2 = d2.clone() * T::half();
        let ln_beta = half_d1.clone().ln_beta(half_d2.clone());
        kernels::fisher_snedecor_log_pdf_on_support(x, d1, d2, half_d1, half_d2, ln_beta)
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Frechet {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::frechet_log_pdf_on_support(x, scalar(self.scale()), scalar(self.shape()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Gamma {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let shape = scalar::<T>(self.shape());
        kernels::gamma_log_pdf_on_support(x, shape.clone(), scalar(self.rate()), shape.ln_gamma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Gumbel {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::gumbel_log_pdf(x, scalar(self.location()), scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for HalfNormal {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::half_normal_log_pdf_on_support(x, scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for InverseGamma {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let shape = scalar::<T>(self.shape());
        kernels::inverse_gamma_log_pdf_on_support(
            x,
            shape.clone(),
            scalar(self.scale()),
            shape.ln_gamma(),
        )
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Laplace {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::laplace_log_pdf(x, scalar(self.location()), scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Logistic {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::logistic_log_pdf(x, scalar(self.location()), scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for LogNormal {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::lognormal_log_pdf_on_support(x, scalar(self.mu()), scalar(self.sigma()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Lomax {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::lomax_log_pdf_on_support(x, scalar(self.shape()), scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Normal {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_log_pdf(x, scalar(mean), scalar(std_dev))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Pareto {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::pareto_log_pdf_on_support(x, scalar(self.scale()), scalar(self.shape()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Rayleigh {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::rayleigh_log_pdf_on_support(x, scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for StudentT {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let freedom = scalar::<T>(self.freedom());
        let log_normalizer = T::half() * (freedom.clone() * T::pi()).ln()
            + (freedom.clone() * T::half()).ln_gamma()
            - ((freedom.clone() + T::one()) * T::half()).ln_gamma();
        kernels::student_t_log_pdf(
            x,
            scalar(self.location()),
            scalar(self.scale()),
            freedom,
            log_normalizer,
        )
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Uniform {
    fn ln_pdf_kernel(&self, _x: T) -> T {
        kernels::uniform_log_pdf_on_support(scalar(self.hi() - self.lo()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for VonMises {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let kappa = scalar::<T>(self.kappa());
        let log_normalizer = T::ln_tau() + kappa.clone().ln_bessel_i0();
        kernels::von_mises_log_pdf(x, scalar(self.mu()), kappa, log_normalizer)
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Wald {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::wald_log_pdf_on_support(x, scalar(self.mu()), scalar(self.shape()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Weibull {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::weibull_log_pdf_on_support(x, scalar(self.scale()), scalar(self.shape()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}
