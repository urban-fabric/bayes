use crate::ContinuousDensityKernel;
use crate::continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel, HalfNormal,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh, StudentT, Uniform,
    VonMises, Wald, Weibull,
};
use bayes_kernels as kernels;
use euclid::LazyReal;

impl<T: LazyReal> ContinuousDensityKernel<T> for Beta<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let alpha = self.alpha();
        let beta = self.beta();
        kernels::beta_log_pdf_on_unit(x, alpha.clone(), beta.clone(), alpha.ln_beta(beta))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Cauchy<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::cauchy_log_pdf(x, self.location(), self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for ChiSquared<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let half_freedom = self.freedom() * T::half();
        kernels::chi_squared_log_pdf_on_support(x, half_freedom.clone(), half_freedom.ln_gamma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Exponential<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::exponential_log_pdf_on_support(x, self.rate())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for FisherSnedecor<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let d1 = self.d1();
        let d2 = self.d2();
        let half_d1 = d1.clone() * T::half();
        let half_d2 = d2.clone() * T::half();
        let ln_beta = half_d1.clone().ln_beta(half_d2.clone());
        kernels::fisher_snedecor_log_pdf_on_support(x, d1, d2, half_d1, half_d2, ln_beta)
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Frechet<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::frechet_log_pdf_on_support(x, self.scale(), self.shape())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Gamma<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let shape = self.shape();
        kernels::gamma_log_pdf_on_support(x, shape.clone(), self.rate(), shape.ln_gamma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Gumbel<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::gumbel_log_pdf(x, self.location(), self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for HalfNormal<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::half_normal_log_pdf_on_support(x, self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for InverseGamma<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let shape = self.shape();
        kernels::inverse_gamma_log_pdf_on_support(x, shape.clone(), self.scale(), shape.ln_gamma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Laplace<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::laplace_log_pdf(x, self.location(), self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Logistic<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::logistic_log_pdf(x, self.location(), self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for LogNormal<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::lognormal_log_pdf_on_support(x, self.mu(), self.sigma())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Lomax<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::lomax_log_pdf_on_support(x, self.shape(), self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Normal<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let (mean, std_dev) = self.params();
        kernels::normal_log_pdf(x, mean, std_dev)
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Pareto<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::pareto_log_pdf_on_support(x, self.scale(), self.shape())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Rayleigh<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::rayleigh_log_pdf_on_support(x, self.scale())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for StudentT<T> {
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

impl<T: LazyReal> ContinuousDensityKernel<T> for Uniform<T> {
    fn ln_pdf_kernel(&self, _x: T) -> T {
        kernels::uniform_log_pdf_on_support(self.hi() - self.lo())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for VonMises<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let kappa = self.kappa();
        let log_normalizer = T::ln_tau() + kappa.clone().ln_bessel_i0();
        kernels::von_mises_log_pdf(x, self.mu(), kappa, log_normalizer)
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Wald<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::wald_log_pdf_on_support(x, self.mu(), self.shape())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

impl<T: LazyReal> ContinuousDensityKernel<T> for Weibull<T> {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::weibull_log_pdf_on_support(x, self.scale(), self.shape())
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}
