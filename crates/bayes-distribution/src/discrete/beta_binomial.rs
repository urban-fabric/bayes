use crate::continuous::Beta;
use crate::{Discrete, DiscreteMassKernel, Moments};
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Element, Natural, Real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaBinomial<N: Natural = u64> {
    n: N,
    alpha: N::Real,
    beta: N::Real,
}

impl<N: Natural> BetaBinomial<N> {
    pub fn new(n: N, alpha: N::Real, beta: N::Real) -> Self {
        Self { n, alpha, beta }
    }

    pub fn from_beta(trials: N, prior: Beta<N::Real>) -> Self {
        Self {
            n: trials,
            alpha: prior.alpha(),
            beta: prior.beta(),
        }
    }

    pub fn trials(&self) -> N {
        self.n.clone()
    }

    pub fn alpha(&self) -> N::Real {
        self.alpha.clone()
    }

    pub fn beta(&self) -> N::Real {
        self.beta.clone()
    }
}

impl<N> DiscreteMassKernel<N> for BetaBinomial<N>
where
    N: Natural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let n = self.trials();
        let k_real = k.to_real();
        let n_real = n.to_real();
        let alpha = self.alpha();
        let beta = self.beta();
        let ln_beta_posterior =
            (k_real.clone() + alpha.clone()).ln_beta(n_real - k_real + beta.clone());
        kernels::beta_binomial_log_pmf(n.ln_binomial(&k), ln_beta_posterior, alpha.ln_beta(beta))
    }

    fn pmf_kernel(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<N> Discrete<N> for BetaBinomial<N>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let n = self.trials();
        let valid = k.le(&n);
        <N::Real as Compare>::select(&valid, self.ln_pmf_kernel(k), N::Real::neg_infinity())
    }
}

impl<N: Natural> Moments<N::Real> for BetaBinomial<N> {
    fn mean(&self) -> N::Real {
        let sum = self.alpha() + self.beta();
        self.trials().to_real() * self.alpha() / sum
    }

    fn variance(&self) -> N::Real {
        let n = self.trials().to_real();
        let sum = self.alpha() + self.beta();
        n.clone() * self.alpha() * self.beta() * (sum.clone() + n)
            / (sum.clone() * sum.clone() * (sum + N::Real::one()))
    }

    fn std_dev(&self) -> N::Real {
        self.variance().sqrt()
    }

    fn skewness(&self) -> N::Real {
        N::Real::nan()
    }

    fn kurtosis(&self) -> N::Real {
        N::Real::nan()
    }

    fn entropy(&self) -> N::Real {
        N::Real::nan()
    }

    fn median(&self) -> N::Real {
        N::Real::nan()
    }

    fn mode(&self) -> N::Real {
        let alpha = self.alpha();
        let beta = self.beta();
        let valid = alpha.gt(&N::Real::one()).and(&beta.gt(&N::Real::one()));
        let body = self.trials().to_real() * (alpha.clone() - N::Real::one())
            / (alpha + beta - N::Real::two());
        N::Real::select(&valid, body, N::Real::nan())
    }
}
