use super::{neg_inf, scalar};
use crate::discrete::{
    Bernoulli, BetaBinomial, Binomial, DiscreteUniform, Geometric, Hypergeometric,
    NegativeBinomial, Poisson,
};
use crate::{Discrete, DiscreteMassKernel, IntegerDiscrete, IntegerMassKernel};
use bayes_kernels as kernels;
use euclid::{LazyBool, LazyCompare, LazyInteger, LazyNatural, LazyReal};

fn select_real_for_natural<N>(
    mask: &<N as LazyCompare>::Mask,
    on_true: N::Real,
    on_false: N::Real,
) -> N::Real
where
    N: LazyNatural + LazyCompare,
    N::Real: LazyCompare<Mask = <N as LazyCompare>::Mask>,
{
    <N::Real as LazyCompare>::select(mask, on_true, on_false)
}

fn select_real_for_integer<I>(
    mask: &<I as LazyCompare>::Mask,
    on_true: I::Real,
    on_false: I::Real,
) -> I::Real
where
    I: LazyInteger + LazyCompare,
    I::Real: LazyCompare<Mask = <I as LazyCompare>::Mask>,
{
    <I::Real as LazyCompare>::select(mask, on_true, on_false)
}

fn ln_binomial_real<T: LazyReal>(n: T, k: T) -> T {
    let one = T::one();
    let n_minus_k = n.clone() - k.clone();
    (n + one.clone()).ln_gamma() - (k + one.clone()).ln_gamma() - (n_minus_k + one).ln_gamma()
}

impl<N: LazyNatural> DiscreteMassKernel<N> for Bernoulli {
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        kernels::bernoulli_log_pmf(k.to_real(), scalar(self.p()))
    }
}

impl<N: LazyNatural> DiscreteMassKernel<N> for BetaBinomial {
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let n = N::from_u64(self.trials());
        let k_real = k.to_real();
        let n_real = n.to_real();
        let alpha = scalar::<N::Real>(self.alpha());
        let beta = scalar::<N::Real>(self.beta());
        let ln_beta_posterior =
            (k_real.clone() + alpha.clone()).ln_beta(n_real - k_real + beta.clone());
        kernels::beta_binomial_log_pmf(n.ln_binomial(&k), ln_beta_posterior, alpha.ln_beta(beta))
    }
}

impl<N: LazyNatural> DiscreteMassKernel<N> for Binomial {
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let n = N::from_u64(self.trials());
        let p = scalar::<N::Real>(self.p());
        let k_real = k.to_real();
        let failures = n.to_real() - k_real.clone();
        n.ln_binomial(&k) + k_real * p.clone().ln() + failures * (N::Real::one() - p).ln()
    }
}

impl<N: LazyNatural> DiscreteMassKernel<N> for Geometric {
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let p = scalar::<N::Real>(self.p());
        (k.to_real() - N::Real::one()) * (N::Real::one() - p.clone()).ln() + p.ln()
    }
}

impl<N: LazyNatural> DiscreteMassKernel<N> for Hypergeometric {
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let population = scalar::<N::Real>(self.population() as f64);
        let successes = scalar::<N::Real>(self.successes() as f64);
        let draws = scalar::<N::Real>(self.draws() as f64);
        let k = k.to_real();
        ln_binomial_real(successes.clone(), k.clone())
            + ln_binomial_real(population.clone() - successes, draws.clone() - k)
            - ln_binomial_real(population, draws)
    }
}

impl<N: LazyNatural> DiscreteMassKernel<N> for NegativeBinomial {
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let r = scalar::<N::Real>(self.successes());
        let p = scalar::<N::Real>(self.p());
        let k_real = k.to_real();
        let ln_coefficient =
            (k_real.clone() + r.clone()).ln_gamma() - r.clone().ln_gamma() - k.ln_factorial();
        ln_coefficient + r * p.clone().ln() + k_real * (N::Real::one() - p).ln()
    }
}

impl<N: LazyNatural> DiscreteMassKernel<N> for Poisson {
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let lambda = scalar::<N::Real>(self.lambda());
        k.to_real() * lambda.clone().ln() - lambda - k.ln_factorial()
    }
}

impl<I: LazyInteger> IntegerMassKernel<I> for DiscreteUniform {
    fn ln_pmf_kernel(&self, _k: I) -> I::Real {
        kernels::discrete_uniform_log_pmf_on_support(scalar((self.hi() - self.lo()) as f64))
    }
}

impl<N> Discrete<N> for Bernoulli
where
    N: LazyNatural + LazyCompare,
    N::Real: LazyCompare<Mask = <N as LazyCompare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let zero_count = N::zero();
        let one_count = N::one();
        let zero = N::Real::zero();

        if self.p() <= 0.0 {
            let valid = k.eq(&zero_count);
            return select_real_for_natural::<N>(&valid, zero, neg_inf());
        }

        if self.p() >= 1.0 {
            let valid = k.eq(&one_count);
            return select_real_for_natural::<N>(&valid, zero, neg_inf());
        }

        let is_zero = k.eq(&zero_count);
        let is_one = k.eq(&one_count);
        let valid = is_zero.or(&is_one);
        select_real_for_natural::<N>(&valid, self.ln_pmf_kernel(k), neg_inf())
    }
}

impl<N> Discrete<N> for BetaBinomial
where
    N: LazyNatural + LazyCompare,
    N::Real: LazyCompare<Mask = <N as LazyCompare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let n = N::from_u64(self.trials());
        let valid = k.le(&n);
        select_real_for_natural::<N>(&valid, self.ln_pmf_kernel(k), neg_inf())
    }
}

impl<N> Discrete<N> for Binomial
where
    N: LazyNatural + LazyCompare,
    N::Real: LazyCompare<Mask = <N as LazyCompare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let zero_count = N::zero();
        let n = N::from_u64(self.trials());
        let zero = N::Real::zero();

        if self.p() <= 0.0 {
            let valid = k.eq(&zero_count);
            return select_real_for_natural::<N>(&valid, zero, neg_inf());
        }

        if self.p() >= 1.0 {
            let valid = k.eq(&n);
            return select_real_for_natural::<N>(&valid, zero, neg_inf());
        }

        let valid = k.le(&n);
        select_real_for_natural::<N>(&valid, self.ln_pmf_kernel(k), neg_inf())
    }
}

impl<N> Discrete<N> for Geometric
where
    N: LazyNatural + LazyCompare,
    N::Real: LazyCompare<Mask = <N as LazyCompare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let one_count = N::one();
        let zero = N::Real::zero();

        if self.p() >= 1.0 {
            let valid = k.eq(&one_count);
            return select_real_for_natural::<N>(&valid, zero, neg_inf());
        }

        let valid = k.ge(&one_count);
        select_real_for_natural::<N>(&valid, self.ln_pmf_kernel(k), neg_inf())
    }
}

impl<N> Discrete<N> for Hypergeometric
where
    N: LazyNatural + LazyCompare,
    N::Real: LazyCompare<Mask = <N as LazyCompare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let k_min =
            N::from_u64((self.draws() + self.successes()).saturating_sub(self.population()));
        let k_max = N::from_u64(self.draws().min(self.successes()));
        let valid = k.ge(&k_min).and(&k.le(&k_max));
        select_real_for_natural::<N>(&valid, self.ln_pmf_kernel(k), neg_inf())
    }
}

impl<N> Discrete<N> for NegativeBinomial
where
    N: LazyNatural + LazyCompare,
    N::Real: LazyCompare<Mask = <N as LazyCompare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        if self.p() >= 1.0 {
            let zero_count = N::zero();
            let valid = k.eq(&zero_count);
            return select_real_for_natural::<N>(&valid, N::Real::zero(), neg_inf());
        }

        self.ln_pmf_kernel(k)
    }
}

impl<N: LazyNatural> Discrete<N> for Poisson {
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k)
    }
}

impl<I> IntegerDiscrete<I> for DiscreteUniform
where
    I: LazyInteger + LazyCompare,
    I::Real: LazyCompare<Mask = <I as LazyCompare>::Mask>,
{
    fn pmf(&self, k: I) -> I::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: I) -> I::Real {
        let lo = I::from_i64(self.lo());
        let hi = I::from_i64(self.hi());
        let valid = k.ge(&lo).and(&k.lt(&hi));
        select_real_for_integer::<I>(&valid, self.ln_pmf_kernel(k), neg_inf())
    }
}
