use super::neg_inf;
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

impl<N> DiscreteMassKernel<N> for Bernoulli<N::Real>
where
    N: LazyNatural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        kernels::bernoulli_log_pmf(k.to_real(), self.p())
    }
}

impl<N> DiscreteMassKernel<N> for BetaBinomial<N>
where
    N: LazyNatural,
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
}

impl<N> DiscreteMassKernel<N> for Binomial<N>
where
    N: LazyNatural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let n = self.trials();
        let p = self.p();
        let k_real = k.to_real();
        let failures = n.to_real() - k_real.clone();
        n.ln_binomial(&k) + k_real * p.clone().ln() + failures * (N::Real::one() - p).ln()
    }
}

impl<N> DiscreteMassKernel<N> for Geometric<N::Real>
where
    N: LazyNatural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let p = self.p();
        (k.to_real() - N::Real::one()) * (N::Real::one() - p.clone()).ln() + p.ln()
    }
}

impl<N> DiscreteMassKernel<N> for Hypergeometric<N>
where
    N: LazyNatural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let population = self.population().to_real();
        let successes = self.successes().to_real();
        let draws = self.draws().to_real();
        let k = k.to_real();
        ln_binomial_real(successes.clone(), k.clone())
            + ln_binomial_real(population.clone() - successes, draws.clone() - k)
            - ln_binomial_real(population, draws)
    }
}

impl<N> DiscreteMassKernel<N> for NegativeBinomial<N::Real>
where
    N: LazyNatural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let r = self.successes();
        let p = self.p();
        let k_real = k.to_real();
        let ln_coefficient =
            (k_real.clone() + r.clone()).ln_gamma() - r.clone().ln_gamma() - k.ln_factorial();
        ln_coefficient + r * p.clone().ln() + k_real * (N::Real::one() - p).ln()
    }
}

impl<N> DiscreteMassKernel<N> for Poisson<N::Real>
where
    N: LazyNatural,
{
    fn ln_pmf_kernel(&self, k: N) -> N::Real {
        let lambda = self.lambda();
        k.to_real() * lambda.clone().ln() - lambda - k.ln_factorial()
    }
}

impl<I> IntegerMassKernel<I> for DiscreteUniform<I>
where
    I: LazyInteger,
{
    fn ln_pmf_kernel(&self, _k: I) -> I::Real {
        kernels::discrete_uniform_log_pmf_on_support((self.hi() - self.lo()).to_real())
    }
}

impl<N> Discrete<N> for Bernoulli<N::Real>
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
        let p = self.p();

        let is_zero = k.eq(&zero_count);
        let is_one = k.eq(&one_count);
        let valid = is_zero.or(&is_one);

        let zero_case = select_real_for_natural::<N>(&is_zero, zero.clone(), neg_inf());
        let one_case = select_real_for_natural::<N>(&is_one, zero, neg_inf());
        let regular = select_real_for_natural::<N>(&valid, self.ln_pmf_kernel(k), neg_inf());

        let p_is_zero = p.le(&N::Real::zero());
        let p_is_one = p.ge(&N::Real::one());
        select_real_for_natural::<N>(
            &p_is_zero,
            zero_case,
            select_real_for_natural::<N>(&p_is_one, one_case, regular),
        )
    }
}

impl<N> Discrete<N> for BetaBinomial<N>
where
    N: LazyNatural + LazyCompare,
    N::Real: LazyCompare<Mask = <N as LazyCompare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let n = self.trials();
        let valid = k.le(&n);
        select_real_for_natural::<N>(&valid, self.ln_pmf_kernel(k), neg_inf())
    }
}

impl<N> Discrete<N> for Binomial<N>
where
    N: LazyNatural + LazyCompare,
    N::Real: LazyCompare<Mask = <N as LazyCompare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let zero_count = N::zero();
        let n = self.trials();
        let zero = N::Real::zero();
        let p = self.p();

        let is_zero = k.eq(&zero_count);
        let is_n = k.eq(&n);
        let valid = k.le(&n);

        let zero_case = select_real_for_natural::<N>(&is_zero, zero.clone(), neg_inf());
        let one_case = select_real_for_natural::<N>(&is_n, zero, neg_inf());
        let regular = select_real_for_natural::<N>(&valid, self.ln_pmf_kernel(k), neg_inf());

        let p_is_zero = p.le(&N::Real::zero());
        let p_is_one = p.ge(&N::Real::one());
        select_real_for_natural::<N>(
            &p_is_zero,
            zero_case,
            select_real_for_natural::<N>(&p_is_one, one_case, regular),
        )
    }
}

impl<N> Discrete<N> for Geometric<N::Real>
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
        let p = self.p();

        let is_one = k.eq(&one_count);
        let valid = k.ge(&one_count);
        let one_case = select_real_for_natural::<N>(&is_one, zero, neg_inf());
        let regular = select_real_for_natural::<N>(&valid, self.ln_pmf_kernel(k), neg_inf());

        let p_is_one = p.ge(&N::Real::one());
        select_real_for_natural::<N>(&p_is_one, one_case, regular)
    }
}

impl<N> Discrete<N> for Hypergeometric<N>
where
    N: LazyNatural + LazyCompare,
    N::Real: LazyCompare<Mask = <N as LazyCompare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let population = self.population();
        let successes = self.successes();
        let draws = self.draws();
        let k_real = k.to_real();
        let lower_valid =
            (draws.to_real() - k_real.clone()).le(&(population.to_real() - successes.to_real()));
        let valid = k.le(&successes).and(&k.le(&draws)).and(&lower_valid);
        select_real_for_natural::<N>(&valid, self.ln_pmf_kernel(k), neg_inf())
    }
}

impl<N> Discrete<N> for NegativeBinomial<N::Real>
where
    N: LazyNatural + LazyCompare,
    N::Real: LazyCompare<Mask = <N as LazyCompare>::Mask>,
{
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        let zero_count = N::zero();
        let is_zero = k.eq(&zero_count);
        let one_case = select_real_for_natural::<N>(&is_zero, N::Real::zero(), neg_inf());
        let p_is_one = self.p().ge(&N::Real::one());
        select_real_for_natural::<N>(&p_is_one, one_case, self.ln_pmf_kernel(k))
    }
}

impl<N: LazyNatural> Discrete<N> for Poisson<N::Real> {
    fn pmf(&self, k: N) -> N::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k)
    }
}

impl<I> IntegerDiscrete<I> for DiscreteUniform<I>
where
    I: LazyInteger + LazyCompare,
    I::Real: LazyCompare<Mask = <I as LazyCompare>::Mask>,
{
    fn pmf(&self, k: I) -> I::Real {
        self.ln_pmf(k).exp()
    }

    fn ln_pmf(&self, k: I) -> I::Real {
        let lo = self.lo();
        let hi = self.hi();
        let valid = k.ge(&lo).and(&k.lt(&hi));
        select_real_for_integer::<I>(&valid, self.ln_pmf_kernel(k), neg_inf())
    }
}
