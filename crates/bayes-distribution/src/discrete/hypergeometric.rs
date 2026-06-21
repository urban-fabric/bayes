use crate::{Discrete, DiscreteMassKernel, Moments};
use euclid::{Boolean, Compare, Element, Natural, Real};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hypergeometric<N: Natural = u64> {
    population: N,
    successes: N,
    draws: N,
}

impl<N: Natural> Hypergeometric<N> {
    pub fn new(population: N, successes: N, draws: N) -> Self {
        Self {
            population,
            successes,
            draws,
        }
    }

    pub fn population(&self) -> N {
        self.population.clone()
    }

    pub fn successes(&self) -> N {
        self.successes.clone()
    }

    pub fn draws(&self) -> N {
        self.draws.clone()
    }
}

impl<N> DiscreteMassKernel<N> for Hypergeometric<N>
where
    N: Natural,
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

    fn pmf_kernel(&self, k: N) -> N::Real {
        self.ln_pmf_kernel(k).exp()
    }
}

impl<N> Discrete<N> for Hypergeometric<N>
where
    N: Natural + Compare,
    N::Real: Compare<Mask = <N as Compare>::Mask>,
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
        <N::Real as Compare>::select(&valid, self.ln_pmf_kernel(k), N::Real::neg_infinity())
    }
}

impl<N: Natural> Moments<N::Real> for Hypergeometric<N> {
    fn mean(&self) -> N::Real {
        self.draws().to_real() * self.successes().to_real() / self.population().to_real()
    }

    fn variance(&self) -> N::Real {
        let population = self.population().to_real();
        let successes = self.successes().to_real();
        let draws = self.draws().to_real();
        let body = draws.clone()
            * (successes.clone() / population.clone())
            * ((population.clone() - successes) / population.clone())
            * ((population.clone() - draws) / (population.clone() - N::Real::one()));
        N::Real::select(&population.le(&N::Real::one()), N::Real::zero(), body)
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
        N::Real::nan()
    }
}

fn ln_binomial_real<T: Real>(n: T, k: T) -> T {
    let one = T::one();
    let n_minus_k = n.clone() - k.clone();
    (n + one.clone()).ln_gamma() - (k + one.clone()).ln_gamma() - (n_minus_k + one).ln_gamma()
}
