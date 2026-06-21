//! Discrete distribution parameter types for the lazy layer.

use crate::continuous::Beta;
use euclid::{LazyInteger, LazyNatural, LazyReal};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bernoulli<R: LazyReal = f64> {
    p: R,
}

impl<R: LazyReal> Bernoulli<R> {
    pub fn new(p: R) -> Self {
        Self { p }
    }

    pub fn p(&self) -> R {
        self.p.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaBinomial<N: LazyNatural = u64> {
    n: N,
    alpha: N::Real,
    beta: N::Real,
}

impl<N: LazyNatural> BetaBinomial<N> {
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binomial<N: LazyNatural = u64> {
    n: N,
    p: N::Real,
}

impl<N: LazyNatural> Binomial<N> {
    pub fn new(n: N, p: N::Real) -> Self {
        Self { n, p }
    }

    pub fn trials(&self) -> N {
        self.n.clone()
    }

    pub fn p(&self) -> N::Real {
        self.p.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiscreteUniform<I: LazyInteger = i64> {
    lo: I,
    hi: I,
}

impl<I: LazyInteger> DiscreteUniform<I> {
    pub fn new(lo: I, hi: I) -> Self {
        Self { lo, hi }
    }

    pub fn lo(&self) -> I {
        self.lo.clone()
    }

    pub fn hi(&self) -> I {
        self.hi.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometric<R: LazyReal = f64> {
    p: R,
}

impl<R: LazyReal> Geometric<R> {
    pub fn new(p: R) -> Self {
        Self { p }
    }

    pub fn p(&self) -> R {
        self.p.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hypergeometric<N: LazyNatural = u64> {
    population: N,
    successes: N,
    draws: N,
}

impl<N: LazyNatural> Hypergeometric<N> {
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NegativeBinomial<R: LazyReal = f64> {
    r: R,
    p: R,
}

impl<R: LazyReal> NegativeBinomial<R> {
    pub fn new(successes: R, p: R) -> Self {
        Self { r: successes, p }
    }

    pub fn successes(&self) -> R {
        self.r.clone()
    }

    pub fn p(&self) -> R {
        self.p.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poisson<R: LazyReal = f64> {
    lambda: R,
}

impl<R: LazyReal> Poisson<R> {
    pub fn new(lambda: R) -> Self {
        Self { lambda }
    }

    pub fn lambda(&self) -> R {
        self.lambda.clone()
    }
}
