//! Discrete distribution parameter types for the lazy layer.

use crate::continuous::Beta;
use bayes_error::BayesError;
use euclid::{LazyInteger, LazyNatural, LazyReal};

fn finite(name: &'static str, value: f64) -> Result<f64, BayesError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(BayesError::NonFinite { name, value })
    }
}

fn probability(name: &'static str, value: f64) -> Result<f64, BayesError> {
    finite(name, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(BayesError::ProbabilityOutOfRange { value })
    }
}

fn positive(name: &'static str, value: f64) -> Result<f64, BayesError> {
    finite(name, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(BayesError::NotPositive { name, value })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bernoulli<R: LazyReal = f64> {
    p: R,
}

impl Bernoulli<f64> {
    pub fn new(p: f64) -> Result<Self, BayesError> {
        Ok(Self {
            p: probability("p", p)?,
        })
    }
}

impl<R: LazyReal> Bernoulli<R> {
    pub fn from_lazy(p: R) -> Self {
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

impl BetaBinomial<u64> {
    pub fn new(n: u64, alpha: f64, beta: f64) -> Result<Self, BayesError> {
        Ok(Self {
            n,
            alpha: positive("alpha", alpha)?,
            beta: positive("beta", beta)?,
        })
    }
}

impl<N: LazyNatural> BetaBinomial<N> {
    pub fn from_lazy(n: N, alpha: N::Real, beta: N::Real) -> Self {
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

impl Binomial<u64> {
    pub fn new(n: u64, p: f64) -> Result<Self, BayesError> {
        Ok(Self {
            n,
            p: probability("p", p)?,
        })
    }
}

impl<N: LazyNatural> Binomial<N> {
    pub fn from_lazy(n: N, p: N::Real) -> Self {
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

impl DiscreteUniform<i64> {
    pub fn new(lo: i64, hi: i64) -> Result<Self, BayesError> {
        if lo >= hi {
            return Err(BayesError::EmptyIntRange { lo, hi });
        }
        Ok(Self { lo, hi })
    }
}

impl<I: LazyInteger> DiscreteUniform<I> {
    pub fn from_lazy(lo: I, hi: I) -> Self {
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

impl Geometric<f64> {
    pub fn new(p: f64) -> Result<Self, BayesError> {
        finite("p", p)?;
        if p <= 0.0 || p > 1.0 {
            return Err(BayesError::ProbabilityOutOfRange { value: p });
        }
        Ok(Self { p })
    }
}

impl<R: LazyReal> Geometric<R> {
    pub fn from_lazy(p: R) -> Self {
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

impl Hypergeometric<u64> {
    pub fn new(population: u64, successes: u64, draws: u64) -> Result<Self, BayesError> {
        if successes > population {
            return Err(BayesError::Malformed {
                what: "successes must not exceed population",
            });
        }
        if draws > population {
            return Err(BayesError::Malformed {
                what: "draws must not exceed population",
            });
        }
        Ok(Self {
            population,
            successes,
            draws,
        })
    }
}

impl<N: LazyNatural> Hypergeometric<N> {
    pub fn from_lazy(population: N, successes: N, draws: N) -> Self {
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

impl NegativeBinomial<f64> {
    pub fn new(r: f64, p: f64) -> Result<Self, BayesError> {
        finite("p", p)?;
        if p <= 0.0 || p > 1.0 {
            return Err(BayesError::ProbabilityOutOfRange { value: p });
        }
        Ok(Self {
            r: positive("r", r)?,
            p,
        })
    }
}

impl<R: LazyReal> NegativeBinomial<R> {
    pub fn from_lazy(successes: R, p: R) -> Self {
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

impl Poisson<f64> {
    pub const MAX_LAMBDA: f64 = 1.844e19;

    pub fn new(lambda: f64) -> Result<Self, BayesError> {
        finite("lambda", lambda)?;
        if lambda <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "lambda",
                value: lambda,
            });
        }
        if lambda > Self::MAX_LAMBDA {
            return Err(BayesError::AboveMaximum {
                name: "lambda",
                value: lambda,
                max: Self::MAX_LAMBDA,
            });
        }
        Ok(Self { lambda })
    }
}

impl<R: LazyReal> Poisson<R> {
    pub fn from_lazy(lambda: R) -> Self {
        Self { lambda }
    }

    pub fn lambda(&self) -> R {
        self.lambda.clone()
    }
}
