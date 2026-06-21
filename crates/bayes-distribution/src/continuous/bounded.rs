use super::{finite, positive};
use bayes_error::BayesError;
use euclid::LazyReal;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beta<R: LazyReal = f64> {
    alpha: R,
    beta: R,
}

impl Beta<f64> {
    pub fn new(alpha: f64, beta: f64) -> Result<Self, BayesError> {
        Ok(Self {
            alpha: positive("alpha", alpha)?,
            beta: positive("beta", beta)?,
        })
    }
}

impl<R: LazyReal> Beta<R> {
    pub fn from_lazy(alpha: R, beta: R) -> Self {
        Self { alpha, beta }
    }

    pub fn alpha(&self) -> R {
        self.alpha.clone()
    }

    pub fn beta(&self) -> R {
        self.beta.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniform<R: LazyReal = f64> {
    lo: R,
    hi: R,
}

impl Uniform<f64> {
    pub fn new(lo: f64, hi: f64) -> Result<Self, BayesError> {
        finite("lo", lo)?;
        finite("hi", hi)?;
        if lo >= hi {
            return Err(BayesError::EmptyRange { lo, hi });
        }
        Ok(Self { lo, hi })
    }
}

impl<R: LazyReal> Uniform<R> {
    pub fn from_lazy(lo: R, hi: R) -> Self {
        Self { lo, hi }
    }

    pub fn lo(&self) -> R {
        self.lo.clone()
    }

    pub fn hi(&self) -> R {
        self.hi.clone()
    }
}
