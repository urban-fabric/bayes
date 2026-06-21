//! Closure-backed **discrete** adapters: lift a PMF (and optional log-PMF, CDF,
//! and quantile) into the [`Discrete`] / [`DiscreteCdf`] spine.

use crate::{Discrete, DiscreteCdf, Support};
use euclid::Natural;
use std::fmt;

/// A closure-backed discrete mass function over explicit support bounds.
#[derive(Clone, Copy)]
pub struct DiscreteMass<X, P, L = fn(X) -> f64> {
    min: X,
    max: X,
    pmf: P,
    ln_pmf: Option<L>,
}

impl<X, P, L> fmt::Debug for DiscreteMass<X, P, L>
where
    X: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscreteMass")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

impl<X, P> DiscreteMass<X, P, fn(X) -> f64> {
    /// Build from a PMF and support bounds.
    pub fn from_pmf(min: X, max: X, pmf: P) -> Self {
        Self {
            min,
            max,
            pmf,
            ln_pmf: None,
        }
    }
}

impl<X, P, L> DiscreteMass<X, P, L> {
    /// Build from a PMF plus a numerically stable log-PMF.
    pub fn from_pmf_and_ln_pmf(min: X, max: X, pmf: P, ln_pmf: L) -> Self {
        Self {
            min,
            max,
            pmf,
            ln_pmf: Some(ln_pmf),
        }
    }
}

impl<X, P, L> Support<X> for DiscreteMass<X, P, L>
where
    X: Copy,
{
    fn min(&self) -> X {
        self.min
    }
    fn max(&self) -> X {
        self.max
    }
}

impl<X, P, L> Discrete<X> for DiscreteMass<X, P, L>
where
    X: Copy + Natural<Real = f64>,
    P: Fn(X) -> f64,
    L: Fn(X) -> f64,
{
    fn pmf(&self, k: X) -> f64 {
        (self.pmf)(k)
    }

    fn ln_pmf(&self, k: X) -> f64 {
        match &self.ln_pmf {
            Some(ln_pmf) => ln_pmf(k),
            None => self.pmf(k).ln(),
        }
    }
}

/// A closure-backed discrete distribution with PMF, CDF, and quantile.
#[derive(Clone, Copy)]
pub struct DiscreteFn<X, P, C, Q, L = fn(X) -> f64> {
    mass: DiscreteMass<X, P, L>,
    cdf: C,
    inverse_cdf: Q,
}

impl<X, P, C, Q, L> fmt::Debug for DiscreteFn<X, P, C, Q, L>
where
    X: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscreteFn")
            .field("min", &self.mass.min)
            .field("max", &self.mass.max)
            .finish()
    }
}

impl<X, P, C, Q> DiscreteFn<X, P, C, Q, fn(X) -> f64> {
    /// Build from PMF, CDF, quantile, and support bounds.
    pub fn new(min: X, max: X, pmf: P, cdf: C, inverse_cdf: Q) -> Self {
        Self {
            mass: DiscreteMass::from_pmf(min, max, pmf),
            cdf,
            inverse_cdf,
        }
    }
}

impl<X, P, C, Q, L> DiscreteFn<X, P, C, Q, L> {
    /// Build from PMF, stable log-PMF, CDF, quantile, and support bounds.
    pub fn with_ln_pmf(min: X, max: X, pmf: P, ln_pmf: L, cdf: C, inverse_cdf: Q) -> Self {
        Self {
            mass: DiscreteMass::from_pmf_and_ln_pmf(min, max, pmf, ln_pmf),
            cdf,
            inverse_cdf,
        }
    }
}

impl<X, P, C, Q, L> Support<X> for DiscreteFn<X, P, C, Q, L>
where
    X: Copy,
{
    fn min(&self) -> X {
        self.mass.min()
    }
    fn max(&self) -> X {
        self.mass.max()
    }
}

impl<X, P, C, Q, L> Discrete<X> for DiscreteFn<X, P, C, Q, L>
where
    X: Copy + Natural<Real = f64>,
    P: Fn(X) -> f64,
    L: Fn(X) -> f64,
{
    fn pmf(&self, k: X) -> f64 {
        self.mass.pmf(k)
    }

    fn ln_pmf(&self, k: X) -> f64 {
        self.mass.ln_pmf(k)
    }
}

impl<X, P, C, Q, L> DiscreteCdf<X> for DiscreteFn<X, P, C, Q, L>
where
    X: Copy + PartialOrd + Natural<Real = f64>,
    C: Fn(X) -> f64,
    Q: Fn(f64) -> X,
{
    fn cdf(&self, k: X) -> f64 {
        if k < self.min() {
            0.0
        } else if k >= self.max() {
            1.0
        } else {
            (self.cdf)(k)
        }
    }

    fn inverse_cdf(&self, p: f64) -> X {
        (self.inverse_cdf)(p)
    }
}
