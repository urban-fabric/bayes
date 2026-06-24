//! Closure-backed discrete adapters.

use crate::{Discrete, DiscreteCdf, Support};
use euclid::{Compare, Element, Natural, Real};
use std::fmt;

/// A closure-backed discrete mass function over explicit support bounds.
#[derive(Clone, Copy)]
pub struct DiscreteMass<
    X: Natural = u64,
    P = fn(X) -> <X as Natural>::Real,
    L = fn(X) -> <X as Natural>::Real,
> {
    min: X,
    max: X,
    pmf: P,
    ln_pmf: Option<L>,
}

impl<X, P, L> fmt::Debug for DiscreteMass<X, P, L>
where
    X: Natural + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscreteMass")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

impl<X, P> DiscreteMass<X, P, fn(X) -> X::Real>
where
    X: Natural,
{
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

impl<X, P, L> DiscreteMass<X, P, L>
where
    X: Natural,
{
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
    X: Copy + Natural,
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
    X: Copy + Natural,
    P: Fn(X) -> X::Real,
    L: Fn(X) -> X::Real,
{
    fn pmf(&self, k: X) -> X::Real {
        (self.pmf)(k)
    }

    fn ln_pmf(&self, k: X) -> X::Real {
        match &self.ln_pmf {
            Some(ln_pmf) => ln_pmf(k),
            None => self.pmf(k).ln(),
        }
    }
}

/// A closure-backed discrete distribution with PMF, CDF, and quantile.
#[derive(Clone, Copy)]
pub struct DiscreteFn<
    X: Natural = u64,
    P = fn(X) -> <X as Natural>::Real,
    C = fn(X) -> <X as Natural>::Real,
    Q = fn(<X as Natural>::Real) -> X,
    L = fn(X) -> <X as Natural>::Real,
> {
    mass: DiscreteMass<X, P, L>,
    cdf: C,
    inverse_cdf: Q,
}

impl<X, P, C, Q, L> fmt::Debug for DiscreteFn<X, P, C, Q, L>
where
    X: Natural + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscreteFn")
            .field("min", &self.mass.min)
            .field("max", &self.mass.max)
            .finish()
    }
}

impl<X, P, C, Q> DiscreteFn<X, P, C, Q, fn(X) -> X::Real>
where
    X: Natural,
{
    /// Build from PMF, CDF, quantile, and support bounds.
    pub fn new(min: X, max: X, pmf: P, cdf: C, inverse_cdf: Q) -> Self {
        Self {
            mass: DiscreteMass::from_pmf(min, max, pmf),
            cdf,
            inverse_cdf,
        }
    }
}

impl<X, P, C, Q, L> DiscreteFn<X, P, C, Q, L>
where
    X: Natural,
{
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
    X: Copy + Natural,
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
    X: Copy + Natural,
    P: Fn(X) -> X::Real,
    L: Fn(X) -> X::Real,
{
    fn pmf(&self, k: X) -> X::Real {
        self.mass.pmf(k)
    }

    fn ln_pmf(&self, k: X) -> X::Real {
        self.mass.ln_pmf(k)
    }
}

impl<X, P, C, Q, L> DiscreteCdf<X> for DiscreteFn<X, P, C, Q, L>
where
    X: Copy + Natural + Compare,
    X::Real: Compare<Mask = <X as Compare>::Mask>,
    C: Fn(X) -> X::Real,
    Q: Fn(X::Real) -> X,
{
    fn cdf(&self, k: X) -> X::Real {
        let below = k.lt(&self.min());
        let above = k.ge(&self.max());
        <X::Real as Compare>::select(
            &below,
            X::Real::zero(),
            <X::Real as Compare>::select(&above, X::Real::one(), (self.cdf)(k)),
        )
    }

    fn inverse_cdf(&self, p: X::Real) -> X {
        (self.inverse_cdf)(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_adapter_uses_associated_real_surface() {
        let coin = DiscreteFn::new(
            0_u32,
            1_u32,
            |k: u32| if k == 0 { 0.25_f32 } else { 0.75_f32 },
            |k: u32| if k == 0 { 0.25_f32 } else { 1.0_f32 },
            |p: f32| if p <= 0.25 { 0 } else { 1 },
        );

        assert_eq!(coin.pmf(1_u32), 0.75_f32);
        assert_eq!(coin.cdf(0_u32), 0.25_f32);
        assert_eq!(coin.inverse_cdf(0.9_f32), 1_u32);
    }
}
