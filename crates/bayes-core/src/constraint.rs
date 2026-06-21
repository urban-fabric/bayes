//! Lazy support constraints and unconstraining transforms.
//!
//! This module is value-level over `euclid` masks: membership predicates return
//! the carrier's mask type, and transforms/Jacobians are written over
//! [`Real`]. Host validation and error reporting live in the eager extension
//! crates.

use euclid::{Boolean, Compare, Integer, Natural, Real};

/// Whether a real support boundary includes its endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Boundary {
    Open,
    Closed,
}

/// A real-valued support constraint.
///
/// `Constraint` is intentionally generic in the boundary values so distributions
/// such as `Uniform<T>` and `Pareto<T>` can expose parameter-dependent lazy
/// supports.
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint<R = f64> {
    /// All finite real values.
    Real,
    /// `(0, infinity)`.
    Positive,
    /// `[0, infinity)`.
    NonNegative,
    /// `(0, 1)`.
    UnitInterval,
    /// `(lo, hi)`.
    Interval { lo: R, hi: R },
    /// `[lo, hi]`.
    ClosedInterval { lo: R, hi: R },
    /// A single lower-bounded real support.
    Lower { lo: R, boundary: Boundary },
    /// A single upper-bounded real support.
    Upper { hi: R, boundary: Boundary },
    /// A real support with independently chosen endpoint openness.
    Bounded {
        lo: R,
        hi: R,
        lower: Boundary,
        upper: Boundary,
    },
}

/// Explicit name for real-valued constraints.
pub type RealConstraint<R = f64> = Constraint<R>;

impl<R: Real> Constraint<R> {
    /// Whether `x` is in the support, returned as the carrier's mask type.
    pub fn contains(&self, x: R) -> R::Mask {
        match self {
            Constraint::Real => x.is_finite(),
            Constraint::Positive => x.gt(&R::zero()),
            Constraint::NonNegative => x.ge(&R::zero()),
            Constraint::UnitInterval => {
                let above = x.gt(&R::zero());
                let below = x.lt(&R::one());
                above.and(&below)
            }
            Constraint::Interval { lo, hi } => {
                let above = x.gt(lo);
                let below = x.lt(hi);
                above.and(&below)
            }
            Constraint::ClosedInterval { lo, hi } => {
                let above = x.ge(lo);
                let below = x.le(hi);
                above.and(&below)
            }
            Constraint::Lower { lo, boundary } => contains_lower(&x, lo, *boundary),
            Constraint::Upper { hi, boundary } => contains_upper(&x, hi, *boundary),
            Constraint::Bounded {
                lo,
                hi,
                lower,
                upper,
            } => {
                let above = contains_lower(&x, lo, *lower);
                let below = contains_upper(&x, hi, *upper);
                above.and(&below)
            }
        }
    }

    /// Whether `x` is a finite interior point where the smooth transform is defined.
    pub fn contains_interior(&self, x: R) -> R::Mask {
        match self {
            Constraint::Real => x.is_finite(),
            Constraint::Positive | Constraint::NonNegative => x.gt(&R::zero()),
            Constraint::UnitInterval => {
                let above = x.gt(&R::zero());
                let below = x.lt(&R::one());
                above.and(&below)
            }
            Constraint::Interval { lo, hi }
            | Constraint::ClosedInterval { lo, hi }
            | Constraint::Bounded { lo, hi, .. } => {
                let above = x.gt(lo);
                let below = x.lt(hi);
                above.and(&below)
            }
            Constraint::Lower { lo, .. } => x.gt(lo),
            Constraint::Upper { hi, .. } => x.lt(hi),
        }
    }

    /// Whether this real support has a smooth map to/from unconstrained real coordinates.
    pub fn has_continuous_transform(&self) -> bool {
        true
    }

    /// Map a constrained value to unconstrained real coordinates.
    pub fn to_unconstrained(&self, x: R) -> R {
        match self {
            Constraint::Real => x,
            Constraint::Positive | Constraint::NonNegative => x.ln(),
            Constraint::UnitInterval => x.logit(),
            Constraint::Interval { lo, hi }
            | Constraint::ClosedInterval { lo, hi }
            | Constraint::Bounded { lo, hi, .. } => {
                ((x - lo.clone()) / (hi.clone() - lo.clone())).logit()
            }
            Constraint::Lower { lo, .. } => (x - lo.clone()).ln(),
            Constraint::Upper { hi, .. } => (hi.clone() - x).ln(),
        }
    }

    /// Map an unconstrained real coordinate into this support's interior.
    pub fn to_constrained(&self, y: R) -> R {
        match self {
            Constraint::Real => y,
            Constraint::Positive | Constraint::NonNegative => y.exp(),
            Constraint::UnitInterval => y.sigmoid(),
            Constraint::Interval { lo, hi }
            | Constraint::ClosedInterval { lo, hi }
            | Constraint::Bounded { lo, hi, .. } => {
                lo.clone() + (hi.clone() - lo.clone()) * y.sigmoid()
            }
            Constraint::Lower { lo, .. } => lo.clone() + y.exp(),
            Constraint::Upper { hi, .. } => hi.clone() - y.exp(),
        }
    }

    /// `ln |d to_constrained(y) / dy|`.
    pub fn log_abs_det_jacobian(&self, y: R) -> R {
        match self {
            Constraint::Real => R::zero(),
            Constraint::Positive | Constraint::NonNegative => y,
            Constraint::UnitInterval => y.clone().ln_sigmoid() + y.ln_1m_sigmoid(),
            Constraint::Interval { lo, hi }
            | Constraint::ClosedInterval { lo, hi }
            | Constraint::Bounded { lo, hi, .. } => {
                (hi.clone() - lo.clone()).ln() + y.clone().ln_sigmoid() + y.ln_1m_sigmoid()
            }
            Constraint::Lower { .. } | Constraint::Upper { .. } => y,
        }
    }

    /// `ln |d to_unconstrained(x) / dx|`, in constrained coordinates.
    pub fn log_abs_det_inverse_jacobian(&self, x: R) -> R {
        match self {
            Constraint::Real => R::select(&x.is_finite(), R::zero(), R::nan()),
            Constraint::Positive => inverse_lower_jacobian(x, R::zero(), Boundary::Open),
            Constraint::NonNegative => inverse_lower_jacobian(x, R::zero(), Boundary::Closed),
            Constraint::UnitInterval => inverse_interval_jacobian(
                x,
                R::zero(),
                R::one(),
                Boundary::Closed,
                Boundary::Closed,
            ),
            Constraint::Interval { lo, hi } | Constraint::ClosedInterval { lo, hi } => {
                inverse_interval_jacobian(
                    x,
                    lo.clone(),
                    hi.clone(),
                    Boundary::Closed,
                    Boundary::Closed,
                )
            }
            Constraint::Lower { lo, boundary } => inverse_lower_jacobian(x, lo.clone(), *boundary),
            Constraint::Upper { hi, boundary } => inverse_upper_jacobian(x, hi.clone(), *boundary),
            Constraint::Bounded {
                lo,
                hi,
                lower,
                upper,
            } => inverse_interval_jacobian(x, lo.clone(), hi.clone(), *lower, *upper),
        }
    }
}

fn contains_lower<R: Real>(x: &R, lo: &R, boundary: Boundary) -> R::Mask {
    match boundary {
        Boundary::Open => x.gt(lo),
        Boundary::Closed => x.ge(lo),
    }
}

fn contains_upper<R: Real>(x: &R, hi: &R, boundary: Boundary) -> R::Mask {
    match boundary {
        Boundary::Open => x.lt(hi),
        Boundary::Closed => x.le(hi),
    }
}

fn inverse_lower_jacobian<R: Real>(x: R, lo: R, boundary: Boundary) -> R {
    let delta = x - lo;
    let positive = delta.gt(&R::zero());
    let at_boundary = delta.eq(&R::zero());
    let safe = R::select(&positive, delta, R::one());
    let outside = R::nan();
    let boundary_value = match boundary {
        Boundary::Open => outside.clone(),
        Boundary::Closed => R::infinity(),
    };
    R::select(
        &positive,
        -safe.ln(),
        R::select(&at_boundary, boundary_value, outside),
    )
}

fn inverse_upper_jacobian<R: Real>(x: R, hi: R, boundary: Boundary) -> R {
    let delta = hi - x;
    let positive = delta.gt(&R::zero());
    let at_boundary = delta.eq(&R::zero());
    let safe = R::select(&positive, delta, R::one());
    let outside = R::nan();
    let boundary_value = match boundary {
        Boundary::Open => outside.clone(),
        Boundary::Closed => R::infinity(),
    };
    R::select(
        &positive,
        -safe.ln(),
        R::select(&at_boundary, boundary_value, outside),
    )
}

fn inverse_interval_jacobian<R: Real>(x: R, lo: R, hi: R, lower: Boundary, upper: Boundary) -> R {
    let interior = x.gt(&lo).and(&x.lt(&hi));
    let at_lower = x.eq(&lo);
    let at_upper = x.eq(&hi);
    let midpoint = (lo.clone() + hi.clone()) * R::half();
    let safe = R::select(&interior, x, midpoint);
    let body = (hi.clone() - lo.clone()).ln() - (safe.clone() - lo).ln() - (hi - safe).ln();
    let lower_value = match lower {
        Boundary::Open => R::nan(),
        Boundary::Closed => R::infinity(),
    };
    let upper_value = match upper {
        Boundary::Open => R::nan(),
        Boundary::Closed => R::infinity(),
    };
    let boundary = R::select(
        &at_lower,
        lower_value,
        R::select(&at_upper, upper_value, R::nan()),
    );
    R::select(&interior, body, boundary)
}

/// A natural-count support constraint.
#[derive(Clone, Debug, PartialEq)]
pub enum NaturalConstraint<N = u64> {
    All,
    AtLeast { lo: N },
    AtMost { hi: N },
    Interval { lo: N, hi: N },
}

impl<N> NaturalConstraint<N>
where
    N: Natural + Compare,
{
    pub fn contains(&self, x: N) -> N::Mask {
        match self {
            NaturalConstraint::All => x.eq(&x),
            NaturalConstraint::AtLeast { lo } => x.ge(lo),
            NaturalConstraint::AtMost { hi } => x.le(hi),
            NaturalConstraint::Interval { lo, hi } => x.ge(lo).and(&x.le(hi)),
        }
    }
}

/// A signed-integer support constraint.
#[derive(Clone, Debug, PartialEq)]
pub enum IntegerConstraint<I = i64> {
    All,
    AtLeast { lo: I },
    AtMost { hi: I },
    Interval { lo: I, hi: I },
}

impl<I> IntegerConstraint<I>
where
    I: Integer + Compare,
{
    pub fn contains(&self, x: I) -> I::Mask {
        match self {
            IntegerConstraint::All => x.eq(&x),
            IntegerConstraint::AtLeast { lo } => x.ge(lo),
            IntegerConstraint::AtMost { hi } => x.le(hi),
            IntegerConstraint::Interval { lo, hi } => x.ge(lo).and(&x.le(hi)),
        }
    }
}

/// A distribution that can describe the support of its random variable.
pub trait Constrained {
    type Value;
    type Constraint;

    fn support_constraint(&self) -> Self::Constraint;
}

#[cfg(test)]
mod tests;
