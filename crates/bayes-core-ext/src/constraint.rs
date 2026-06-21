//! **Support constraints** — the domain a random variable lives in, as structured
//! metadata, plus the canonical bijection to/from **unconstrained ℝ**.
//!
//! This is the bridge an inference layer (built *on top of* `bayes`) needs: to work
//! in unconstrained space it maps a constrained value to ℝ, operates there, and
//! maps back, correcting the density by the transform's log-Jacobian. The maps here
//! are the standard ones (exp for positive, logit/sigmoid for bounded). All
//! closed-form — in scope per the dividing principle.
//!
//! Support membership and smooth reparameterization are not identical. Closed
//! boundaries such as `0` for [`Constraint::NonNegative`] are valid support points,
//! but the smooth unconstraining map uses the positive interior and reaches the
//! boundary only at an infinite unconstrained coordinate.

use euclid::{LazyBool, LazyReal};

/// The domain of a scalar random variable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Constraint {
    /// All of `ℝ`.
    Real,
    /// `(0, ∞)`.
    Positive,
    /// `[0, ∞)`. The smooth unconstraining map uses the positive interior.
    NonNegative,
    /// `(0, 1)`.
    UnitInterval,
    /// `(lo, hi)`.
    Interval { lo: f64, hi: f64 },
    /// The integers `ℤ` (discrete — no continuous reparameterization).
    Integer,
}

impl Constraint {
    /// Whether `x` lies in the constrained domain.
    pub fn contains(&self, x: f64) -> bool {
        match *self {
            Constraint::Real => x.is_finite(),
            Constraint::Positive => x > 0.0,
            Constraint::NonNegative => x >= 0.0,
            Constraint::UnitInterval => x > 0.0 && x < 1.0,
            Constraint::Interval { lo, hi } => x > lo && x < hi,
            Constraint::Integer => x.is_finite() && x.fract() == 0.0,
        }
    }

    /// Whether this constraint has a smooth continuous map to and from
    /// unconstrained real coordinates.
    pub fn has_continuous_transform(&self) -> bool {
        !matches!(self, Constraint::Integer)
    }

    /// Whether `x` is a finite, interior point where the smooth unconstraining map
    /// is defined. For closed supports, this is narrower than [`contains`](Self::contains).
    pub fn contains_interior(&self, x: f64) -> bool {
        match *self {
            Constraint::Real => x.is_finite(),
            Constraint::Positive | Constraint::NonNegative => x > 0.0,
            Constraint::UnitInterval => x > 0.0 && x < 1.0,
            Constraint::Interval { lo, hi } => x > lo && x < hi,
            Constraint::Integer => false,
        }
    }

    /// Map a constrained value to **unconstrained ℝ** (the inverse of
    /// [`to_constrained`](Self::to_constrained)) for constraints with a continuous
    /// transform. Boundary support points map to infinities; discrete constraints
    /// return `NaN`.
    pub fn to_unconstrained<T: LazyReal>(&self, x: T) -> T {
        match *self {
            Constraint::Real => x,
            Constraint::Positive | Constraint::NonNegative => x.ln(),
            Constraint::UnitInterval => x.logit(),
            Constraint::Interval { lo, hi } => ((x - T::scalar(lo)) / T::scalar(hi - lo)).logit(),
            Constraint::Integer => T::nan(),
        }
    }

    /// Map an unconstrained `y ∈ ℝ` into the constrained domain for constraints
    /// with a continuous transform. Discrete constraints return `NaN`.
    pub fn to_constrained<T: LazyReal>(&self, y: T) -> T {
        match *self {
            Constraint::Real => y,
            Constraint::Positive | Constraint::NonNegative => y.exp(),
            Constraint::UnitInterval => y.sigmoid(),
            Constraint::Interval { lo, hi } => T::scalar(lo) + T::scalar(hi - lo) * y.sigmoid(),
            Constraint::Integer => T::nan(),
        }
    }

    /// `ln |d to_constrained(y) / dy|` — the Jacobian correction for transforming a
    /// density between constrained and unconstrained space. Discrete constraints
    /// return `NaN`.
    pub fn log_abs_det_jacobian<T: LazyReal>(&self, y: T) -> T {
        match *self {
            Constraint::Real => T::zero(),
            Constraint::Positive | Constraint::NonNegative => y,
            Constraint::UnitInterval => y.ln_sigmoid() + y.ln_1m_sigmoid(),
            Constraint::Interval { lo, hi } => {
                T::scalar((hi - lo).ln()) + y.ln_sigmoid() + y.ln_1m_sigmoid()
            }
            Constraint::Integer => T::nan(),
        }
    }

    /// `ln |d to_unconstrained(x) / dx|`, evaluated in constrained coordinates.
    ///
    /// This is the inverse change-of-variables correction for the smooth
    /// unconstraining map. It is the negative of
    /// [`log_abs_det_jacobian`](Self::log_abs_det_jacobian) at
    /// `to_unconstrained(x)` for interior support points. Boundary points on
    /// closed supports return `+∞`; points outside the support return `NaN`.
    pub fn log_abs_det_inverse_jacobian<T: LazyReal>(&self, x: T) -> T {
        match *self {
            Constraint::Real => T::select(&x.is_finite(), T::zero(), T::nan()),
            Constraint::Positive => {
                let valid = x.gt(&T::zero());
                let safe = T::select(&valid, x, T::one());
                T::select(&valid, -safe.ln(), T::nan())
            }
            Constraint::NonNegative => {
                let positive = x.gt(&T::zero());
                let at_zero = x.eq(&T::zero());
                let safe = T::select(&positive, x, T::one());
                T::select(
                    &positive,
                    -safe.ln(),
                    T::select(&at_zero, T::infinity(), T::nan()),
                )
            }
            Constraint::UnitInterval => {
                let interior = x.gt(&T::zero()).and(&x.lt(&T::one()));
                let boundary = x.eq(&T::zero()).or(&x.eq(&T::one()));
                let safe = T::select(&interior, x, T::half());
                let body = -safe.clone().ln() - (-safe).ln_1p();
                T::select(
                    &interior,
                    body,
                    T::select(&boundary, T::infinity(), T::nan()),
                )
            }
            Constraint::Interval { lo, hi } => {
                let lo_t = T::scalar(lo);
                let hi_t = T::scalar(hi);
                let interior = x.gt(&lo_t).and(&x.lt(&hi_t));
                let boundary = x.eq(&lo_t).or(&x.eq(&hi_t));
                let safe = T::select(&interior, x, T::scalar(0.5 * (lo + hi)));
                let body =
                    T::scalar((hi - lo).ln()) - (safe.clone() - lo_t).ln() - (hi_t - safe).ln();
                T::select(
                    &interior,
                    body,
                    T::select(&boundary, T::infinity(), T::nan()),
                )
            }
            Constraint::Integer => T::nan(),
        }
    }
}

/// A distribution that declares its support as a [`Constraint`] — the metadata an
/// inference layer reads to unconstrain the variable.
pub trait Constrained {
    fn support_constraint(&self) -> Constraint;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn membership() {
        assert!(Constraint::Positive.contains(2.0));
        assert!(!Constraint::Positive.contains(0.0));
        assert!(Constraint::NonNegative.contains(0.0));
        assert!(Constraint::UnitInterval.contains(0.5));
        assert!(!Constraint::UnitInterval.contains(1.0));
        assert!(Constraint::Integer.contains(3.0));
        assert!(!Constraint::Integer.contains(3.5));
        assert!(Constraint::Interval { lo: -1.0, hi: 1.0 }.contains(0.0));
    }

    #[test]
    fn continuous_transform_metadata_distinguishes_boundaries_and_discrete_supports() {
        assert!(Constraint::NonNegative.has_continuous_transform());
        assert!(Constraint::NonNegative.contains(0.0));
        assert!(!Constraint::NonNegative.contains_interior(0.0));
        assert!(Constraint::NonNegative.contains_interior(1.0));
        assert!(!Constraint::Integer.has_continuous_transform());
        assert!(!Constraint::Integer.contains_interior(3.0));
    }

    #[test]
    fn transforms_round_trip_into_the_domain() {
        let cases = [
            Constraint::Real,
            Constraint::Positive,
            Constraint::NonNegative,
            Constraint::UnitInterval,
            Constraint::Interval { lo: -2.0, hi: 5.0 },
        ];
        for c in cases {
            for &y in &[-3.0, -0.5, 0.0, 1.2, 4.0] {
                let x = c.to_constrained(y);
                assert!(
                    c.contains(x),
                    "{c:?}: to_constrained({y})={x} not in domain"
                );
                close(c.to_unconstrained(x), y, 1e-9);
            }
        }
    }

    #[test]
    fn nonnegative_boundary_maps_to_infinite_unconstrained_coordinate() {
        assert_eq!(
            Constraint::NonNegative.to_unconstrained(0.0),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn integer_constraints_do_not_claim_a_continuous_transform() {
        assert!(Constraint::Integer.to_unconstrained(3.0_f64).is_nan());
        assert!(Constraint::Integer.to_constrained(3.0_f64).is_nan());
        assert!(Constraint::Integer.log_abs_det_jacobian(3.0_f64).is_nan());
        assert!(
            Constraint::Integer
                .log_abs_det_inverse_jacobian(3.0_f64)
                .is_nan()
        );
    }

    #[test]
    fn jacobian_matches_a_finite_difference() {
        let c = Constraint::UnitInterval;
        let y = 0.7;
        let h = 1e-6;
        let numeric = ((c.to_constrained(y + h) - c.to_constrained(y - h)) / (2.0 * h)).ln();
        close(c.log_abs_det_jacobian(y), numeric, 1e-5);
    }

    #[test]
    fn logistic_jacobian_is_stable_in_the_tails() {
        let unit = Constraint::UnitInterval;
        close(unit.log_abs_det_jacobian(1000.0), -1000.0, 1e-12);
        close(unit.log_abs_det_jacobian(-1000.0), -1000.0, 1e-12);

        let interval = Constraint::Interval { lo: -2.0, hi: 3.0 };
        close(
            interval.log_abs_det_jacobian(1000.0),
            5.0_f64.ln() - 1000.0,
            1e-12,
        );
    }

    #[test]
    fn inverse_jacobian_cancels_forward_jacobian() {
        let cases = [
            Constraint::Real,
            Constraint::Positive,
            Constraint::NonNegative,
            Constraint::UnitInterval,
            Constraint::Interval { lo: -2.0, hi: 5.0 },
        ];
        for c in cases {
            for &y in &[-3.0, -0.5, 0.0, 1.2, 4.0] {
                let x = c.to_constrained(y);
                close(
                    c.log_abs_det_jacobian(y) + c.log_abs_det_inverse_jacobian(x),
                    0.0,
                    1e-9,
                );
            }
        }
        assert_eq!(
            Constraint::NonNegative.log_abs_det_inverse_jacobian(0.0),
            f64::INFINITY
        );
        assert!(
            Constraint::UnitInterval
                .log_abs_det_inverse_jacobian(2.0_f64)
                .is_nan()
        );
    }
}
