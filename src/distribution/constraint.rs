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

fn sigmoid(y: f64) -> f64 {
    1.0 / (1.0 + (-y).exp())
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
    pub fn to_unconstrained(&self, x: f64) -> f64 {
        match *self {
            Constraint::Real => x,
            Constraint::Positive | Constraint::NonNegative => x.ln(),
            Constraint::UnitInterval => (x / (1.0 - x)).ln(),
            Constraint::Interval { lo, hi } => {
                let u = (x - lo) / (hi - lo);
                (u / (1.0 - u)).ln()
            }
            Constraint::Integer => f64::NAN,
        }
    }

    /// Map an unconstrained `y ∈ ℝ` into the constrained domain for constraints
    /// with a continuous transform. Discrete constraints return `NaN`.
    pub fn to_constrained(&self, y: f64) -> f64 {
        match *self {
            Constraint::Real => y,
            Constraint::Positive | Constraint::NonNegative => y.exp(),
            Constraint::UnitInterval => sigmoid(y),
            Constraint::Interval { lo, hi } => lo + (hi - lo) * sigmoid(y),
            Constraint::Integer => f64::NAN,
        }
    }

    /// `ln |d to_constrained(y) / dy|` — the Jacobian correction for transforming a
    /// density between constrained and unconstrained space. Discrete constraints
    /// return `NaN`.
    pub fn log_abs_det_jacobian(&self, y: f64) -> f64 {
        match *self {
            Constraint::Real => 0.0,
            // d exp(y)/dy = exp(y) ⇒ ln = y.
            Constraint::Positive | Constraint::NonNegative => y,
            // d σ(y)/dy = σ(1-σ).
            Constraint::UnitInterval => {
                let s = sigmoid(y);
                (s * (1.0 - s)).ln()
            }
            Constraint::Interval { lo, hi } => {
                let s = sigmoid(y);
                (hi - lo).ln() + (s * (1.0 - s)).ln()
            }
            Constraint::Integer => f64::NAN,
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
                close(c.to_unconstrained(x), y, 1e-9); // round trips
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
        assert!(Constraint::Integer.to_unconstrained(3.0).is_nan());
        assert!(Constraint::Integer.to_constrained(3.0).is_nan());
        assert!(Constraint::Integer.log_abs_det_jacobian(3.0).is_nan());
    }

    #[test]
    fn jacobian_matches_a_finite_difference() {
        let c = Constraint::UnitInterval;
        let y = 0.7;
        let h = 1e-6;
        let numeric = ((c.to_constrained(y + h) - c.to_constrained(y - h)) / (2.0 * h)).ln();
        close(c.log_abs_det_jacobian(y), numeric, 1e-5);
    }
}
