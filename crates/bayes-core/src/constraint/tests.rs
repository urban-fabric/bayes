use super::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn real_membership_returns_masks() {
    assert!(Constraint::<f64>::Positive.contains(2.0));
    assert!(!Constraint::<f64>::Positive.contains(0.0));
    assert!(Constraint::<f64>::NonNegative.contains(0.0));
    assert!(Constraint::<f64>::UnitInterval.contains(0.5));
    assert!(!Constraint::<f64>::UnitInterval.contains(1.0));
    assert!(Constraint::<f64>::Interval { lo: -1.0, hi: 1.0 }.contains(0.0));
    assert!(Constraint::<f64>::ClosedInterval { lo: -1.0, hi: 1.0 }.contains(1.0));
}

#[test]
fn transforms_round_trip_into_the_domain() {
    let cases: [Constraint<f64>; 5] = [
        Constraint::Real,
        Constraint::Positive,
        Constraint::NonNegative,
        Constraint::UnitInterval,
        Constraint::Interval { lo: -2.0, hi: 5.0 },
    ];
    for c in cases {
        for y in [-3.0, -0.5, 0.0, 1.2, 4.0] {
            let x = c.to_constrained(y);
            assert!(c.contains(x), "{c:?}: to_constrained({y})={x}");
            close(c.to_unconstrained(x), y, 1e-9);
        }
    }
}

#[test]
fn jacobian_inverse_corrections_cancel() {
    let cases: [Constraint<f64>; 5] = [
        Constraint::Positive,
        Constraint::NonNegative,
        Constraint::UnitInterval,
        Constraint::Interval { lo: -2.0, hi: 5.0 },
        Constraint::Bounded {
            lo: -2.0,
            hi: 5.0,
            lower: Boundary::Closed,
            upper: Boundary::Open,
        },
    ];
    for c in cases {
        for y in [-2.0, -0.2, 0.0, 1.3, 3.5] {
            let x = c.to_constrained(y);
            close(
                c.log_abs_det_jacobian(y) + c.log_abs_det_inverse_jacobian(x),
                0.0,
                1e-9,
            );
        }
    }
}

#[test]
fn discrete_membership_returns_masks() {
    let count = NaturalConstraint::Interval { lo: 2_u64, hi: 5 };
    assert!(!count.contains(1));
    assert!(count.contains(2));
    assert!(count.contains(5));

    let signed = IntegerConstraint::Interval { lo: -3_i64, hi: 4 };
    assert!(!signed.contains(-4));
    assert!(signed.contains(0));
    assert!(signed.contains(4));
}
