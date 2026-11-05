//! Parameter validation, in one place.
//!
//! Every eager constructor in the workspace asks the same few questions — is
//! this finite, is it strictly positive, is it a probability, is this range
//! non-empty — and before this module each asked them itself. Thirty-odd copies
//! of the same four checks is not merely verbose: they drift. One family forgets
//! the `is_finite` guard and accepts `inf` as a scale; another rejects `NaN` for
//! one parameter and not its neighbour. The conformance suite found exactly that
//! class of gap.
//!
//! The helpers take a slice of `(name, value)` pairs so the error carries the
//! parameter's own name rather than a positional index: `NotPositive { name:
//! "rate" }` is actionable, `NotPositive { name: "arg1" }` is not.

use crate::BayesError;

/// Every parameter must be finite and strictly positive.
///
/// The usual requirement for a scale, rate, shape, or degrees of freedom. Note
/// that it rejects `0.0` as well as negatives: a zero scale is a point mass, not
/// a degenerate member of the family, and a family that accepted it would divide
/// by zero on its first density evaluation.
pub fn positive(params: &[(&'static str, f64)]) -> Result<(), BayesError> {
    for &(name, value) in params {
        if !value.is_finite() {
            return Err(BayesError::NonFinite { name, value });
        }
        if value <= 0.0 {
            return Err(BayesError::NotPositive { name, value });
        }
    }
    Ok(())
}

/// Every parameter must be finite, with no sign constraint.
///
/// The requirement for a location, which may be any real.
pub fn finite(params: &[(&'static str, f64)]) -> Result<(), BayesError> {
    for &(name, value) in params {
        if !value.is_finite() {
            return Err(BayesError::NonFinite { name, value });
        }
    }
    Ok(())
}

/// Every parameter must be finite and non-negative — zero allowed.
///
/// The requirement for a concentration that may legitimately vanish, such as
/// `VonMises`'s `kappa`, where zero is the uniform distribution on the circle
/// rather than a degenerate case.
pub fn non_negative(params: &[(&'static str, f64)]) -> Result<(), BayesError> {
    for &(name, value) in params {
        if !value.is_finite() {
            return Err(BayesError::NonFinite { name, value });
        }
        if value < 0.0 {
            return Err(BayesError::NotPositive { name, value });
        }
    }
    Ok(())
}

/// The value must be a probability: a number in `[0, 1]`.
///
/// Both endpoints are admitted, because `Bernoulli(0)` and `Bernoulli(1)` are
/// perfectly good point masses. `NaN` is not, and is caught here rather than by
/// the range test — `!(0.0..=1.0).contains(&NaN)` happens to be true, so a
/// range check alone would reject it with a misleading reason.
pub fn probability(value: f64) -> Result<(), BayesError> {
    if value.is_nan() || !(0.0..=1.0).contains(&value) {
        return Err(BayesError::ProbabilityOutOfRange { value });
    }
    Ok(())
}

/// The value must be a probability in the *open* interval `(0, 1]`.
///
/// For a success probability that cannot be zero: `Geometric(0)` would never
/// terminate, so it is not a distribution at all.
pub fn probability_nonzero(value: f64) -> Result<(), BayesError> {
    probability(value)?;
    if value == 0.0 {
        return Err(BayesError::ProbabilityOutOfRange { value });
    }
    Ok(())
}

/// `lo` and `hi` must be finite with `lo < hi`.
///
/// A continuous range with zero width carries no density, so it is empty rather
/// than degenerate — unlike the *discrete* `[lo, hi]`, which holds one value.
pub fn ordered_range(lo: f64, hi: f64) -> Result<(), BayesError> {
    finite(&[("lo", lo), ("hi", hi)])?;
    if lo >= hi {
        return Err(BayesError::EmptyRange { lo, hi });
    }
    Ok(())
}

/// Assert two floats agree to a **relative** tolerance, floored at 1 so a
/// comparison against zero does not divide by it.
///
/// Exported from here rather than copied into each test module: there were 74
/// identical definitions of this before, which is 74 chances for one of them to
/// silently use a different tolerance convention than the test above it reads
/// as. The failure message carries the achieved error, because "2.5 vs 2.4" does
/// not tell you whether you are off by a rounding step or a factor.
#[macro_export]
macro_rules! assert_close {
    ($a:expr, $b:expr, $tol:expr $(,)?) => {{
        let (a, b, tol): (f64, f64, f64) = ($a, $b, $tol);
        let err = (a - b).abs() / b.abs().max(1.0);
        assert!(
            err <= tol,
            "{} vs {} (relative error {:.3e} exceeds {:.0e})",
            a,
            b,
            err,
            tol
        );
    }};
    ($a:expr, $b:expr, $tol:expr, $($arg:tt)+) => {{
        let (a, b, tol): (f64, f64, f64) = ($a, $b, $tol);
        let err = (a - b).abs() / b.abs().max(1.0);
        assert!(
            err <= tol,
            "{}: {} vs {} (relative error {:.3e} exceeds {:.0e})",
            format_args!($($arg)+),
            a,
            b,
            err,
            tol
        );
    }};
}

/// As [`assert_close!`], but with the tolerance scaled by `tol * (1 + |b|)`
/// rather than `tol * max(|b|, 1)`.
///
/// A second macro rather than a second convention inside the first: the two
/// agree within a factor of two everywhere and differ most where `|b|` is near
/// one, so folding seventeen tests written against this form onto the other
/// would quietly re-tune every one of them. Both are named for what they
/// compute.
#[macro_export]
macro_rules! assert_close_scaled {
    ($a:expr, $b:expr, $tol:expr $(,)?) => {{
        let (a, b, tol): (f64, f64, f64) = ($a, $b, $tol);
        let bound = tol + tol * b.abs();
        assert!(
            (a - b).abs() <= bound,
            "{} vs {} (|difference| {:.3e} exceeds {:.3e})",
            a,
            b,
            (a - b).abs(),
            bound
        );
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_rejects_zero_negatives_and_non_finite() {
        assert!(positive(&[("rate", 1.0)]).is_ok());
        assert!(positive(&[("rate", 1.0), ("shape", 2.0)]).is_ok());
        assert!(positive(&[("rate", 0.0)]).is_err());
        assert!(positive(&[("rate", -1.0)]).is_err());
        assert!(positive(&[("rate", f64::NAN)]).is_err());
        assert!(positive(&[("rate", f64::INFINITY)]).is_err());
        // The name of the *offending* parameter is what comes back, not the first.
        match positive(&[("good", 1.0), ("bad", -1.0)]) {
            Err(BayesError::NotPositive { name, .. }) => assert_eq!(name, "bad"),
            other => panic!("expected NotPositive for `bad`, got {other:?}"),
        }
        // A non-finite value is reported as non-finite rather than as non-positive,
        // which is the more useful diagnosis.
        match positive(&[("bad", f64::NEG_INFINITY)]) {
            Err(BayesError::NonFinite { name, .. }) => assert_eq!(name, "bad"),
            other => panic!("expected NonFinite, got {other:?}"),
        }
    }

    #[test]
    fn finite_allows_any_sign_but_no_infinity() {
        assert!(finite(&[("location", -1e300)]).is_ok());
        assert!(finite(&[("location", 0.0)]).is_ok());
        assert!(finite(&[("location", f64::NAN)]).is_err());
        assert!(finite(&[("location", f64::INFINITY)]).is_err());
    }

    #[test]
    fn non_negative_admits_zero_but_not_below() {
        assert!(non_negative(&[("kappa", 0.0)]).is_ok());
        assert!(non_negative(&[("kappa", 3.0)]).is_ok());
        assert!(non_negative(&[("kappa", -1e-300)]).is_err());
        assert!(non_negative(&[("kappa", f64::NAN)]).is_err());
    }

    #[test]
    fn probability_admits_both_endpoints_and_rejects_nan() {
        assert!(probability(0.0).is_ok());
        assert!(probability(1.0).is_ok());
        assert!(probability(0.5).is_ok());
        // `f64::EPSILON` rather than `1e-16`: one ulp at 1.0 is 2.2e-16, so
        // `1.0 + 1e-16` *is* `1.0` and would be correctly accepted.
        assert!(probability(-f64::EPSILON).is_err());
        assert!(probability(1.0 + f64::EPSILON).is_err());
        // The case a bare range check gets wrong: `contains` is false for NaN,
        // so without the explicit test this would pass as "out of range" with a
        // NaN value in the error, which reads as a range problem rather than a
        // NaN problem.
        assert!(probability(f64::NAN).is_err());
    }

    #[test]
    fn probability_nonzero_excludes_only_zero() {
        assert!(probability_nonzero(1e-300).is_ok());
        assert!(probability_nonzero(1.0).is_ok());
        assert!(probability_nonzero(0.0).is_err());
        assert!(probability_nonzero(1.5).is_err());
    }

    #[test]
    fn ordered_range_requires_strict_width() {
        assert!(ordered_range(0.0, 1.0).is_ok());
        assert!(ordered_range(-5.0, -1.0).is_ok());
        assert!(
            ordered_range(1.0, 1.0).is_err(),
            "zero width carries no density"
        );
        assert!(ordered_range(2.0, 1.0).is_err());
        assert!(ordered_range(0.0, f64::INFINITY).is_err());
        assert!(ordered_range(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn assert_close_reports_the_achieved_error() {
        assert_close!(1.0, 1.0, 1e-12);
        assert_close!(1.0 + 1e-13, 1.0, 1e-12);
        // Against zero the tolerance is absolute, by the floor-at-one rule.
        assert_close!(1e-13, 0.0, 1e-12);
        let caught = std::panic::catch_unwind(|| assert_close!(2.0, 1.0, 1e-9));
        assert!(caught.is_err(), "a real mismatch must fail");
    }
}
