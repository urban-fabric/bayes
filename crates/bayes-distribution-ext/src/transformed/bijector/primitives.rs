//! The maps themselves.
//!
//! Each one states the interval it maps onto, because that is what decides
//! which base distribution it can legitimately be applied to and what the
//! resulting support is.

use super::Bijector;
use bayes_core_ext::BayesError;

/// The identity `g(x) = x`, mapping `ℝ → ℝ`.
///
/// The unit of [`Compose`](super::Compose): composing with it on either side
/// changes nothing, including the Jacobian, which is `ln 1 = 0`. Useful as the
/// starting point of a fold over a list of maps, and as the "no transform"
/// case in generic code that is parameterized over a bijector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Identity;

impl Bijector for Identity {
    fn forward(&self, x: f64) -> f64 {
        x
    }
    fn inverse(&self, y: f64) -> f64 {
        y
    }
    fn inverse_log_det_jacobian(&self, _y: f64) -> f64 {
        0.0
    }
}

/// The affine map `g(x) = scale·x + shift` (`scale ≠ 0`), mapping `ℝ → ℝ`.
///
/// Shifts and scales a base distribution — turns a standard normal into any
/// normal. A negative `scale` reflects, which is why
/// [`inverse_log_det_jacobian`](Bijector::inverse_log_det_jacobian) takes an
/// absolute value: the density of a reflected variable is still positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub scale: f64,
    pub shift: f64,
}

impl Affine {
    pub fn new(scale: f64, shift: f64) -> Result<Self, BayesError> {
        if !scale.is_finite() {
            return Err(BayesError::NonFinite {
                name: "scale",
                value: scale,
            });
        }
        if !shift.is_finite() {
            return Err(BayesError::NonFinite {
                name: "shift",
                value: shift,
            });
        }
        if scale == 0.0 {
            return Err(BayesError::Malformed {
                what: "affine scale must be non-zero",
            });
        }
        Ok(Self { scale, shift })
    }
}

impl Bijector for Affine {
    fn forward(&self, x: f64) -> f64 {
        self.scale * x + self.shift
    }
    fn inverse(&self, y: f64) -> f64 {
        (y - self.shift) / self.scale
    }
    fn inverse_log_det_jacobian(&self, _y: f64) -> f64 {
        -self.scale.abs().ln()
    }
}

/// The exponential map `g(x) = eˣ`, mapping `ℝ → (0, ∞)`.
///
/// `Transformed(Normal, Exp)` is the log-normal.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Exp;

impl Bijector for Exp {
    fn forward(&self, x: f64) -> f64 {
        x.exp()
    }
    fn inverse(&self, y: f64) -> f64 {
        y.ln()
    }
    fn inverse_log_det_jacobian(&self, y: f64) -> f64 {
        -y.ln()
    }
}

/// The logistic-sigmoid map `g(x) = 1/(1+e⁻ˣ)`, mapping `ℝ → (0, 1)`.
///
/// `Transformed(Normal, Sigmoid)` is the logit-normal, a distribution on a
/// probability.
///
/// # Saturation, and why it is lopsided
///
/// The open interval `(0, 1)` is a claim about real numbers, not about `f64`,
/// and the two endpoints are not reached symmetrically:
///
/// - **Upper:** above `x ≈ 36.7` the result is `1` *exactly*, because
///   `1 + e^{-37}` is 1 to within an ulp — there is no `f64` between
///   `1 - 5.6e-17` and `1`.
/// - **Lower:** `0` is not reached until `x ≈ -745`, because `1/(1 + e^{40})`
///   is `4.2e-18`, which is a perfectly ordinary `f64`. Near zero the
///   representable numbers are dense; near one they are not.
///
/// So the forward map does reach both endpoints eventually, and `inverse` sends
/// them back to `±inf`. That is an honest report of a limit rather than a
/// defect, and [`Transformed`](crate::Transformed) handles the endpoints by
/// giving them zero density.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sigmoid;

impl Bijector for Sigmoid {
    fn forward(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    fn inverse(&self, y: f64) -> f64 {
        (y / (1.0 - y)).ln()
    }
    fn inverse_log_det_jacobian(&self, y: f64) -> f64 {
        -y.ln() - (1.0 - y).ln()
    }
}

/// The softplus map `g(x) = ln(1 + eˣ)`, mapping `ℝ → (0, ∞)`.
///
/// The other route onto the positive half-line, and a gentler one than [`Exp`]:
/// softplus is asymptotically the identity for large `x` rather than
/// exponential, so it moves far less probability mass into the tail. For a
/// standard normal base, `Exp` gives a log-normal whose mean is `e^{1/2}` while
/// softplus gives something with a mean near `0.8`.
///
/// Every branch here exists to avoid an overflow that the naive formula walks
/// straight into: `ln(1 + exp(x))` overflows at `x = 710` even though the
/// answer is just `710`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Softplus;

impl Bijector for Softplus {
    fn forward(&self, x: f64) -> f64 {
        if x > 0.0 {
            // ln(1 + eˣ) = x + ln(1 + e⁻ˣ), which cannot overflow.
            x + (-x).exp().ln_1p()
        } else {
            x.exp().ln_1p()
        }
    }

    /// `g⁻¹(y) = ln(e^y − 1)`.
    fn inverse(&self, y: f64) -> f64 {
        if y > std::f64::consts::LN_2 {
            // = y + ln(1 − e⁻ʸ): keeps the subtraction away from 1 for large y,
            // where `exp(y) - 1` would overflow and `ln` of it would be lost.
            y + (-(-y).exp()).ln_1p()
        } else {
            // Small y: `exp_m1` carries the digits `exp(y) - 1` discards. At
            // `y = 1e-305` the subtraction underflows to exactly zero, whose
            // log is `-inf`; `exp_m1` returns `1e-305` and the log is right.
            y.exp_m1().ln()
        }
    }

    /// `d g⁻¹/dy = e^y/(e^y − 1) = 1/(1 − e⁻ʸ)`, so the log-det is
    /// `−ln(1 − e⁻ʸ)`.
    fn inverse_log_det_jacobian(&self, y: f64) -> f64 {
        -(-(-y).exp()).ln_1p()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::super::probe::behaves_like_a_bijector;
    use super::*;

    #[test]
    fn every_primitive_is_a_bijector_with_the_jacobian_it_claims() {
        let reals = [-8.0, -1.5, -0.25, 0.0, 0.25, 1.5, 8.0];
        behaves_like_a_bijector("Identity", &Identity, &reals);
        behaves_like_a_bijector("Affine(2, -1)", &Affine::new(2.0, -1.0).unwrap(), &reals);
        behaves_like_a_bijector("Affine(-0.5, 3)", &Affine::new(-0.5, 3.0).unwrap(), &reals);
        behaves_like_a_bijector("Exp", &Exp, &reals);
        behaves_like_a_bijector("Sigmoid", &Sigmoid, &reals);
        behaves_like_a_bijector("Softplus", &Softplus, &reals);
    }

    #[test]
    fn affine_rejects_noninvertible_or_nonfinite_parameters() {
        assert!(Affine::new(0.0, 0.0).is_err());
        assert!(Affine::new(f64::NAN, 0.0).is_err());
        assert!(Affine::new(1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn the_maps_land_in_the_intervals_they_advertise() {
        // Inside the range where `(0, 1)` and `(0, inf)` are representable.
        for &x in &[-36.0, -1.0, 0.0, 1.0, 36.0] {
            let p = Sigmoid.forward(x);
            assert!(p > 0.0 && p < 1.0, "Sigmoid({x}) = {p} left (0, 1)");
            assert!(Exp.forward(x) > 0.0, "Exp({x}) left (0, inf)");
            assert!(
                Softplus.forward(x) > 0.0,
                "Softplus({x}) = {} left (0, inf)",
                Softplus.forward(x)
            );
        }
        assert!((Sigmoid.forward(0.0) - 0.5).abs() < 1e-12);
        assert!((Softplus.forward(0.0) - std::f64::consts::LN_2).abs() < 1e-12);
    }

    /// A sigmoid saturates, and **lopsidedly**: it hits 1 at `x ≈ 36.7` but
    /// does not hit 0 until `x ≈ -745`, because `f64` is dense near zero and
    /// sparse near one. Asserted rather than avoided — it is the reason
    /// [`crate::Transformed`] has to treat the endpoints of an open support as
    /// zero-density instead of assuming they are unreachable.
    #[test]
    fn sigmoid_saturates_lopsidedly_where_f64_runs_out_of_room() {
        assert!(Sigmoid.forward(36.0) < 1.0);
        assert_eq!(Sigmoid.forward(40.0), 1.0);
        // The lower end is still resolving a value this far out...
        assert!(Sigmoid.forward(-40.0) > 0.0);
        assert!(Sigmoid.forward(-700.0) > 0.0);
        // ...and only reaches zero once `exp(-x)` itself overflows.
        assert_eq!(Sigmoid.forward(-800.0), 0.0);
        // The inverse reports saturation as an infinity rather than inventing a
        // finite preimage.
        assert_eq!(Sigmoid.inverse(1.0), f64::INFINITY);
        assert_eq!(Sigmoid.inverse(0.0), f64::NEG_INFINITY);
    }

    /// The arguments where the naive formulas overflow or cancel. `Softplus` is
    /// asymptotically the identity, so these have known exact answers.
    #[test]
    fn softplus_survives_the_arguments_that_overflow_the_naive_formula() {
        // ln(1 + exp(800)) would be `inf`; the answer is 800 to within an ulp.
        assert!((Softplus.forward(800.0) - 800.0).abs() < 1e-9);
        assert!((Softplus.inverse(800.0) - 800.0).abs() < 1e-9);
        // Far negative: ln(1 + eˣ) ≈ eˣ, which must not round to zero.
        assert!(Softplus.forward(-700.0) > 0.0);
        // And the round trip holds across the whole span.
        for &x in &[-700.0, -100.0, -1e-8, 1e-8, 100.0, 700.0] {
            let back = Softplus.inverse(Softplus.forward(x));
            assert!(
                (back - x).abs() < 1e-6 * x.abs().max(1.0),
                "Softplus round trip at {x} gave {back}"
            );
        }
    }
}
