//! Invertible scalar maps for [`Transformed`](super::Transformed) distributions.
//!
//! A [`Bijector`] provides the forward map, its inverse, and the log of the
//! absolute derivative of the *inverse* (the Jacobian term in the change of
//! variables). All maps here are strictly monotone, so the density is well-defined.

/// A strictly monotone, invertible map `g: ℝ → S`.
pub trait Bijector {
    /// `y = g(x)`.
    fn forward(&self, x: f64) -> f64;
    /// `x = g⁻¹(y)`.
    fn inverse(&self, y: f64) -> f64;
    /// `ln |d g⁻¹(y) / dy|` — the Jacobian correction for `p_Y(y)`.
    fn inverse_log_det_jacobian(&self, y: f64) -> f64;
}

/// The affine map `g(x) = scale·x + shift` (`scale ≠ 0`). Shifts and scales a base
/// distribution (e.g. turns a standard normal into any normal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub scale: f64,
    pub shift: f64,
}

impl Affine {
    pub fn new(scale: f64, shift: f64) -> Self {
        Self { scale, shift }
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
        // d g⁻¹/dy = 1/scale ⇒ ln|1/scale| = -ln|scale|.
        -self.scale.abs().ln()
    }
}

/// The exponential map `g(x) = eˣ`, mapping `ℝ → (0, ∞)`. `Transformed(Normal, Exp)`
/// is the log-normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exp;

impl Bijector for Exp {
    fn forward(&self, x: f64) -> f64 {
        x.exp()
    }
    fn inverse(&self, y: f64) -> f64 {
        y.ln()
    }
    fn inverse_log_det_jacobian(&self, y: f64) -> f64 {
        // d ln(y)/dy = 1/y ⇒ ln(1/y) = -ln y.
        -y.ln()
    }
}

/// The logistic-sigmoid map `g(x) = 1/(1+e⁻ˣ)`, mapping `ℝ → (0, 1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sigmoid;

impl Bijector for Sigmoid {
    fn forward(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    fn inverse(&self, y: f64) -> f64 {
        // logit.
        (y / (1.0 - y)).ln()
    }
    fn inverse_log_det_jacobian(&self, y: f64) -> f64 {
        // d logit(y)/dy = 1/(y(1-y)) ⇒ ln = -ln y - ln(1-y).
        -y.ln() - (1.0 - y).ln()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    fn round_trips(b: &impl Bijector, x: f64) {
        close(b.inverse(b.forward(x)), x, 1e-10);
    }

    #[test]
    fn bijectors_round_trip() {
        round_trips(&Affine::new(2.0, -1.0), 3.3);
        round_trips(&Exp, 1.7);
        round_trips(&Sigmoid, -0.8);
    }

    #[test]
    fn sigmoid_maps_into_the_unit_interval() {
        let s = Sigmoid;
        for &x in &[-10.0, 0.0, 10.0] {
            let y = s.forward(x);
            assert!(y > 0.0 && y < 1.0);
        }
        close(s.forward(0.0), 0.5, 1e-12);
    }
}
