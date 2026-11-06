//! **Moment- and cumulant-generating functions.**
//!
//! The MGF `M(t) = E[e^{tX}]` and its log, the CGF `K(t) = ln M(t)`, are the
//! closed-form transforms behind the exponential-tilt tail bounds: the Chernoff
//! method is `P(X ≥ a) ≤ e^{−t a}·M(t)` minimized over `t`, so a distribution's
//! MGF is exactly the input that turns that abstract bound into a concrete tail
//! for a named family (feed `mgf(t)` to `bayes_statistics::chernoff_bound`).
//!
//! ```
//! use bayes_distribution_ext::{MomentGenerating, Normal};
//! // Chernoff tail for the standard normal at a = 2, tilted at t = a/σ² = 2:
//! // e^{−t·a}·M(t) = e^{−4}·e^{2} = e^{−2}.
//! let z = Normal::new(0.0, 1.0).unwrap();
//! let bound = (-2.0_f64 * 2.0).exp() * z.mgf(2.0);
//! assert!((bound - (-2.0_f64).exp()).abs() < 1e-12); // e^{−a²/2σ²}
//! ```
//!
//! Several MGFs exist only on one side of the origin (the expectation diverges
//! beyond a threshold); there the methods return `+∞`, which makes the Chernoff
//! bound vacuously `1` rather than silently wrong. Families with no elementary
//! MGF (Cauchy, Student-t, Beta, log-normal, Pareto, …) are intentionally absent.

use crate::{Bernoulli, Binomial, ChiSquared, Exponential, Gamma, Geometric, Normal, Poisson};

/// A family with a closed-form moment-generating function.
pub trait MomentGenerating {
    /// The moment-generating function `M(t) = E[e^{tX}]`. Returns `+∞` for `t`
    /// outside the region where the expectation is finite.
    fn mgf(&self, t: f64) -> f64;

    /// The cumulant-generating function `K(t) = ln M(t)`. Returns `+∞` where the
    /// MGF diverges. Some families override this with a numerically stabler form.
    fn cgf(&self, t: f64) -> f64 {
        self.mgf(t).ln()
    }
}

/// `M(t) = exp(μt + ½σ²t²)`.
impl MomentGenerating for Normal {
    fn mgf(&self, t: f64) -> f64 {
        self.cgf(t).exp()
    }
    fn cgf(&self, t: f64) -> f64 {
        let (mu, sigma) = self.params();
        mu * t + 0.5 * sigma * sigma * t * t
    }
}

/// `M(t) = λ/(λ−t)` for `t < λ`, else `+∞`.
impl MomentGenerating for Exponential {
    fn mgf(&self, t: f64) -> f64 {
        let rate = self.rate();
        if t < rate {
            rate / (rate - t)
        } else {
            f64::INFINITY
        }
    }
}

/// `M(t) = (1 − t/β)^{−k}` for `t < β` (shape `k`, rate `β`), else `+∞`.
impl MomentGenerating for Gamma {
    fn mgf(&self, t: f64) -> f64 {
        let (shape, rate) = (self.shape(), self.rate());
        if t < rate {
            (1.0 - t / rate).powf(-shape)
        } else {
            f64::INFINITY
        }
    }
    fn cgf(&self, t: f64) -> f64 {
        let (shape, rate) = (self.shape(), self.rate());
        if t < rate {
            -shape * (1.0 - t / rate).ln()
        } else {
            f64::INFINITY
        }
    }
}

/// `M(t) = exp(λ(e^t − 1))`.
impl MomentGenerating for Poisson {
    fn mgf(&self, t: f64) -> f64 {
        self.cgf(t).exp()
    }
    fn cgf(&self, t: f64) -> f64 {
        self.lambda() * (t.exp() - 1.0)
    }
}

/// `M(t) = (1 − p) + p·e^t`.
impl MomentGenerating for Bernoulli {
    fn mgf(&self, t: f64) -> f64 {
        let p = self.p();
        (1.0 - p) + p * t.exp()
    }
}

/// `M(t) = ((1 − p) + p·e^t)^n`.
impl MomentGenerating for Binomial {
    fn mgf(&self, t: f64) -> f64 {
        let p = self.p();
        ((1.0 - p) + p * t.exp()).powi(self.trials() as i32)
    }
}

/// `M(t) = p·e^t / (1 − (1−p)·e^t)` for `t < −ln(1−p)` (trial-count
/// parameterization), else `+∞`.
impl MomentGenerating for Geometric {
    fn mgf(&self, t: f64) -> f64 {
        let p = self.p();
        let q_exp = (1.0 - p) * t.exp();
        if q_exp < 1.0 {
            p * t.exp() / (1.0 - q_exp)
        } else {
            f64::INFINITY
        }
    }
}

/// `M(t) = (1 − 2t)^{−k/2}` for `t < ½` (`k` degrees of freedom), else `+∞`.
impl MomentGenerating for ChiSquared {
    fn mgf(&self, t: f64) -> f64 {
        if t < 0.5 {
            (1.0 - 2.0 * t).powf(-0.5 * self.freedom())
        } else {
            f64::INFINITY
        }
    }
    fn cgf(&self, t: f64) -> f64 {
        if t < 0.5 {
            -0.5 * self.freedom() * (1.0 - 2.0 * t).ln()
        } else {
            f64::INFINITY
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;

    #[test]
    fn mgf_is_one_and_cgf_zero_at_origin() {
        let dists: [&dyn MomentGenerating; 8] = [
            &Normal::new(2.0, 3.0).unwrap(),
            &Exponential::new(1.5).unwrap(),
            &Gamma::new(3.0, 2.0).unwrap(),
            &Poisson::new(4.0).unwrap(),
            &Bernoulli::new(0.3).unwrap(),
            &Binomial::new(5, 0.4).unwrap(),
            &Geometric::new(0.5).unwrap(),
            &ChiSquared::new(4.0).unwrap(),
        ];
        for d in dists {
            assert_close!(d.mgf(0.0), 1.0, 1e-12);
            assert_close!(d.cgf(0.0), 0.0, 1e-12);
        }
    }

    #[test]
    fn mgf_closed_forms_match_known_values() {
        let e = std::f64::consts::E;

        // Normal(2,3): M(0.5) = exp(2·0.5 + ½·9·0.25) = exp(2.125).
        let n = Normal::new(2.0, 3.0).unwrap();
        assert_close!(n.mgf(0.5), 2.125_f64.exp(), 1e-12);
        assert_close!(n.cgf(0.5), 2.125, 1e-12);

        // Exponential(2): M(1) = 2/(2−1) = 2; diverges at t ≥ rate.
        let ex = Exponential::new(2.0).unwrap();
        assert_close!(ex.mgf(1.0), 2.0, 1e-12);
        assert_eq!(ex.mgf(2.0), f64::INFINITY);
        assert_eq!(ex.mgf(3.0), f64::INFINITY);

        // Gamma(3,2): M(1) = (2/1)^3 = 8; CGF = 3·ln2.
        let g = Gamma::new(3.0, 2.0).unwrap();
        assert_close!(g.mgf(1.0), 8.0, 1e-12);
        assert_close!(g.cgf(1.0), 3.0 * 2.0_f64.ln(), 1e-12);
        assert_eq!(g.mgf(2.0), f64::INFINITY);

        // Poisson(2): M(1) = exp(2(e−1)).
        let p = Poisson::new(2.0).unwrap();
        assert_close!(p.mgf(1.0), (2.0 * (e - 1.0)).exp(), 1e-12);
        assert_close!(p.cgf(1.0), 2.0 * (e - 1.0), 1e-12);

        // Bernoulli(0.3): M(1) = 0.7 + 0.3e.
        assert_close!(Bernoulli::new(0.3).unwrap().mgf(1.0), 0.7 + 0.3 * e, 1e-12);

        // Binomial(4,0.3): M(1) = (0.7 + 0.3e)^4.
        assert_close!(
            Binomial::new(4, 0.3).unwrap().mgf(1.0),
            (0.7 + 0.3 * e).powi(4),
            1e-12,
        );

        // Geometric(0.5) trials: finite below t = −ln(1−p) = ln 2 ≈ 0.693.
        let geo = Geometric::new(0.5).unwrap();
        let m_half = 0.5 * 0.5_f64.exp() / (1.0 - 0.5 * 0.5_f64.exp());
        assert_close!(geo.mgf(0.5), m_half, 1e-12);
        assert_eq!(geo.mgf(1.0), f64::INFINITY);

        // ChiSquared(4): M(0.25) = (1−0.5)^{−2} = 4; diverges at t ≥ ½.
        let chi = ChiSquared::new(4.0).unwrap();
        assert_close!(chi.mgf(0.25), 4.0, 1e-12);
        assert_eq!(chi.mgf(0.5), f64::INFINITY);
    }

    #[test]
    fn mgf_generates_mean_and_variance_by_finite_difference() {
        // K'(0) = mean, K''(0) = variance. Check against a central difference.
        let g = Gamma::new(3.0, 2.0).unwrap(); // mean 1.5, variance 0.75
        let h = 1e-4;
        let k_mean = (g.cgf(h) - g.cgf(-h)) / (2.0 * h);
        let k_var = (g.cgf(h) - 2.0 * g.cgf(0.0) + g.cgf(-h)) / (h * h);
        assert_close!(k_mean, 1.5, 1e-6);
        assert_close!(k_var, 0.75, 1e-6);
    }
}
