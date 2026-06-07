//! **Closed-form Kullback–Leibler divergence** between distributions of the same
//! family. `D_KL(p ‖ q) = ∫ p ln(p/q)` in **nats** — an analytic formula per
//! family (no Monte-Carlo estimate), so it stays on the closed-form side of the
//! library's boundary. Cross-entropy follows as `H(p, q) = H(p) + D_KL(p ‖ q)`.

use crate::distribution::{Bernoulli, Beta, Exponential, Gamma, Normal, Poisson};
use crate::function::beta::ln_beta;
use crate::function::gamma::{digamma, ln_gamma};

/// Closed-form Kullback–Leibler divergence `D_KL(self ‖ other)`, in nats.
pub trait KlDivergence<Rhs = Self> {
    fn kl_divergence(&self, other: &Rhs) -> f64;
}

impl KlDivergence for Normal {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (m1, s1) = self.params();
        let (m2, s2) = other.params();
        (s2 / s1).ln() + (s1 * s1 + (m1 - m2) * (m1 - m2)) / (2.0 * s2 * s2) - 0.5
    }
}

impl KlDivergence for Exponential {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.rate(), other.rate());
        (l1 / l2).ln() + l2 / l1 - 1.0
    }
}

impl KlDivergence for Gamma {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (a1, b1) = (self.shape(), self.rate());
        let (a2, b2) = (other.shape(), other.rate());
        (a1 - a2) * digamma(a1) - ln_gamma(a1)
            + ln_gamma(a2)
            + a2 * (b1.ln() - b2.ln())
            + a1 * (b2 - b1) / b1
    }
}

impl KlDivergence for Beta {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (a1, b1) = (self.alpha(), self.beta());
        let (a2, b2) = (other.alpha(), other.beta());
        ln_beta(a2, b2) - ln_beta(a1, b1)
            + (a1 - a2) * digamma(a1)
            + (b1 - b2) * digamma(b1)
            + (a2 - a1 + b2 - b1) * digamma(a1 + b1)
    }
}

impl KlDivergence for Bernoulli {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (p, q) = (self.p(), other.p());
        let term = |a: f64, b: f64| if a > 0.0 { a * (a / b).ln() } else { 0.0 };
        term(p, q) + term(1.0 - p, 1.0 - q)
    }
}

impl KlDivergence for Poisson {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.lambda(), other.lambda());
        l1 * (l1 / l2).ln() - l1 + l2
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn self_divergence_is_zero() {
        close(
            Normal::new(1.0, 2.0)
                .unwrap()
                .kl_divergence(&Normal::new(1.0, 2.0).unwrap()),
            0.0,
            1e-12,
        );
        close(
            Exponential::new(3.0)
                .unwrap()
                .kl_divergence(&Exponential::new(3.0).unwrap()),
            0.0,
            1e-12,
        );
        close(
            Gamma::new(2.0, 1.5)
                .unwrap()
                .kl_divergence(&Gamma::new(2.0, 1.5).unwrap()),
            0.0,
            1e-10,
        );
        close(
            Beta::new(2.0, 3.0)
                .unwrap()
                .kl_divergence(&Beta::new(2.0, 3.0).unwrap()),
            0.0,
            1e-10,
        );
        close(
            Poisson::new(4.0)
                .unwrap()
                .kl_divergence(&Poisson::new(4.0).unwrap()),
            0.0,
            1e-12,
        );
    }

    #[test]
    fn kl_is_non_negative_and_asymmetric() {
        let p = Normal::new(0.0, 1.0).unwrap();
        let q = Normal::new(2.0, 3.0).unwrap();
        assert!(p.kl_divergence(&q) > 0.0);
        assert!(q.kl_divergence(&p) > 0.0);
        assert_ne!(p.kl_divergence(&q), q.kl_divergence(&p));
    }

    #[test]
    fn normal_kl_known_value() {
        close(
            Normal::new(0.0, 1.0)
                .unwrap()
                .kl_divergence(&Normal::new(1.0, 1.0).unwrap()),
            0.5,
            1e-12,
        );
    }

    #[test]
    fn bernoulli_kl_matches_definition() {
        let kl = Bernoulli::new(0.3)
            .unwrap()
            .kl_divergence(&Bernoulli::new(0.5).unwrap());
        let expect = 0.3 * (0.3f64 / 0.5).ln() + 0.7 * (0.7f64 / 0.5).ln();
        close(kl, expect, 1e-12);
    }
}
