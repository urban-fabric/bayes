//! **Generalized entropies** — Rényi `H_α` and Tsallis `S_q`.
//!
//! Both are one-parameter deformations of the Shannon entropy
//! ([`Moments::entropy`](crate::Moments::entropy)), built from the same core,
//! the *information potential* `∫ f(x)^α dx` (a sum `Σ p_i^α` in the discrete
//! case):
//!
//! - Rényi:  `H_α = (1/(1−α)) · ln ∫ f^α`
//! - Tsallis: `S_q = (1/(q−1)) · (1 − ∫ f^q)`
//!
//! Both converge to the Shannon entropy as the order → 1. Each family supplies
//! one method, [`ln_power_integral`](GeneralizedEntropy::ln_power_integral) =
//! `ln ∫ f^α`, and the two entropies follow as provided methods. Orders must be
//! positive and `≠ 1` (the formulas have a removable singularity there).

use crate::{Bernoulli, Exponential, Geometric, Laplace, Normal, Uniform};

/// A family with a closed-form information potential `∫ f^α`.
pub trait GeneralizedEntropy {
    /// `ln ∫ f(x)^α dx` (continuous) or `ln Σ p_i^α` (discrete) — the shared core
    /// of the Rényi and Tsallis entropies.
    fn ln_power_integral(&self, alpha: f64) -> f64;

    /// Rényi entropy of order `α` (`α > 0`, `α ≠ 1`): `ln ∫ f^α / (1 − α)`.
    fn renyi_entropy(&self, alpha: f64) -> f64 {
        self.ln_power_integral(alpha) / (1.0 - alpha)
    }

    /// Tsallis entropy of order `q` (`q ≠ 1`): `(1 − ∫ f^q) / (q − 1)`.
    fn tsallis_entropy(&self, q: f64) -> f64 {
        (1.0 - self.ln_power_integral(q).exp()) / (q - 1.0)
    }
}

/// `ln ∫ f^α = −½ ln α + ½(1−α) ln(2πσ²)`.
impl GeneralizedEntropy for Normal {
    fn ln_power_integral(&self, alpha: f64) -> f64 {
        let sigma = self.params().1;
        let two_pi_var = std::f64::consts::TAU * sigma * sigma;
        -0.5 * alpha.ln() + 0.5 * (1.0 - alpha) * two_pi_var.ln()
    }
}

/// `ln ∫ f^α = (α−1) ln λ − ln α` (rate `λ`).
impl GeneralizedEntropy for Exponential {
    fn ln_power_integral(&self, alpha: f64) -> f64 {
        (alpha - 1.0) * self.rate().ln() - alpha.ln()
    }
}

/// `ln ∫ f^α = (1−α) ln(b−a)` — Rényi entropy is `ln(b−a)` for every order.
impl GeneralizedEntropy for Uniform {
    fn ln_power_integral(&self, alpha: f64) -> f64 {
        (1.0 - alpha) * (self.hi() - self.lo()).ln()
    }
}

/// `ln ∫ f^α = (1−α) ln(2b) − ln α` (scale `b`).
impl GeneralizedEntropy for Laplace {
    fn ln_power_integral(&self, alpha: f64) -> f64 {
        (1.0 - alpha) * (2.0 * self.scale()).ln() - alpha.ln()
    }
}

/// `ln Σ p_i^α = ln(p^α + (1−p)^α)`.
impl GeneralizedEntropy for Bernoulli {
    fn ln_power_integral(&self, alpha: f64) -> f64 {
        let p = self.p();
        (p.powf(alpha) + (1.0 - p).powf(alpha)).ln()
    }
}

/// `ln Σ p_i^α = α ln p − ln(1 − (1−p)^α)` (trial-count parameterization).
impl GeneralizedEntropy for Geometric {
    fn ln_power_integral(&self, alpha: f64) -> f64 {
        let p = self.p();
        alpha * p.ln() - (1.0 - (1.0 - p).powf(alpha)).ln()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::Moments;
    use bayes_core_ext::assert_close;

    #[test]
    fn renyi_limits_to_shannon_entropy_as_order_approaches_one() {
        // H_α → H (the differential/Shannon entropy) as α → 1.
        let cases: [(&dyn GeneralizedEntropy, f64); 4] = [
            (
                &Normal::new(0.0, 2.0).unwrap(),
                Normal::new(0.0, 2.0).unwrap().entropy(),
            ),
            (
                &Exponential::new(1.5).unwrap(),
                Exponential::new(1.5).unwrap().entropy(),
            ),
            (
                &Laplace::new(0.0, 1.3).unwrap(),
                Laplace::new(0.0, 1.3).unwrap().entropy(),
            ),
            (
                &Bernoulli::new(0.3).unwrap(),
                Bernoulli::new(0.3).unwrap().entropy(),
            ),
        ];
        for (d, shannon) in cases {
            assert_close!(d.renyi_entropy(1.0 + 1e-6), shannon, 1e-4);
            assert_close!(d.tsallis_entropy(1.0 + 1e-6), shannon, 1e-4);
        }
    }

    #[test]
    fn renyi_closed_forms_match_known_values() {
        // Uniform Rényi entropy is ln(width) for every order.
        let u = Uniform::new(2.0, 6.0).unwrap();
        assert_close!(u.renyi_entropy(0.5), 4.0_f64.ln(), 1e-12);
        assert_close!(u.renyi_entropy(3.0), 4.0_f64.ln(), 1e-12);

        // Exponential(λ): H_α = −ln λ + ln α/(α−1).
        let e = Exponential::new(2.0).unwrap();
        let alpha = 2.0;
        assert_close!(
            e.renyi_entropy(alpha),
            -2.0_f64.ln() + alpha.ln() / (alpha - 1.0),
            1e-12,
        );

        // Normal collision entropy (α=2): ½ln(2πσ²) + ln2/2.
        let n = Normal::new(0.0, 1.0).unwrap();
        assert_close!(
            n.renyi_entropy(2.0),
            0.5 * std::f64::consts::TAU.ln() + 0.5 * 2.0_f64.ln(),
            1e-12,
        );

        // Bernoulli(½) collision entropy: ln Σ p² = ln(0.5) ⇒ H_2 = ln 2.
        assert_close!(
            Bernoulli::new(0.5).unwrap().renyi_entropy(2.0),
            2.0_f64.ln(),
            1e-12,
        );
    }

    #[test]
    fn renyi_is_nonincreasing_in_order() {
        // H_α is monotonically non-increasing in α for a fixed law.
        let n = Normal::new(1.0, 2.0).unwrap();
        let (h_half, h_two, h_five) = (
            n.renyi_entropy(0.5),
            n.renyi_entropy(2.0),
            n.renyi_entropy(5.0),
        );
        assert!(
            h_half >= h_two && h_two >= h_five,
            "{h_half} {h_two} {h_five}"
        );
    }
}
