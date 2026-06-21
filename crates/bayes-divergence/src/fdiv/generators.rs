use super::*;

/// `f(t) = t ln t` — the **Kullback–Leibler** generator. `D_f = Σ pᵢ ln(pᵢ/qᵢ)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Kl;

impl FGenerator for Kl {
    fn f(&self, t: f64) -> f64 {
        if t > 0.0 { t * t.ln() } else { 0.0 }
    }
}

/// `f(t) = −ln t` — the **reverse-KL** generator. `D_f(p ‖ q) = KL(q ‖ p)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReverseKl;

impl FGenerator for ReverseKl {
    fn f(&self, t: f64) -> f64 {
        if t > 0.0 { -t.ln() } else { f64::INFINITY }
    }
    fn f_infinity(&self) -> f64 {
        0.0
    }
}

/// `f(t) = (t − 1)²` — the **Pearson χ²** generator. `D_f = Σ (pᵢ−qᵢ)²/qᵢ`, an
/// upper bound on KL (`klDiv_le_chiSqDiv`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChiSquared;

impl FGenerator for ChiSquared {
    fn f(&self, t: f64) -> f64 {
        (t - 1.0).powi(2)
    }
}

/// `f(t) = ½(√t − 1)²` — the **squared Hellinger** generator.
/// `D_f = ½ Σ (√pᵢ − √qᵢ)² = 1 − Σ √(pᵢqᵢ) ∈ [0, 1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SquaredHellinger;

impl FGenerator for SquaredHellinger {
    fn f(&self, t: f64) -> f64 {
        0.5 * (t.sqrt() - 1.0).powi(2)
    }
    fn f_infinity(&self) -> f64 {
        0.5
    }
}

/// `f(t) = ½|t − 1|` — the **total-variation** generator.
/// `D_f = ½ Σ |pᵢ − qᵢ| ∈ [0, 1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TotalVariation;

impl FGenerator for TotalVariation {
    fn f(&self, t: f64) -> f64 {
        0.5 * (t - 1.0).abs()
    }
    fn f_infinity(&self) -> f64 {
        0.5
    }
}

/// `f(t) = (t − 1) ln t` — the **Jeffreys** (symmetrized-KL) generator.
/// `D_f = KL(p‖q) + KL(q‖p)`, symmetric in `p` and `q`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Jeffreys;

impl FGenerator for Jeffreys {
    fn f(&self, t: f64) -> f64 {
        if t > 0.0 {
            (t - 1.0) * t.ln()
        } else {
            f64::INFINITY
        }
    }
}

/// The **α-divergence** generator (Amari / Tsallis convention),
/// `f_α(t) = (tᵅ − α(t−1) − 1) / (α(α−1))` for `α ∉ {0, 1}` — a one-parameter
/// family of convex generators that sweeps across the others:
///
/// - `α → 1` is KL ([`Kl`]; returned exactly at `α = 1`),
/// - `α → 0` is reverse KL ([`ReverseKl`]; returned exactly at `α = 0`),
/// - `α = 2` is `½ χ²` (half [`ChiSquared`]),
/// - `α = ½` is `4 ×` [`SquaredHellinger`].
///
/// Distinct from the parent module's [`RenyiDivergence`](super::RenyiDivergence),
/// which is a (monotone) *transform* of the order-`α` divergence rather than an
/// f-divergence itself. Convex with `f_α(1) = 0` for every `α`, so it inherits
/// all three `fDiv_*` guarantees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Alpha(pub f64);

impl FGenerator for Alpha {
    fn f(&self, t: f64) -> f64 {
        let a = self.0;
        if (a - 1.0).abs() < 1e-12 {
            return Kl.f(t);
        }
        if a.abs() < 1e-12 {
            return ReverseKl.f(t);
        }
        // `t.powf(a)` already gives the correct `t = 0` limit for `a > 0` (`0`),
        // making `f(0) = 1/a`; for `a < 0` it diverges, matching `f(0⁺) = +∞`.
        (t.powf(a) - a * (t - 1.0) - 1.0) / (a * (a - 1.0))
    }
    fn f_infinity(&self) -> f64 {
        let a = self.0;
        if a > 1.0 {
            f64::INFINITY
        } else if a.abs() < 1e-12 {
            0.0
        } else if a > 0.0 {
            // 0 < α < 1: lim_{t→∞} f_α(t)/t = 1/(1−α).
            1.0 / (1.0 - a)
        } else {
            f64::INFINITY
        }
    }
}

/// A **custom** generator wrapping any convex `f` with `f(1) = 0`. The escape
/// hatch for f-divergences not named above — the guarantees still hold provided
/// the contract (convex, `f(1) = 0`) is met.
///
/// ```
/// use bayes_divergence::{f_divergence, Custom};
/// // The Pearson χ² generator, supplied as a closure.
/// let chi2 = Custom::new(|t: f64| (t - 1.0).powi(2));
/// let d = f_divergence(&[0.5, 0.5], &[0.25, 0.75], &chi2);
/// assert!(d >= 0.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Custom<F> {
    f: F,
    f_infinity: f64,
}

impl<F: Fn(f64) -> f64> Custom<F> {
    /// A custom generator with the default recession constant `+∞` (the second
    /// distribution must dominate the first — as for KL and χ²).
    pub fn new(f: F) -> Self {
        Self {
            f,
            f_infinity: f64::INFINITY,
        }
    }

    /// A custom generator with an explicit recession constant
    /// `f′(∞) = lim_{t→∞} f(t)/t`, for generators finite under support mismatch.
    pub fn with_recession(f: F, f_infinity: f64) -> Self {
        Self { f, f_infinity }
    }
}

impl<F: Fn(f64) -> f64> FGenerator for Custom<F> {
    fn f(&self, t: f64) -> f64 {
        (self.f)(t)
    }
    fn f_infinity(&self) -> f64 {
        self.f_infinity
    }
}
