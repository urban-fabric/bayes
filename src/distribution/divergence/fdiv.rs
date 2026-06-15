//! **The f-divergence umbrella** — one generic divergence over discrete
//! probability vectors, parameterized by a convex generator.
//!
//! Every divergence in the parent module is a special case of the
//! **f-divergence**
//!
//! ```text
//! D_f(p ‖ q) = Σ qᵢ · f(pᵢ / qᵢ)
//! ```
//!
//! for a convex generator `f : (0, ∞) → ℝ` with `f(1) = 0`. Picking `f`
//! recovers each named divergence:
//!
//! | generator | `f(t)` | `D_f(p ‖ q)` |
//! |---|---|---|
//! | [`Kl`] | `t ln t` | `Σ pᵢ ln(pᵢ/qᵢ)` (Kullback–Leibler) |
//! | [`ReverseKl`] | `−ln t` | `Σ qᵢ ln(qᵢ/pᵢ)` = `KL(q ‖ p)` |
//! | [`ChiSquared`] | `(t−1)²` | `Σ (pᵢ−qᵢ)²/qᵢ` (Pearson χ²) |
//! | [`SquaredHellinger`] | `½(√t−1)²` | `½ Σ (√pᵢ−√qᵢ)²` |
//! | [`TotalVariation`] | `½‖t−1‖` | `½ Σ ‖pᵢ−qᵢ‖` |
//! | [`Jeffreys`] | `(t−1) ln t` | `KL(p‖q) + KL(q‖p)` |
//! | [`Alpha`]`(α)` | `(tᵅ − α(t−1) − 1)/(α(α−1))` | the α-divergence family |
//! | [`Custom`] | any convex `f` with `f(1)=0` | user-supplied |
//!
//! ## Why one abstraction covers them all
//!
//! Convexity of `f` plus `f(1) = 0` are *exactly* the hypotheses behind the
//! three universal guarantees every f-divergence enjoys — all machine-checked
//! generically (for an arbitrary convex `f`) in the `riemann` Lean development
//! (`RiemannLean/Bayes/Divergence.lean`):
//!
//! - **Nonnegativity** `D_f(p ‖ q) ≥ 0` — `fDiv_nonneg` (one Jensen step;
//!   `klDiv_nonneg`/Gibbs is the `f = t ln t` instance, `klDiv_eq_fDiv`).
//! - **The data-processing inequality** — `fDiv_dataProcessing`: pushing both
//!   arguments through any stochastic [`Channel`] cannot increase the divergence.
//! - **Joint convexity** in `(p, q)` — `fDiv_convex` (the two-point
//!   `perspective_two` applied termwise).
//!
//! and the comparison `KL ≤ χ²` (`klDiv_le_chiSqDiv`) is the `Kl`-vs-`ChiSquared`
//! instance.
//!
//! ## Conventions
//!
//! Arguments are probability vectors over a shared, index-aligned support; a
//! shorter vector is treated as zero-padded (a category absent from one side has
//! probability `0` there — the same convention the parent module's
//! `js_divergence`/`total_variation` use). The limit terms are handled with the
//! standard f-divergence conventions: `0 · f(0/0) = 0`, `q · f(0) = q · f(0⁺)`
//! when `pᵢ = 0`, and `pᵢ · f′(∞)` (the recession constant
//! [`FGenerator::f_infinity`]) when `qᵢ = 0 < pᵢ` — so support mismatch yields a
//! finite value where the generator allows it (total variation, Hellinger) and
//! `+∞` where it does not (KL, χ²).

use crate::error::BayesError;

/// A convex **generator** `f : (0, ∞) → ℝ` with `f(1) = 0`, defining the
/// f-divergence `D_f(p ‖ q) = Σ qᵢ f(pᵢ/qᵢ)` via [`f_divergence`].
///
/// Implementors supply [`f`](Self::f) (whose value at `0` must be the limit
/// `f(0⁺)`) and, where the generator stays finite under support mismatch, the
/// recession constant [`f_infinity`](Self::f_infinity). Convexity and `f(1) = 0`
/// are the implementor's contract; they are what make every resulting divergence
/// nonnegative, data-processing-monotone, and jointly convex
/// (`RiemannLean/Bayes/Divergence.lean`: `fDiv_nonneg`, `fDiv_dataProcessing`,
/// `fDiv_convex`).
pub trait FGenerator {
    /// `f(t)` for `t ≥ 0`. The value at `t = 0` must be the right limit `f(0⁺)`
    /// (`0` for [`Kl`], `+∞` for [`ReverseKl`]/[`Jeffreys`], `½` for
    /// [`SquaredHellinger`]/[`TotalVariation`], `1` for [`ChiSquared`]).
    fn f(&self, t: f64) -> f64;

    /// The **recession constant** `f′(∞) = lim_{t→∞} f(t)/t`, used for the
    /// `qᵢ = 0 < pᵢ` term `pᵢ · f′(∞)`. Defaults to `+∞` (KL, χ², Jeffreys —
    /// the second distribution must dominate the first); override with the
    /// finite value for generators that tolerate disjoint support
    /// (`½` for total variation and squared Hellinger, `0` for reverse KL).
    fn f_infinity(&self) -> f64 {
        f64::INFINITY
    }

    /// One term `qᵢ f(pᵢ/qᵢ)` of the divergence sum, with the limit conventions
    /// for the `qᵢ = 0` and `pᵢ = qᵢ = 0` cells.
    fn term(&self, p: f64, q: f64) -> f64 {
        if q > 0.0 {
            q * self.f(p / q)
        } else if p > 0.0 {
            p * self.f_infinity()
        } else {
            0.0
        }
    }
}

/// `D_f(p ‖ q) = Σ qᵢ f(pᵢ/qᵢ)` for the given `generator`, over two probability
/// vectors aligned by index (the shorter is zero-padded). `≥ 0` for any convex
/// generator with `f(1) = 0` (`fDiv_nonneg`); `0` when `p == q`.
///
/// ```
/// use bayes::{f_divergence, Kl, ChiSquaredGen};
/// let p = [0.5, 0.5];
/// let q = [0.25, 0.75];
/// let kl = f_divergence(&p, &q, &Kl);
/// let chi2 = f_divergence(&p, &q, &ChiSquaredGen);
/// assert!(kl <= chi2);            // KL ≤ χ² (klDiv_le_chiSqDiv)
/// assert!(f_divergence(&p, &p, &Kl).abs() < 1e-15);  // self-divergence is 0
/// ```
pub fn f_divergence<G: FGenerator + ?Sized>(p: &[f64], q: &[f64], generator: &G) -> f64 {
    let n = p.len().max(q.len());
    (0..n)
        .map(|i| {
            let pi = p.get(i).copied().unwrap_or(0.0);
            let qi = q.get(i).copied().unwrap_or(0.0);
            generator.term(pi, qi)
        })
        .sum()
}

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
/// use bayes::{f_divergence, Custom};
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

/// A **row-stochastic channel** (Markov kernel) `K`: `inputs` rows, `outputs`
/// columns, each row a probability vector. Post-processing a distribution `p` is
/// the pushforward `(pK)ₖ = Σᵢ pᵢ Kᵢₖ` ([`apply`](Self::apply)).
///
/// The **data-processing inequality** is the reason channels matter here: for
/// *every* f-divergence,
///
/// ```text
/// D_f(pK ‖ qK) ≤ D_f(p ‖ q)
/// ```
///
/// — proved in `riemann` (`RiemannLean/Bayes/Divergence.lean`:
/// `fDiv_dataProcessing`, with the KL specialization `klDiv_dataProcessing` and
/// the deterministic two-group case `klDiv_ge_binary`). No channel — coarse-
/// graining, marginalizing, adding noise, any Markov post-processing — can
/// increase the divergence between two distributions; it only ever loses
/// discriminating information.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    /// `rows[i]` is the output distribution given input `i`; each sums to `1`.
    rows: Vec<Vec<f64>>,
    outputs: usize,
}

impl Channel {
    /// Build a channel from its rows, validating that it is rectangular and
    /// row-stochastic (each row finite, non-negative, summing to `1`).
    pub fn new(rows: Vec<Vec<f64>>) -> Result<Self, BayesError> {
        let outputs = rows.first().map(Vec::len).unwrap_or(0);
        if rows.is_empty() || outputs == 0 {
            return Err(BayesError::Malformed {
                what: "channel needs at least one input row and one output column",
            });
        }
        for row in &rows {
            if row.len() != outputs {
                return Err(BayesError::Malformed {
                    what: "channel rows must all have the same number of outputs",
                });
            }
            let mut sum = 0.0;
            for &v in row {
                if !v.is_finite() || v < 0.0 {
                    return Err(BayesError::Malformed {
                        what: "channel entries must be finite and non-negative",
                    });
                }
                sum += v;
            }
            if (sum - 1.0).abs() > 1e-9 {
                return Err(BayesError::Malformed {
                    what: "each channel row must sum to 1",
                });
            }
        }
        Ok(Self { rows, outputs })
    }

    /// A deterministic **coarse-graining** channel: input `i` is sent to group
    /// `assignment[i]` among `groups` outputs (a 0/1 kernel). Merging categories
    /// is a channel, so the DPI specializes to "coarse-graining only lowers the
    /// divergence" — the `riemann` two-group case is `klDiv_ge_binary`.
    pub fn coarse_grain(assignment: &[usize], groups: usize) -> Result<Self, BayesError> {
        if assignment.is_empty() || groups == 0 {
            return Err(BayesError::Malformed {
                what: "coarse-graining needs at least one input and one group",
            });
        }
        let mut rows = Vec::with_capacity(assignment.len());
        for &g in assignment {
            if g >= groups {
                return Err(BayesError::Malformed {
                    what: "coarse-graining group index out of range",
                });
            }
            let mut row = vec![0.0; groups];
            row[g] = 1.0;
            rows.push(row);
        }
        Ok(Self {
            rows,
            outputs: groups,
        })
    }

    /// The number of input categories (rows).
    pub fn inputs(&self) -> usize {
        self.rows.len()
    }

    /// The number of output categories (columns).
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// The pushforward `pK` — the output distribution `(pK)ₖ = Σᵢ pᵢ Kᵢₖ`. The
    /// input `p` is read positionally over the channel's inputs (zero-padded /
    /// truncated to [`inputs`](Self::inputs)); the result has
    /// [`outputs`](Self::outputs) entries and is itself a probability vector
    /// when `p` is.
    pub fn apply(&self, p: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.outputs];
        for (i, row) in self.rows.iter().enumerate() {
            let pi = p.get(i).copied().unwrap_or(0.0);
            if pi == 0.0 {
                continue;
            }
            for (k, &kik) in row.iter().enumerate() {
                out[k] += pi * kik;
            }
        }
        out
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::{Categorical, KlDivergence};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    // A small bank of full-support distribution pairs for the property checks.
    fn pairs() -> Vec<(Vec<f64>, Vec<f64>)> {
        vec![
            (vec![0.5, 0.5], vec![0.25, 0.75]),
            (vec![0.2, 0.3, 0.5], vec![0.5, 0.1, 0.4]),
            (vec![0.1, 0.2, 0.3, 0.4], vec![0.4, 0.3, 0.2, 0.1]),
            (vec![0.7, 0.2, 0.1], vec![0.34, 0.33, 0.33]),
        ]
    }

    #[test]
    fn kl_generator_matches_the_per_family_discrete_kl() {
        // The umbrella's KL instance reproduces the parent module's `discrete_kl`
        // (and `Categorical::kl_divergence`) on equal, full support.
        for (p, q) in pairs() {
            let cat = Categorical::new(&p)
                .unwrap()
                .kl_divergence(&Categorical::new(&q).unwrap());
            close(f_divergence(&p, &q, &Kl), cat, 1e-12);
        }
    }

    #[test]
    fn named_generators_match_their_closed_forms() {
        for (p, q) in pairs() {
            // χ² = Σ (pᵢ−qᵢ)²/qᵢ.
            let chi2: f64 = p.iter().zip(&q).map(|(&a, &b)| (a - b).powi(2) / b).sum();
            close(f_divergence(&p, &q, &ChiSquared), chi2, 1e-12);
            // ½ Σ (√pᵢ−√qᵢ)².
            let hel: f64 = 0.5
                * p.iter()
                    .zip(&q)
                    .map(|(&a, &b)| (a.sqrt() - b.sqrt()).powi(2))
                    .sum::<f64>();
            close(f_divergence(&p, &q, &SquaredHellinger), hel, 1e-12);
            // ½ Σ |pᵢ−qᵢ|.
            let tv: f64 = 0.5 * p.iter().zip(&q).map(|(&a, &b)| (a - b).abs()).sum::<f64>();
            close(f_divergence(&p, &q, &TotalVariation), tv, 1e-12);
        }
    }

    #[test]
    fn nonnegative_and_zero_on_self() {
        let gens: [&dyn FGenerator; 7] = [
            &Kl,
            &ReverseKl,
            &ChiSquared,
            &SquaredHellinger,
            &TotalVariation,
            &Jeffreys,
            &Alpha(2.0),
        ];
        for (p, q) in pairs() {
            for g in gens {
                // `f_divergence` accepts `&dyn FGenerator` (the `?Sized` bound),
                // so generators can be held and dispatched dynamically.
                assert!(
                    f_divergence(&p, &q, g) >= -1e-12,
                    "f-divergence went negative"
                );
                close(f_divergence(&p, &p, g), 0.0, 1e-12);
            }
        }
    }

    #[test]
    fn reverse_kl_swaps_arguments_and_jeffreys_is_the_sum() {
        for (p, q) in pairs() {
            close(
                f_divergence(&p, &q, &ReverseKl),
                f_divergence(&q, &p, &Kl),
                1e-12,
            );
            close(
                f_divergence(&p, &q, &Jeffreys),
                f_divergence(&p, &q, &Kl) + f_divergence(&q, &p, &Kl),
                1e-12,
            );
        }
    }

    #[test]
    fn kl_is_below_chi_squared() {
        // klDiv_le_chiSqDiv, instantiated through the umbrella.
        for (p, q) in pairs() {
            assert!(f_divergence(&p, &q, &Kl) <= f_divergence(&p, &q, &ChiSquared) + 1e-12);
        }
    }

    #[test]
    fn alpha_recovers_the_named_generators_at_special_orders() {
        for (p, q) in pairs() {
            // α → 1 is KL (returned exactly at α = 1).
            close(
                f_divergence(&p, &q, &Alpha(1.0)),
                f_divergence(&p, &q, &Kl),
                1e-12,
            );
            // α → 0 is reverse KL.
            close(
                f_divergence(&p, &q, &Alpha(0.0)),
                f_divergence(&p, &q, &ReverseKl),
                1e-12,
            );
            // α = 2 is ½ χ².
            close(
                f_divergence(&p, &q, &Alpha(2.0)),
                0.5 * f_divergence(&p, &q, &ChiSquared),
                1e-12,
            );
            // α = ½ is 4 × squared Hellinger.
            close(
                f_divergence(&p, &q, &Alpha(0.5)),
                4.0 * f_divergence(&p, &q, &SquaredHellinger),
                1e-12,
            );
        }
    }

    #[test]
    fn custom_generator_matches_a_named_one() {
        let chi2 = Custom::new(|t: f64| (t - 1.0).powi(2));
        for (p, q) in pairs() {
            close(
                f_divergence(&p, &q, &chi2),
                f_divergence(&p, &q, &ChiSquared),
                1e-12,
            );
        }
    }

    #[test]
    fn support_mismatch_is_finite_or_infinite_per_generator() {
        // q lacks mass where p has it: KL/χ² diverge, TV/Hellinger stay finite.
        let p = [0.5, 0.5];
        let q = [1.0, 0.0];
        assert!(f_divergence(&p, &q, &Kl).is_infinite());
        assert!(f_divergence(&p, &q, &ChiSquared).is_infinite());
        assert!(f_divergence(&p, &q, &TotalVariation).is_finite());
        assert!(f_divergence(&p, &q, &SquaredHellinger).is_finite());
        // TV of disjoint-ish supports: ½(|0.5−1| + |0.5−0|) = 0.5.
        close(f_divergence(&p, &q, &TotalVariation), 0.5, 1e-12);
        // Extra category in q that p never touches is harmless for KL.
        close(
            f_divergence(&[0.5, 0.5], &[0.25, 0.25, 0.5], &Kl),
            f_divergence(&[0.5, 0.5, 0.0], &[0.25, 0.25, 0.5], &Kl),
            1e-12,
        );
    }

    #[test]
    fn channel_apply_is_the_pushforward_and_stays_a_distribution() {
        // A 2→2 channel: identity-ish mixing.
        let k = Channel::new(vec![vec![0.9, 0.1], vec![0.2, 0.8]]).unwrap();
        assert_eq!((k.inputs(), k.outputs()), (2, 2));
        let out = k.apply(&[0.6, 0.4]);
        // (0.6·0.9 + 0.4·0.2, 0.6·0.1 + 0.4·0.8) = (0.62, 0.38).
        close(out[0], 0.62, 1e-12);
        close(out[1], 0.38, 1e-12);
        close(out.iter().sum::<f64>(), 1.0, 1e-12);
    }

    #[test]
    fn data_processing_inequality_holds_for_every_generator() {
        // D_f(pK ‖ qK) ≤ D_f(p ‖ q) — fDiv_dataProcessing, checked numerically.
        let k = Channel::new(vec![
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.6, 0.3],
            vec![0.25, 0.25, 0.5],
        ])
        .unwrap();
        for (p, q) in pairs().into_iter().filter(|(p, _)| p.len() == 3) {
            let (pk, qk) = (k.apply(&p), k.apply(&q));
            for (name, before, after) in [
                ("KL", f_divergence(&p, &q, &Kl), f_divergence(&pk, &qk, &Kl)),
                (
                    "chi2",
                    f_divergence(&p, &q, &ChiSquared),
                    f_divergence(&pk, &qk, &ChiSquared),
                ),
                (
                    "TV",
                    f_divergence(&p, &q, &TotalVariation),
                    f_divergence(&pk, &qk, &TotalVariation),
                ),
                (
                    "H2",
                    f_divergence(&p, &q, &SquaredHellinger),
                    f_divergence(&pk, &qk, &SquaredHellinger),
                ),
            ] {
                assert!(
                    after <= before + 1e-12,
                    "DPI failed for {name}: {after} > {before}"
                );
            }
        }
    }

    #[test]
    fn coarse_graining_merges_categories_and_only_lowers_divergence() {
        // Merge categories {0,1} → 0 and {2} → 1 of a 3-category support.
        let k = Channel::coarse_grain(&[0, 0, 1], 2).unwrap();
        let p = [0.2, 0.3, 0.5];
        let q = [0.5, 0.1, 0.4];
        let pk = k.apply(&p);
        let qk = k.apply(&q);
        close(pk[0], 0.5, 1e-12); // 0.2 + 0.3
        close(pk[1], 0.5, 1e-12);
        close(qk[0], 0.6, 1e-12); // 0.5 + 0.1
        close(qk[1], 0.4, 1e-12);
        // klDiv_ge_binary: collapsing onto the two-group partition lowers KL.
        assert!(f_divergence(&pk, &qk, &Kl) <= f_divergence(&p, &q, &Kl) + 1e-12);
    }

    #[test]
    fn channel_rejects_non_stochastic_input() {
        assert!(Channel::new(vec![]).is_err());
        assert!(Channel::new(vec![vec![0.5, 0.4]]).is_err()); // row sum ≠ 1
        assert!(Channel::new(vec![vec![0.5, 0.5], vec![1.0]]).is_err()); // ragged
        assert!(Channel::new(vec![vec![1.5, -0.5]]).is_err()); // negative entry
        assert!(Channel::coarse_grain(&[0, 2], 2).is_err()); // group index out of range
    }
}
