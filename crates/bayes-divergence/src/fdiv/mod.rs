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

use bayes_core::{BayesError, validate_row_stochastic};
use euclid::Real;

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

/// A convex generator over an arbitrary [`Real`] carrier.
///
/// This is the lazy/backend-generic companion to [`FGenerator`]. It preserves the same
/// f-divergence limit conventions while expressing support cases with value-level masks rather
/// than host numeric branching.
pub trait RealFGenerator<R: Real> {
    /// `f(t)` for `t >= 0`, with the same right-limit convention at `0` as [`FGenerator::f`].
    fn f_real(&self, t: R) -> R;

    /// The recession constant `f'(infinity) = lim t->infinity f(t)/t`.
    fn f_infinity_real(&self) -> R {
        R::infinity()
    }

    /// One term `q_i f(p_i/q_i)` of the divergence sum.
    fn term_real(&self, p: R, q: R) -> R {
        let q_positive = q.gt(&R::zero());
        let p_positive = p.gt(&R::zero());
        let regular = q.clone() * self.f_real(p.clone() / q);
        let mismatch = p * self.f_infinity_real();
        R::select(
            &q_positive,
            regular,
            R::select(&p_positive, mismatch, R::zero()),
        )
    }
}

/// `D_f(p ‖ q) = Σ qᵢ f(pᵢ/qᵢ)` for the given `generator`, over two probability
/// vectors aligned by index (the shorter is zero-padded). `≥ 0` for any convex
/// generator with `f(1) = 0` (`fDiv_nonneg`); `0` when `p == q`.
///
/// ```
/// use bayes_divergence::{f_divergence, Kl, ChiSquaredGen};
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

/// `D_f(p || q) = sum_i q_i f(p_i/q_i)` over two probability vectors in any [`Real`]
/// carrier.
///
/// This is the numeric reduction only: it assumes the inputs represent probability vectors and
/// uses zero-padding for unequal lengths, mirroring [`f_divergence`].
pub fn f_divergence_real<R, G>(p: &[R], q: &[R], generator: &G) -> R
where
    R: Real,
    G: RealFGenerator<R> + ?Sized,
{
    let n = p.len().max(q.len());
    (0..n)
        .map(|i| {
            let pi = p.get(i).cloned().unwrap_or_else(R::zero);
            let qi = q.get(i).cloned().unwrap_or_else(R::zero);
            generator.term_real(pi, qi)
        })
        .fold(R::zero(), |sum, term| sum + term)
}

mod channel;
mod generators;

pub use channel::Channel;
pub use generators::{
    Alpha, ChiSquared, Custom, Jeffreys, Kl, ReverseKl, SquaredHellinger, TotalVariation,
};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{Categorical, KlDivergence};

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
