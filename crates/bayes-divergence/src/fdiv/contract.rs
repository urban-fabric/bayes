//! The two generator contracts.
//!
//! An *f*-divergence is `D_f(p || q) = Sum q_i f(p_i/q_i)` for a convex `f` with
//! `f(1) = 0`, so a generator is the whole specification of a divergence. The
//! two traits here are the eager and carrier-generic halves of that: the second
//! exists so a generator can be evaluated on a graph node, where `f` cannot
//! branch and a literal cannot be written.
//!
//! The convexity and `f(1) = 0` conditions are what deliver the three universal
//! guarantees — non-negativity, data-processing monotonicity, joint convexity —
//! so a generator that violates them is not merely inaccurate, it breaks the
//! theorems the whole module relies on.
use euclid_numeric::Real;

/// A convex **generator** `f : (0, ∞) → ℝ` with `f(1) = 0`, defining the
/// f-divergence `D_f(p ‖ q) = Σ qᵢ f(pᵢ/qᵢ)` via [`super::f_divergence`].
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
    /// (`0` for [`super::Kl`], `+∞` for [`super::ReverseKl`]/[`super::Jeffreys`], `½` for
    /// [`super::SquaredHellinger`]/[`super::TotalVariation`], `1` for [`super::ChiSquared`]).
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
