//! **Divergence laws**: the inequalities this crate documents as proved.
//!
//! The crate's own documentation lists what the `riemann` Lean development
//! machine-checks about these quantities — Gibbs' `KL ≥ 0`, `0 ≤ H² ≤ 1`,
//! `H² ≤ ½KL`, `KL ≤ χ²`, `D_f ≥ 0`. Those proofs are about the *mathematics*.
//! Whether the closed forms in this crate satisfy them is a separate question,
//! and the right one to ask with a test: a wrong constant in one family's
//! formula breaks an inequality against a *different* family's formula, which
//! no amount of internal consistency can detect.
//!
//! Two cross-identities make the net tighter still, because they tie
//! independently written formulas to each other exactly rather than by
//! inequality:
//!
//! ```text
//!     R_{1/2}(p ‖ q) = −2 ln(1 − H²(p ‖ q))
//!     R_2(p ‖ q)     = ln(1 + χ²(p ‖ q))
//! ```
//!
//! | module | asks |
//! |--------|------|
//! | [`inequalities`] | do the documented bounds hold across every family and parameter pair? |
//! | [`adverse`] | what happens at the boundary, past a convergence condition, and on `NaN`? |
//!
//! [`adverse`] is where the bugs were. A divergence has three distinct ways to
//! be "large" — a genuine infinity from disjoint supports, an infinity from a
//! convergence condition that fails, and a `NaN` from an input that was never a
//! number — and the natural way to write a convergence guard conflates the
//! third with the second. Every comparison against `NaN` is false, so
//! `select(converges, value, inf)` reports a confident infinity for garbage
//! input. The mirror of the same mistake in `max(0)` made a Hellinger distance
//! report `0`, which reads as *identical distributions*.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod adverse;
mod inequalities;

/// Absolute-plus-relative closeness.
///
/// Every quantity here vanishes at a self-comparison, so a purely relative
/// bound has no reference to measure against and flags floating-point zero
/// against floating-point zero. The absolute floor is what makes the identity
/// checks meaningful at both ends of the range.
pub(crate) fn close(a: f64, b: f64, relative: f64, absolute: f64) -> bool {
    (a - b).abs() <= relative * b.abs() + absolute
}
