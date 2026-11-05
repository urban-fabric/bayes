//! **Scoring rules**: the four forecast scores, against the property that
//! defines them.
//!
//! A scoring rule is **proper** when a forecaster minimizes their expected
//! penalty by stating what they actually believe. That is not a nice-to-have
//! property that follows from the formula — it *is* the formula's reason to
//! exist. A rule that is not proper rewards lying, and every one of these four
//! is wrong in a way that breaks it: a sign flipped, an exponent changed, a
//! cumulative sum off by one.
//!
//! So [`propriety`] is the load-bearing module, and it is checkable directly.
//! For a true distribution `q`, the expected loss `Σᵢ qᵢ·L(p, i)` must be
//! smallest at `p = q`. Nothing about that needs a reference table: the minimum
//! is at a point known in advance, and a broken rule puts it somewhere else.
//!
//! | module | asks |
//! |--------|------|
//! | [`propriety`] | is honesty optimal? |
//! | [`identities`] | do the closed forms agree with algebraically different ones? |
//! | [`degenerate`] | is a malformed forecast refused rather than scored? |
//!
//! [`identities`] matters because propriety is a *shape* property — it can hold
//! for a rule that is off by a constant or a factor. The Brier score has an
//! exact alternative form, `1 − 2p_obs + Σpᵢ²`, reached by expanding the square
//! rather than summing it; the ranked score over two categories must be exactly
//! half the Brier score. Those pin the value, not just the shape.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod degenerate;
mod identities;
mod propriety;

/// A forecast loss: lower is better.
pub(crate) type Loss = fn(&[f64], usize) -> f64;

/// The four rules, each expressed as a **loss** so one check covers them all.
///
/// `log_score` and `spherical_score` are stated as scores — higher is better —
/// so they are negated here. Getting that wrong would make the propriety check
/// assert the opposite of the truth and pass only for a broken rule, which is
/// why the conversion lives in one place.
pub(crate) fn losses() -> Vec<(&'static str, Loss)> {
    use bayes_core::{brier_score, log_loss, ranked_probability_score, spherical_score};
    vec![
        ("log_loss", log_loss),
        ("brier_score", brier_score),
        ("ranked_probability_score", ranked_probability_score),
        ("negated spherical_score", |p, i| -spherical_score(p, i)),
    ]
}

/// `Σᵢ qᵢ·L(p, i)`, the expected loss of forecasting `p` when the truth is `q`.
///
/// Categories with `qᵢ = 0` are skipped rather than multiplied, because
/// `0 · inf` is `NaN`: a zero-probability category contributes nothing to an
/// expectation however badly the forecast scores there.
pub(crate) fn expected_loss(loss: Loss, p: &[f64], q: &[f64]) -> f64 {
    q.iter()
        .enumerate()
        .filter(|&(_, &qi)| qi > 0.0)
        .map(|(i, &qi)| qi * loss(p, i))
        .sum()
}
