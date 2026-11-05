//! **Far tail**: the region `tests/conformance/` structurally cannot reach.
//!
//! Every conformance check walks `x` across `[Q(1e-9), Q(1 − 1e-9)]`. Inside
//! that window `1 − cdf(x)` is accurate to nine digits, so the whole class of
//! upper-tail defects was invisible to it — and the class was not small:
//!
//! - `sf` returned **exactly zero** for every family past a survival of about
//!   `1e-16`, because it was `1 − cdf` and there is no `f64` between
//!   `1 − 1.1e-16` and `1`. A standard normal's `sf(9.0)` is `1.13e-19`, the
//!   p-value of a nine-sigma result.
//! - `isf(p)` returned the support bound for every `p` below the same point,
//!   because it formed `1 − p` first. `isf` was also absent from the canonical
//!   sweep's view of a family, so nothing could have tested it.
//! - `Normal::inverse_cdf` was **wrong by 4%** at `p = 1e-15` and infinite
//!   below it: it computed `erf_inv(2p − 1)`, and `2p − 1` is exactly `−1` once
//!   `p` drops under `1e-16`.
//!
//! None of that is reachable from inside the window, and none of it is exotic.
//!
//! | module | asks |
//! |--------|------|
//! | [`survival`] | is the upper tail *computed*, or subtracted away? |
//! | [`quantile`] | does a tail probability round trip back to the point it came from? |
//!
//! The load-bearing check is the round trip in [`quantile`]. `isf(sf(x)) == x`
//! couples the two independent tail paths — a log-survival built from `erfcx`,
//! `ln_1m_exp` or a reflected incomplete beta one way, a quantile the other —
//! into one equation whose answer is known exactly beforehand. That is what
//! exposed the normal quantile; no test of either half alone can see it, which
//! is precisely why a suite of self-consistent halves had missed it.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod quantile;
mod survival;

/// Families whose support is bounded above, so they have no upper tail to
/// probe. Checks that march outward skip these by name rather than by guessing
/// from `hi()`, so that adding a bounded family is a deliberate act.
pub(crate) fn bounded_above(name: &str) -> bool {
    name.starts_with("Uniform")
        || name.starts_with("Triangular")
        || name.starts_with("Beta")
        || name.starts_with("VonMises")
}

/// Relative error, floored so a comparison against zero does not divide by it.
pub(crate) fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}
