//! The six corrections, each returning adjusted p-values in the input order.
//!
//! All of them share one structure: sort, walk the sorted list accumulating a
//! running bound, enforce monotonicity, and scatter back. The differences are
//! only in the per-rank multiplier and the direction of the monotone sweep —
//! which is why they live in one file rather than six.

use super::step::{order_ascending, order_descending, scatter};

/// Validate shared preconditions: no `NaN`, every value in `[0, 1]`.
fn usable(p_values: &[f64]) -> bool {
    !p_values.is_empty()
        && p_values
            .iter()
            .all(|&p| !p.is_nan() && (0.0..=1.0).contains(&p))
}

/// **Bonferroni**: multiply every p-value by the number of tests.
///
/// The simplest FWER control and the most conservative. It assumes nothing at
/// all about dependence between the tests, which is both its virtue and its
/// cost: it is valid always, and it is wasteful whenever the tests are
/// positively correlated (as they usually are).
///
/// Returns an empty vector for empty input or any `NaN`/out-of-range p-value.
pub fn bonferroni(p_values: &[f64]) -> Vec<f64> {
    if !usable(p_values) {
        return Vec::new();
    }
    let m = p_values.len() as f64;
    p_values.iter().map(|&p| (p * m).min(1.0)).collect()
}

/// **Šidák**: `1 - (1-p)^m`.
///
/// Slightly less conservative than Bonferroni and exact rather than approximate
/// — but only under **independence**, which Bonferroni does not require. Prefer
/// Bonferroni unless independence is actually defensible; the gain is small
/// (`1 - (1-p)^m` versus `pm` differ at second order) and the assumption is not.
pub fn sidak(p_values: &[f64]) -> Vec<f64> {
    if !usable(p_values) {
        return Vec::new();
    }
    let m = p_values.len() as f64;
    p_values
        .iter()
        .map(|&p| (-((1.0 - p).powf(m))).mul_add(1.0, 1.0).clamp(0.0, 1.0))
        .collect()
}

/// **Holm**: Bonferroni applied step-down, so it is uniformly more powerful.
///
/// Sort ascending and scale the `i`-th smallest by `m - i` rather than by `m`.
/// Holm dominates Bonferroni — it rejects everything Bonferroni does and
/// sometimes more — under exactly the same (absent) dependence assumptions, so
/// there is no reason to prefer plain Bonferroni except familiarity.
///
/// The running maximum is what makes the result a valid p-value: adjusted values
/// must be monotone in the original ordering, or a smaller raw p-value could end
/// up with a larger adjusted one.
pub fn holm(p_values: &[f64]) -> Vec<f64> {
    if !usable(p_values) {
        return Vec::new();
    }
    let m = p_values.len();
    let idx = order_ascending(p_values);
    let mut running = 0.0_f64;
    let mut sorted_adjusted = Vec::with_capacity(m);
    for (rank, &i) in idx.iter().enumerate() {
        let scaled = (p_values[i] * (m - rank) as f64).min(1.0);
        running = running.max(scaled);
        sorted_adjusted.push(running);
    }
    scatter(&idx, &sorted_adjusted)
}

/// **Holm–Šidák**: the step-down form of Šidák.
///
/// Stands to Šidák as [`holm`] stands to [`bonferroni`], and carries the same
/// independence assumption.
pub fn holm_sidak(p_values: &[f64]) -> Vec<f64> {
    if !usable(p_values) {
        return Vec::new();
    }
    let m = p_values.len();
    let idx = order_ascending(p_values);
    let mut running = 0.0_f64;
    let mut sorted_adjusted = Vec::with_capacity(m);
    for (rank, &i) in idx.iter().enumerate() {
        let remaining = (m - rank) as f64;
        let scaled = 1.0 - (1.0 - p_values[i]).powf(remaining);
        running = running.max(scaled.clamp(0.0, 1.0));
        sorted_adjusted.push(running);
    }
    scatter(&idx, &sorted_adjusted)
}

/// **Benjamini–Hochberg**: control the false discovery rate.
///
/// Sort ascending and scale the `i`-th smallest (1-based) by `m/i`, then sweep
/// *down* taking a running minimum. The step-up direction is the part that is
/// easy to get backwards, and getting it backwards produces adjusted values that
/// look plausible and control nothing.
///
/// Valid under independence and under positive regression dependence, which
/// covers most realistic test batteries. For arbitrary dependence use
/// [`benjamini_yekutieli`].
pub fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    if !usable(p_values) {
        return Vec::new();
    }
    let m = p_values.len();
    let m_f = m as f64;
    // Descending so the running minimum sweeps from the largest p downward,
    // which is what "step-up" means here.
    let idx = order_descending(p_values);
    let mut running = 1.0_f64;
    let mut adjusted_in_desc = Vec::with_capacity(m);
    for (offset, &i) in idx.iter().enumerate() {
        let rank = m - offset; // 1-based rank in ascending order
        let scaled = p_values[i] * m_f / rank as f64;
        running = running.min(scaled.min(1.0));
        adjusted_in_desc.push(running);
    }
    scatter(&idx, &adjusted_in_desc)
}

/// **Benjamini–Yekutieli**: FDR control under *arbitrary* dependence.
///
/// Benjamini–Hochberg scaled by the harmonic number `H(m) = Σ 1/k`, which is the
/// price of dropping the positive-dependence assumption. At `m = 1000` that
/// factor is about 7.5, so this sits between BH and Bonferroni in
/// conservatism — use it when the tests may be dependent in unknown ways.
pub fn benjamini_yekutieli(p_values: &[f64]) -> Vec<f64> {
    if !usable(p_values) {
        return Vec::new();
    }
    let m = p_values.len();
    let harmonic: f64 = (1..=m).map(|k| 1.0 / k as f64).sum();
    let m_f = m as f64;
    let idx = order_descending(p_values);
    let mut running = 1.0_f64;
    let mut adjusted_in_desc = Vec::with_capacity(m);
    for (offset, &i) in idx.iter().enumerate() {
        let rank = m - offset;
        let scaled = p_values[i] * m_f * harmonic / rank as f64;
        running = running.min(scaled.min(1.0));
        adjusted_in_desc.push(running);
    }
    scatter(&idx, &adjusted_in_desc)
}
