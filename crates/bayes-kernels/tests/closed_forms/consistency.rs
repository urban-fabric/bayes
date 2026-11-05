//! Does each kernel agree with its own logarithm and complement?
//!
//! Internal checks: they cannot tell whether a formula is the right
//! distribution — [`super::reductions`] is for that — but they are the ones
//! that catch a *stable* form drifting from the direct one it was supposed to
//! replace, which is the specific hazard of a crate full of `erfcx`,
//! `ln_1m_exp` and reflected incomplete betas.

use super::rel;
use super::table::cases;
use bayes_utils::Failures;

/// Probabilities from the lower tail to the upper, used for both round trips.
const PROBABILITIES: [f64; 11] = [
    1e-300,
    1e-100,
    1e-30,
    1e-8,
    1e-3,
    0.1,
    0.5,
    0.9,
    0.999,
    1.0 - 1e-8,
    1.0 - 1e-12,
];

/// `cdf + exp(ln_sf) == 1` wherever both carry information.
///
/// Guarded to the body, because that is the only place the statement is
/// meaningful: out in either tail one of the two *is* 1 to within an ulp, and
/// demanding they sum exactly would be demanding the impossible rather than
/// testing anything.
#[test]
fn the_cdf_and_the_log_survival_are_complements() {
    let mut f = Failures::default();
    for item in cases() {
        for p in [0.05, 0.25, 0.5, 0.75, 0.95] {
            // Walk to a body point without needing a quantile: bisect the cdf.
            let x = bisect_cdf(&item.cdf, item.lo, p);
            if !x.is_finite() {
                continue;
            }
            let total = (item.cdf)(x) + (item.ln_sf)(x).exp();
            f.close(
                || format!("{}: cdf + exp(ln_sf) at the {p} quantile", item.name),
                total,
                1.0,
                1e-11,
            );
        }
    }
    f.finish("cdf / ln_sf complement");
}

/// A quantile must invert its own CDF, across the full representable range of
/// `p`.
///
/// The bound widens in the upper tail and nowhere else, and for a stated
/// reason: `p = 1 − 1e-12` is a representable number but the *distance* from 1
/// is known to only four digits once it has been rounded, so a quantile there
/// cannot be inverted more precisely than that. The lower tail has no such
/// excuse — `1e-300` is exact — and is held to `1e-9` throughout, which is
/// what catches a kernel forming `ln(1 − p)`.
#[test]
fn every_quantile_inverts_its_own_cdf() {
    let mut f = Failures::default();
    for item in cases() {
        let Some(quantile) = item.quantile.as_ref() else {
            continue;
        };
        for p in PROBABILITIES {
            let x = quantile(p);
            if !x.is_finite() || x <= item.lo {
                continue;
            }
            let back = (item.cdf)(x);
            if back == 0.0 || back >= 1.0 {
                // The CDF cannot resolve the point its own quantile produced;
                // `stability` reports saturation, not this.
                continue;
            }
            // Two separate representation limits, neither of them the formula's
            // fault, so both are derived rather than guessed.
            //
            // Upper tail: `1 - p` has already lost its low-order digits before
            // the quantile sees it.
            //
            // Near a **shifted support bound**: the returned point cannot carry
            // its own offset. `Pareto(2, 1.3)` has `Q(1e-8) = 2.0000000154`,
            // whose ulp is `4.4e-16` — so the offset `1.54e-8` is encoded to a
            // relative accuracy of only `3e-8`, and no CDF can recover `p`
            // better than that from the number it was handed. The bound is
            // `ulp(x) / (x − lo)`.
            let upper = if p > 0.5 {
                4.0 * f64::EPSILON / (1.0 - p)
            } else {
                0.0
            };
            let offset = if item.lo.is_finite() {
                4.0 * f64::EPSILON * x.abs() / (x - item.lo).abs().max(f64::MIN_POSITIVE)
            } else {
                0.0
            };
            let bound = upper.max(offset).max(1e-9);
            f.check(rel(back, p) <= bound, || {
                format!(
                    "{}: quantile({p:.3e}) = {x:.10e}, whose cdf is {back:.10e} — relative \
                     error {:.2e} against a bound of {bound:.2e}",
                    item.name,
                    rel(back, p)
                )
            });
        }
    }
    f.finish("quantile inverts cdf");
}

/// `ln_sf` must be non-increasing, and `cdf` non-decreasing, across many
/// decades.
///
/// Monotonicity is the cheapest way to catch a branch that disagrees with its
/// neighbour. Every stable form in this crate switches branch somewhere — at a
/// median, at a midpoint, at zero — and a seam that does not line up shows up
/// here as a reversal while every single-point check passes.
#[test]
fn the_branch_seams_line_up() {
    let mut f = Failures::default();
    for item in cases() {
        let start = if item.lo.is_finite() { item.lo } else { -1e4 };
        // A geometric fan either side of the support bound, so the sweep lands
        // on both sides of any midpoint or median seam.
        let mut points: Vec<f64> = Vec::new();
        for k in -20..=20 {
            let step = 10.0_f64.powi(k);
            points.push(start + step);
            if !item.lo.is_finite() {
                points.push(start - step);
            }
        }
        points.retain(|x| x.is_finite());
        points.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let (mut previous_cdf, mut previous_ln_sf) = (-1.0_f64, f64::INFINITY);
        for x in points {
            let (c, ls) = ((item.cdf)(x), (item.ln_sf)(x));
            f.check(c.is_finite() && (0.0..=1.0).contains(&c), || {
                format!("{}: cdf({x:.4e}) = {c} is not a probability", item.name)
            });
            f.check(c >= previous_cdf - 1e-12, || {
                format!(
                    "{}: cdf fell at {x:.4e}: {previous_cdf:.12e} -> {c:.12e}",
                    item.name
                )
            });
            f.check(ls <= previous_ln_sf + 1e-12 || !ls.is_finite(), || {
                format!(
                    "{}: ln_sf rose at {x:.4e}: {previous_ln_sf:.12e} -> {ls:.12e}",
                    item.name
                )
            });
            f.check(ls <= 1e-12, || {
                format!(
                    "{}: ln_sf({x:.4e}) = {ls} is positive, so sf exceeds 1",
                    item.name
                )
            });
            previous_cdf = c.max(previous_cdf);
            if ls.is_finite() {
                previous_ln_sf = ls.min(previous_ln_sf);
            }
        }
    }
    f.finish("branch seams");
}

/// `ln_cdf` and `ln_sf` must agree with the logarithms of their linear forms
/// wherever those are representable.
#[test]
fn the_log_forms_agree_with_their_linear_forms() {
    use bayes_kernels as k;
    let mut f = Failures::default();
    // Five families carry a separate `ln_cdf` kernel; each is checked against
    // its own `cdf` rather than against a shared default.
    /// A named family, its log-CDF kernel, and its linear CDF kernel.
    type LogPair = (
        &'static str,
        Box<dyn Fn(f64) -> f64>,
        Box<dyn Fn(f64) -> f64>,
    );
    let pairs: Vec<LogPair> = vec![
        (
            "exponential(2.5)",
            Box::new(|x| k::exponential_ln_cdf(x, 2.5)),
            Box::new(|x| k::exponential_cdf(x, 2.5)),
        ),
        (
            "normal(0,1)",
            Box::new(|x| k::normal_ln_cdf(x, 0.0, 1.0)),
            Box::new(|x| k::normal_cdf(x, 0.0, 1.0)),
        ),
        (
            "logistic(2,0.8)",
            Box::new(|x| k::logistic_ln_cdf(x, 2.0, 0.8)),
            Box::new(|x| k::logistic_cdf(x, 2.0, 0.8)),
        ),
        (
            "gumbel(0,1)",
            Box::new(|x| k::gumbel_ln_cdf(x, 0.0, 1.0)),
            Box::new(|x| k::gumbel_cdf(x, 0.0, 1.0)),
        ),
        (
            "frechet(2,3)",
            Box::new(|x| k::frechet_ln_cdf(x, 3.0, 2.0)),
            Box::new(|x| k::frechet_cdf(x, 3.0, 2.0)),
        ),
    ];
    for (name, log_form, linear) in pairs {
        for i in -40..=40 {
            let x = f64::from(i) / 4.0;
            let (lc, c) = (log_form(x), linear(x));
            if c <= 1e-290 || c >= 1.0 {
                continue;
            }
            // The **reference** is the blunt side here. Where the CDF is near
            // 1, `c` is known to one ulp, so `c.ln()` carries an absolute error
            // of about `eps` and a relative error of `eps / |ln c|` — at
            // `ln_cdf = -2.5e-8` that is `9e-9`. Holding the accurate log form
            // to a tighter bound than its own reference can express would be
            // measuring the instrument, so the bound is that limit.
            let bound = (4.0 * f64::EPSILON / lc.abs().max(f64::MIN_POSITIVE)).max(1e-9);
            f.check(rel(lc, c.ln()) <= bound, || {
                format!(
                    "{name}: ln_cdf({x}) = {lc:.14}, ln(cdf) = {:.14}, bound {bound:.2e}",
                    c.ln()
                )
            });
        }
    }
    // And `ln_sf` against `ln(1 - cdf)` in the body, where the subtraction is
    // still exact — the one region the two must coincide.
    for item in cases() {
        for p in [0.1, 0.3, 0.5, 0.7] {
            let x = bisect_cdf(&item.cdf, item.lo, p);
            if !x.is_finite() {
                continue;
            }
            let direct = (1.0 - (item.cdf)(x)).ln();
            f.check(rel((item.ln_sf)(x), direct) < 1e-9, || {
                format!(
                    "{}: ln_sf at the {p} quantile is {:.12}, ln(1 - cdf) is {direct:.12}",
                    item.name,
                    (item.ln_sf)(x)
                )
            });
        }
    }
    f.finish("log / linear agreement");
}

/// Find `x` with `cdf(x) ≈ target` by bisection, so a check can reach a body
/// point for a family with no quantile kernel.
fn bisect_cdf(cdf: &dyn Fn(f64) -> f64, lo: f64, target: f64) -> f64 {
    let mut low = if lo.is_finite() { lo } else { -1e6 };
    let mut high = low + 1.0;
    for _ in 0..200 {
        if cdf(high) >= target {
            break;
        }
        high = low + (high - low) * 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if cdf(mid) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}
