//! Is the upper tail computed, or subtracted away?

use super::{bounded_above, rel};
use bayes_utils::Failures;
use bayes_utils::families::{ContCase, continuous_cases};

/// Points marching outward from the body, by **powers of ten**.
///
/// Two choices here were both wrong on the first attempt, and both for reasons
/// worth recording.
///
/// The step is the **interquartile range**, not the standard deviation: half
/// the families in this sweep have no finite variance, so a Cauchy or a
/// heavy-tailed Pareto would contribute no points at all.
///
/// And the reach has to be enormous. Doubling 24 times covers an exponential
/// tail easily — `Exponential(1)` is already below `1e-300` after a few dozen
/// IQRs — but it barely dents a polynomial one. A Cauchy's survival is
/// `1/(πz)`, so reaching `ln sf = −40` takes `z ≈ 7e16`, and `Pareto(1, 0.5)`
/// needs `1e300` to get there. Judged by a 24-doubling probe, nine correct
/// heavy-tailed families looked like they had no tail at all.
fn outward(case: &ContCase) -> Vec<f64> {
    let (q1, q3) = (case.dist.inverse_cdf(0.25), case.dist.inverse_cdf(0.75));
    let median = case.dist.median();
    let centre = if median.is_finite() { median } else { q3 };
    let spread = q3 - q1;
    if !centre.is_finite() || !spread.is_finite() || spread <= 0.0 {
        return Vec::new();
    }
    (0..=300)
        .map(|k| centre + spread * 10.0_f64.powi(k))
        .filter(|x| x.is_finite())
        .collect()
}

/// `sf` must equal `exp(ln_sf)`, and must not be zero where that is a
/// representable number.
///
/// This is the invariant the old `1 − cdf` default violated. It is checkable
/// without any external reference because the two are the same quantity by
/// definition, which is what makes it a good adversarial test: it needs no
/// table of true values, and it fails loudly wherever one of the two paths has
/// silently given up.
#[test]
fn the_survival_function_is_computed_rather_than_subtracted() {
    let mut f = Failures::default();
    for case in continuous_cases() {
        if bounded_above(&case.name) {
            continue;
        }
        for x in outward(&case) {
            let (s, ls) = (case.dist.sf(x), case.dist.ln_sf(x));
            f.check(s.is_finite() && (0.0..=1.0).contains(&s), || {
                format!("{}: sf({x:.6e}) = {s} is not a probability", case.name)
            });
            if !ls.is_finite() {
                continue;
            }
            // `exp(ln_sf)` is only a fair target where it does not underflow.
            // `ln(MIN_POSITIVE)` is about −708; staying above −700 keeps the
            // comparison away from the subnormal range.
            if ls > -700.0 {
                let implied = ls.exp();
                f.check(s > 0.0, || {
                    format!(
                        "{}: sf({x:.6e}) is exactly 0, but ln_sf says {ls:.4} — the answer \
                         {implied:.6e} is representable and was thrown away",
                        case.name
                    )
                });
                f.check(rel(s, implied) < 1e-9, || {
                    format!(
                        "{}: sf({x:.6e}) = {s:.6e} but exp(ln_sf) = {implied:.6e}",
                        case.name
                    )
                });
            }
        }
    }
    f.finish("survival is computed, not subtracted");
}

/// Every unbounded family's log-survival must actually reach the far tail.
///
/// Without this, the check above passes vacuously for any family whose `ln_sf`
/// is still the naive `ln(1 − cdf)`: that returns `−inf` out there, the guard
/// skips it, and the gap looks like coverage. Requiring `ln_sf` to get below
/// `−40` somewhere turns a missing override into a named failure.
///
/// Nothing is exempt. `Wald` was, until its `exp(2λ/μ)` factor turned out to
/// cancel exactly against the one inside `Φ(b)` — the exemption had recorded a
/// difficulty, not an impossibility, which is the risk in writing one down.
#[test]
fn every_unbounded_family_has_a_usable_log_survival() {
    const EXEMPT: &[&str] = &[];
    let mut f = Failures::default();
    for case in continuous_cases() {
        if bounded_above(&case.name) || EXEMPT.iter().any(|e| case.name.starts_with(e)) {
            continue;
        }
        let points = outward(&case);
        if points.is_empty() {
            continue;
        }
        let deepest = points
            .iter()
            .map(|&x| case.dist.ln_sf(x))
            .filter(|v| v.is_finite())
            .fold(0.0_f64, f64::min);
        f.check(deepest < -40.0, || {
            format!(
                "{}: ln_sf never gets below {deepest:.3} across 300 decades beyond its IQR, \
                 which means it is still the ln(1 - cdf) default and its tail is unreachable",
                case.name
            )
        });
    }
    f.finish("log-survival reaches the tail");
}

/// The survival must not increase, and `cdf + sf` must still be 1 in the body.
///
/// The second half guards the change that made `sf` derive from `ln_sf`: going
/// through a logarithm costs an ulp or two in the bulk, and this pins how much.
#[test]
fn survival_is_monotone_and_still_complements_the_cdf() {
    let mut f = Failures::default();
    for case in continuous_cases() {
        if bounded_above(&case.name) {
            continue;
        }
        let mut previous = f64::INFINITY;
        for x in outward(&case) {
            let s = case.dist.sf(x);
            f.check(s <= previous + 1e-12, || {
                format!(
                    "{}: sf rose at {x:.6e}: {previous:.6e} -> {s:.6e}",
                    case.name
                )
            });
            previous = s;
        }
        // In the body, where both are `O(1)`, the complement still has to hold
        // to near machine precision.
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = case.dist.inverse_cdf(p);
            if !x.is_finite() {
                continue;
            }
            let total = case.dist.cdf(x) + case.dist.sf(x);
            f.check((total - 1.0).abs() < 1e-12, || {
                format!(
                    "{}: cdf + sf at the {p} quantile is {total}, off by {:.3e}",
                    case.name,
                    (total - 1.0).abs()
                )
            });
        }
    }
    f.finish("survival shape");
}
