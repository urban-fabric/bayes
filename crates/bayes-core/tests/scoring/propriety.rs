//! Is honesty optimal?

use super::{expected_loss, losses};
use bayes_utils::Failures;

/// True distributions to forecast. Strictly positive, so no category's
/// contribution to an expectation is `0 · inf`; the zero case is tested on its
/// own below.
fn truths() -> Vec<Vec<f64>> {
    vec![
        vec![0.5, 0.5],
        vec![0.9, 0.1],
        vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        vec![0.6, 0.3, 0.1],
        vec![0.05, 0.15, 0.6, 0.2],
        vec![0.4, 0.1, 0.1, 0.1, 0.3],
    ]
}

/// Deterministic perturbations of `q`: move mass `epsilon` from one category to
/// another, and separately pull `q` toward the uniform distribution.
///
/// No RNG, so a failure names the exact forecast that beat the truth. The
/// transfers are the sharp ones — a rule with a mis-ordered cumulative sum
/// fails on an *adjacent* transfer while surviving a symmetric mixture.
fn perturbations(q: &[f64]) -> Vec<(String, Vec<f64>)> {
    let k = q.len();
    let mut out = Vec::new();
    for epsilon in [1e-3, 0.01, 0.1] {
        for from in 0..k {
            for to in 0..k {
                if from == to || q[from] <= epsilon {
                    continue;
                }
                let mut p = q.to_vec();
                p[from] -= epsilon;
                p[to] += epsilon;
                out.push((format!("move {epsilon} from {from} to {to}"), p));
            }
        }
        // And a mixture toward uniform, which no transfer reproduces.
        let uniform = 1.0 / k as f64;
        let mixed: Vec<f64> = q
            .iter()
            .map(|&qi| (1.0 - epsilon) * qi + epsilon * uniform)
            .collect();
        out.push((format!("mix {epsilon} toward uniform"), mixed));
    }
    // Drop anything that is not actually a different forecast. Mixing a
    // *uniform* truth toward uniform returns the truth itself, which is not a
    // lie — and a strictly proper rule is quite right to score it identically.
    // Left in, it produced 24 failures that were all this one mistake.
    out.retain(|(_, p)| {
        p.iter()
            .zip(q)
            .any(|(pi, qi)| (pi - qi).abs() > 16.0 * f64::EPSILON)
    });
    out
}

/// The expected loss is **strictly** smallest at the truth.
///
/// This is the whole content of "proper". The margin is derived rather than
/// chosen: a strictly proper rule's expected loss is smooth with a minimum at
/// `q`, so the gap for a transfer of size `ε` grows like `ε²` — the first-order
/// term vanishes at a minimum. Requiring the gap to exceed `0` outright would
/// be testing floating-point noise at `ε = 1e-3`, so the bound is a small
/// fraction of that quadratic scale.
#[test]
fn the_expected_loss_is_minimized_by_telling_the_truth() {
    let mut f = Failures::default();
    for (name, loss) in losses() {
        for q in truths() {
            let honest = expected_loss(loss, &q, &q);
            f.check(honest.is_finite(), || {
                format!("{name}: the honest expected loss for {q:?} is {honest}")
            });
            for (description, p) in perturbations(&q) {
                let lying = expected_loss(loss, &p, &q);
                // The quadratic scale of this perturbation.
                let shift: f64 = p.iter().zip(&q).map(|(pi, qi)| (pi - qi) * (pi - qi)).sum();
                let margin = 1e-6 * shift;
                f.check(lying > honest + margin, || {
                    format!(
                        "{name}: for truth {q:?}, {description} scored {lying:.14} against the \
                         honest {honest:.14} — a proper rule cannot be beaten by a lie"
                    )
                });
            }
        }
    }
    f.finish("propriety");
}

/// A category the truth never produces may be forecast at zero without
/// penalty, and the expected loss must stay finite.
///
/// This is where `0 · inf` lives. `log_loss` is `+inf` on a zero-probability
/// observation, so an implementation that multiplied rather than skipped would
/// return `NaN` for a perfectly sensible forecast — and `NaN` compares false
/// against everything, so the propriety check above would silently pass.
#[test]
fn a_category_with_no_mass_costs_nothing_to_rule_out() {
    let mut f = Failures::default();
    // The truth never produces category 2.
    let q = vec![0.7, 0.3, 0.0];
    for (name, loss) in losses() {
        let honest = expected_loss(loss, &q, &q);
        f.check(honest.is_finite(), || {
            format!("{name}: expected loss against a truth with a zero category is {honest}")
        });
        // Spending mass on the impossible category must cost something.
        let wasteful = vec![0.6, 0.3, 0.1];
        let wasteful_loss = expected_loss(loss, &wasteful, &q);
        f.check(wasteful_loss > honest, || {
            format!(
                "{name}: forecasting an impossible category scored {wasteful_loss:.14}, no \
                 worse than the honest {honest:.14}"
            )
        });
    }
    f.finish("zero-probability categories");
}

/// A perfect forecast attains each rule's optimum exactly.
#[test]
fn a_correct_point_mass_attains_the_optimum() {
    use bayes_core::{brier_score, log_loss, ranked_probability_score, spherical_score};
    let mut f = Failures::default();
    for k in 1..=5 {
        for observed in 0..k {
            let mut p = vec![0.0; k];
            p[observed] = 1.0;
            f.close(
                || format!("log_loss, k={k}, observed={observed}"),
                log_loss(&p, observed),
                0.0,
                0.0,
            );
            f.close(
                || format!("brier_score, k={k}, observed={observed}"),
                brier_score(&p, observed),
                0.0,
                0.0,
            );
            f.close(
                || format!("ranked_probability_score, k={k}, observed={observed}"),
                ranked_probability_score(&p, observed),
                0.0,
                0.0,
            );
            f.close(
                || format!("spherical_score, k={k}, observed={observed}"),
                spherical_score(&p, observed),
                1.0,
                1e-15,
            );
        }
    }
    f.finish("perfect forecasts");
}

/// A confidently wrong point mass is maximally penalized.
///
/// The counterpart of the test above, and the reason `log_loss` returns `+inf`
/// rather than a large finite number: asserting certainty in something that
/// then fails to happen is not a bounded error.
#[test]
fn a_confidently_wrong_point_mass_is_maximally_penalized() {
    use bayes_core::{brier_score, log_loss, spherical_score};
    let mut f = Failures::default();
    let mut p = vec![0.0; 4];
    p[0] = 1.0;
    f.check(log_loss(&p, 2) == f64::INFINITY, || {
        format!(
            "log_loss of a wrong certainty is {}, expected +inf",
            log_loss(&p, 2)
        )
    });
    // Brier's worst multi-class value is 2: one unit of error at the forecast
    // category and one at the observed one.
    f.close(
        || "brier of a wrong certainty".to_string(),
        brier_score(&p, 2),
        2.0,
        1e-15,
    );
    f.close(
        || "spherical of a wrong certainty".to_string(),
        spherical_score(&p, 2),
        0.0,
        1e-15,
    );
    f.finish("confident errors");
}
