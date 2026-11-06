//! Do the closed forms agree with algebraically different ones?
//!
//! Propriety is a property of the *shape* of each rule: it holds just as well
//! for a score that is off by a constant factor, or shifted. These checks pin
//! the value.

use bayes_core::{brier_score, log_loss, ranked_probability_score, spherical_score};
use bayes_utils::Failures;

/// Forecasts spanning the useful range of shapes.
fn forecasts() -> Vec<Vec<f64>> {
    vec![
        vec![1.0],
        vec![0.5, 0.5],
        vec![0.2, 0.8],
        vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        vec![0.7, 0.2, 0.1],
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.25, 0.25, 0.25, 0.25],
        vec![0.02, 0.03, 0.9, 0.05],
    ]
}

/// The Brier score expands: `Σ(pᵢ − δᵢ)² = 1 − 2p_obs + Σpᵢ²`.
///
/// Reached by multiplying out the square rather than summing it, so the two
/// share no arithmetic. A wrong sign on the indicator, or an exponent that
/// should be a square, breaks one and not the other.
#[test]
fn the_brier_score_matches_its_expanded_form() {
    let mut f = Failures::default();
    for p in forecasts() {
        let sum_squares: f64 = p.iter().map(|pi| pi * pi).sum();
        for observed in 0..p.len() {
            let expanded = 1.0 - 2.0 * p[observed] + sum_squares;
            f.close(
                || format!("brier {p:?} at {observed}"),
                brier_score(&p, observed),
                expanded,
                1e-14,
            );
        }
    }
    f.finish("Brier expansion");
}

/// Over **two** categories the ranked score is exactly half the Brier score.
///
/// Both reduce to `(p₀ − 1{obs = 0})²` up to a factor: the ranked score's
/// second term vanishes because the forecast CDF has reached 1 by then, while
/// Brier counts the same discrepancy at both categories. A ranked score whose
/// cumulative sum is off by one fails this and survives the propriety check,
/// because shifting the window still leaves a proper rule.
#[test]
fn the_ranked_score_is_half_the_brier_score_over_two_categories() {
    let mut f = Failures::default();
    for p0 in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0] {
        let p = vec![p0, 1.0 - p0];
        for observed in 0..2 {
            f.close(
                || format!("rps vs brier/2 for {p:?} at {observed}"),
                ranked_probability_score(&p, observed),
                brier_score(&p, observed) / 2.0,
                1e-14,
            );
        }
    }
    f.finish("RPS = Brier/2 for k = 2");
}

/// Uniform forecasts have closed-form scores in `k` alone.
///
/// `brier = (k−1)/k`, `spherical = 1/√k`, `log_loss = ln k`. Three independent
/// constants, so a rule that is correct up to a factor of `k` somewhere shows
/// up in exactly one of them.
#[test]
fn a_uniform_forecast_scores_its_closed_form() {
    let mut f = Failures::default();
    for k in 1..=8_usize {
        let kf = k as f64;
        let p = vec![1.0 / kf; k];
        for observed in 0..k {
            f.close(
                || format!("uniform brier, k={k}"),
                brier_score(&p, observed),
                (kf - 1.0) / kf,
                1e-14,
            );
            f.close(
                || format!("uniform spherical, k={k}"),
                spherical_score(&p, observed),
                1.0 / kf.sqrt(),
                1e-14,
            );
            f.close(
                || format!("uniform log_loss, k={k}"),
                log_loss(&p, observed),
                kf.ln(),
                1e-14,
            );
        }
    }
    f.finish("uniform forecasts");
}

/// The ranked score's final term is always zero, and it is **ordered**: a near
/// miss costs less than a far one.
///
/// The second half is the property that distinguishes this rule from Brier,
/// which treats every wrong category alike. For a forecast concentrated at one
/// end, the score must rise monotonically as the observation moves away.
#[test]
fn the_ranked_score_rewards_near_misses() {
    let mut f = Failures::default();
    // Mass concentrated at the first category.
    let p = vec![0.7, 0.2, 0.07, 0.03];
    let mut previous = -1.0;
    for observed in 0..p.len() {
        let score = ranked_probability_score(&p, observed);
        f.check(score > previous, || {
            format!(
                "rps for {p:?} did not rise as the observation moved from {} to {observed}: \
                 {previous:.12} -> {score:.12}",
                observed.saturating_sub(1)
            )
        });
        previous = score;
    }
    // Brier, by contrast, is the same for every wrong category with equal mass.
    let flat = vec![0.4, 0.2, 0.2, 0.2];
    f.close(
        || "brier treats equal-mass categories alike".to_string(),
        brier_score(&flat, 1),
        brier_score(&flat, 3),
        1e-15,
    );
    // While the ranked score does not.
    f.check(
        ranked_probability_score(&flat, 1) < ranked_probability_score(&flat, 3),
        || "rps should penalize the further category more".to_string(),
    );
    f.finish("ordered scoring");
}

/// Every rule is invariant to the *order* of the categories, except the ranked
/// one — which must not be, since order is its whole point.
///
/// A permutation check is a cheap way to catch an index used where a value
/// belongs. Reversing a forecast and the observation together must leave the
/// three unordered rules untouched.
#[test]
fn only_the_ranked_score_depends_on_category_order() {
    let mut f = Failures::default();
    for p in forecasts() {
        let reversed: Vec<f64> = p.iter().rev().copied().collect();
        for observed in 0..p.len() {
            let mirrored = p.len() - 1 - observed;
            f.close(
                || format!("brier under reversal, {p:?} at {observed}"),
                brier_score(&p, observed),
                brier_score(&reversed, mirrored),
                1e-15,
            );
            f.close(
                || format!("spherical under reversal, {p:?} at {observed}"),
                spherical_score(&p, observed),
                spherical_score(&reversed, mirrored),
                1e-15,
            );
            f.close(
                || format!("log_loss under reversal, {p:?} at {observed}"),
                log_loss(&p, observed),
                log_loss(&reversed, mirrored),
                1e-15,
            );
        }
    }
    // And the ranked score genuinely distinguishes an order: a forecast leaning
    // low scores better on a low observation than its mirror image does.
    let leaning = vec![0.7, 0.2, 0.1];
    let mirrored: Vec<f64> = leaning.iter().rev().copied().collect();
    f.check(
        ranked_probability_score(&leaning, 0) < ranked_probability_score(&mirrored, 0),
        || "the ranked score should not be order-blind".to_string(),
    );
    f.finish("order sensitivity");
}
