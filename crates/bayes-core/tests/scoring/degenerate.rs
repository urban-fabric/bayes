//! Is a malformed forecast refused rather than scored?
//!
//! Every eager rule validates its input and returns `NaN` for anything that is
//! not a probability vector. The alternative — scoring it anyway — is worse
//! than an error, because a score is a number a caller will go on to average.

use bayes_core::{
    brier_score, is_probability_vector, log_loss, log_score, ranked_probability_score,
    spherical_score,
};
use bayes_utils::Failures;

/// Everything that is not a probability vector, and why.
fn malformed() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("empty", vec![]),
        ("sums below one", vec![0.3, 0.3]),
        ("sums above one", vec![0.7, 0.7]),
        ("contains a negative", vec![-0.2, 1.2]),
        ("contains a NaN", vec![f64::NAN, 1.0]),
        ("contains an infinity", vec![f64::INFINITY, 0.0]),
        ("contains a negative infinity", vec![f64::NEG_INFINITY, 1.0]),
        ("all zero", vec![0.0, 0.0, 0.0]),
    ]
}

/// Every rule returns `NaN` for a malformed forecast.
///
/// The negative case is the one worth having. `[-0.2, 1.2]` sums to 1, so a
/// validator that checked only the sum would accept it — and then the Brier
/// score is a perfectly plausible `0.08` for a forecast that assigns negative
/// probability.
#[test]
fn a_malformed_forecast_is_not_scored() {
    let mut f = Failures::default();
    for (why, p) in malformed() {
        f.check(!is_probability_vector(&p), || {
            format!("{p:?} ({why}) was accepted as a probability vector")
        });
        for (name, score) in [
            ("log_score", log_score as fn(&[f64], usize) -> f64),
            ("log_loss", log_loss),
            ("brier_score", brier_score),
            ("spherical_score", spherical_score),
            ("ranked_probability_score", ranked_probability_score),
        ] {
            let value = score(&p, 0);
            f.check(value.is_nan(), || {
                format!("{name} scored {p:?} ({why}) as {value} instead of NaN")
            });
        }
    }
    f.finish("malformed forecasts");
}

/// An observation outside the forecast's categories is `NaN`, not a panic and
/// not a silent zero.
#[test]
fn an_out_of_range_observation_is_not_scored() {
    let mut f = Failures::default();
    let p = vec![0.25, 0.25, 0.5];
    for observed in [3_usize, 4, 100, usize::MAX] {
        for (name, score) in [
            ("log_score", log_score as fn(&[f64], usize) -> f64),
            ("log_loss", log_loss),
            ("brier_score", brier_score),
            ("spherical_score", spherical_score),
            ("ranked_probability_score", ranked_probability_score),
        ] {
            let value = score(&p, observed);
            f.check(value.is_nan(), || {
                format!("{name} scored out-of-range observation {observed} as {value}")
            });
        }
    }
    // And the last valid index is valid.
    f.check(brier_score(&p, 2).is_finite(), || {
        "the final category should be scorable".to_string()
    });
    f.finish("out-of-range observations");
}

/// `log_score` and `log_loss` are exact negatives of each other, including at
/// the infinities.
///
/// `-(-inf)` is `+inf` and `-NaN` is `NaN`, so this holds at both edges — but
/// only if `log_loss` is defined by negating rather than by recomputing, which
/// is what this pins.
#[test]
fn the_log_score_and_log_loss_are_exact_negatives() {
    let mut f = Failures::default();
    for p in [
        vec![0.5, 0.5],
        vec![1.0, 0.0],
        vec![0.1, 0.9],
        vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ] {
        for observed in 0..p.len() {
            let (score, loss) = (log_score(&p, observed), log_loss(&p, observed));
            if score.is_nan() {
                f.check(loss.is_nan(), || {
                    format!("log_score was NaN for {p:?} at {observed} but log_loss was {loss}")
                });
            } else {
                f.check(loss == -score, || {
                    format!("log_loss {loss} is not the negation of log_score {score}")
                });
            }
        }
    }
    // A zero-probability observation: `-inf` becomes `+inf`.
    let certain = vec![1.0, 0.0];
    f.check(log_score(&certain, 1) == f64::NEG_INFINITY, || {
        format!(
            "log_score of an impossible outcome is {}",
            log_score(&certain, 1)
        )
    });
    f.check(log_loss(&certain, 1) == f64::INFINITY, || {
        format!(
            "log_loss of an impossible outcome is {}",
            log_loss(&certain, 1)
        )
    });
    f.finish("log score / loss duality");
}

/// A forecast one tolerance away from summing to 1 is accepted; one clearly
/// past it is not.
///
/// The boundary matters because a caller renormalizing a long weight vector
/// will land a few ulps off, and refusing that would make the rules unusable —
/// while accepting something that sums to `1.1` would make them meaningless.
#[test]
fn the_sum_tolerance_is_where_it_claims_to_be() {
    use bayes_core::PROBABILITY_TOLERANCE;
    let mut f = Failures::default();
    // Just inside.
    let inside = vec![0.5, 0.5 - PROBABILITY_TOLERANCE / 2.0];
    f.check(is_probability_vector(&inside), || {
        format!("{inside:?} is within tolerance and should be accepted")
    });
    f.check(brier_score(&inside, 0).is_finite(), || {
        "a forecast within tolerance should be scorable".to_string()
    });
    // Clearly outside.
    let outside = vec![0.5, 0.5 - PROBABILITY_TOLERANCE * 100.0];
    f.check(!is_probability_vector(&outside), || {
        format!("{outside:?} is outside tolerance and should be refused")
    });
    f.check(brier_score(&outside, 0).is_nan(), || {
        "a forecast outside tolerance should not be scored".to_string()
    });
    f.finish("sum tolerance");
}
