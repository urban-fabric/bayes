//! Probability-vector utilities and proper scoring rules.
//!
//! These are closed-form helpers shared by discrete distributions, divergence
//! calculations, channels, and downstream inference layers: finite simplex
//! validation, weight normalization, and forecast scores against an observed
//! category.

use crate::BayesError;
use euclid_numeric::Real;

/// Default tolerance for probability-vector sum checks.
pub const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// Sum a probability-like vector, returning `NaN` if any entry is not finite.
pub fn probability_sum(p: &[f64]) -> f64 {
    if p.iter().any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    p.iter().sum()
}

/// Sum a probability-like vector over any [`Real`] carrier.
///
/// This is the lazy/backend-generic numeric reduction only. It does not perform eager
/// finite/nonnegative/simplex validation because generic comparisons produce masks rather than
/// host booleans.
pub fn probability_sum_real<R: Real>(p: &[R]) -> R {
    p.iter().cloned().fold(R::zero(), |sum, pi| sum + pi)
}

/// Whether `p` is a non-empty probability vector: all entries finite and
/// nonnegative, with sum `1` up to [`PROBABILITY_TOLERANCE`].
pub fn is_probability_vector(p: &[f64]) -> bool {
    is_probability_vector_with_tolerance(p, PROBABILITY_TOLERANCE)
}

/// Whether `p` is a probability vector with a caller-specified sum tolerance.
pub fn is_probability_vector_with_tolerance(p: &[f64], tolerance: f64) -> bool {
    if p.is_empty() || !tolerance.is_finite() || tolerance < 0.0 {
        return false;
    }
    if p.iter().any(|&x| !x.is_finite() || x < 0.0) {
        return false;
    }
    (p.iter().sum::<f64>() - 1.0).abs() <= tolerance
}

/// Validate that `p` is a probability vector.
pub fn validate_probability_vector(p: &[f64]) -> Result<(), BayesError> {
    validate_probability_row(p, PROBABILITY_TOLERANCE)
}

fn validate_probability_row(p: &[f64], tolerance: f64) -> Result<(), BayesError> {
    if p.is_empty() {
        return Err(BayesError::Malformed {
            what: "probability vector must be non-empty",
        });
    }
    for &pi in p {
        if !pi.is_finite() || pi < 0.0 {
            return Err(BayesError::Malformed {
                what: "probabilities must be finite and non-negative",
            });
        }
    }
    if (p.iter().sum::<f64>() - 1.0).abs() > tolerance {
        return Err(BayesError::Malformed {
            what: "probabilities must sum to 1",
        });
    }
    Ok(())
}

/// Whether `rows` is a non-empty row-stochastic kernel: rectangular rows, each
/// row a probability vector.
pub fn is_row_stochastic(rows: &[Vec<f64>]) -> bool {
    is_row_stochastic_with_tolerance(rows, PROBABILITY_TOLERANCE)
}

/// Whether `rows` is row-stochastic with a caller-specified sum tolerance.
pub fn is_row_stochastic_with_tolerance(rows: &[Vec<f64>], tolerance: f64) -> bool {
    validate_row_stochastic_with_tolerance(rows, tolerance).is_ok()
}

/// Validate a row-stochastic kernel represented as rows of probabilities.
pub fn validate_row_stochastic(rows: &[Vec<f64>]) -> Result<(), BayesError> {
    validate_row_stochastic_with_tolerance(rows, PROBABILITY_TOLERANCE)
}

/// Validate a row-stochastic kernel with a caller-specified sum tolerance.
pub fn validate_row_stochastic_with_tolerance(
    rows: &[Vec<f64>],
    tolerance: f64,
) -> Result<(), BayesError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(BayesError::Malformed {
            what: "probability tolerance must be finite and non-negative",
        });
    }
    let outputs = rows.first().map(Vec::len).unwrap_or(0);
    if rows.is_empty() || outputs == 0 {
        return Err(BayesError::Malformed {
            what: "row-stochastic kernel needs at least one row and one column",
        });
    }
    for row in rows {
        if row.len() != outputs {
            return Err(BayesError::Malformed {
                what: "row-stochastic kernel rows must all have the same length",
            });
        }
        validate_probability_row(row, tolerance)?;
    }
    Ok(())
}

/// Normalize nonnegative weights to a probability vector.
///
/// Returns an empty vector for empty, negative, non-finite, all-zero, or
/// overflowed inputs. This is the distribution-layer counterpart of
/// `bayes_statistics::normalized_weights`.
pub fn normalize_probability_weights(weights: &[f64]) -> Vec<f64> {
    if weights.is_empty() || weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Vec::new();
    }
    let sum = weights.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Vec::new();
    }
    weights.iter().map(|&w| w / sum).collect()
}

/// Normalize weights over any [`Real`] carrier.
///
/// This is the lazy/backend-generic numeric transform only. It does not perform
/// eager finite/nonnegative/all-zero validation.
pub fn normalize_probability_weights_real<R: Real>(weights: &[R]) -> Vec<R> {
    if weights.is_empty() {
        return Vec::new();
    }
    let sum = weights.iter().cloned().fold(R::zero(), |acc, w| acc + w);
    weights.iter().cloned().map(|w| w / sum.clone()).collect()
}

/// Normalize each row of a rectangular nonnegative kernel to sum to `1`.
///
/// Returns an empty matrix for malformed, ragged, empty, non-finite, negative,
/// all-zero, or overflowed rows.
pub fn normalize_probability_rows(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let outputs = rows.first().map(Vec::len).unwrap_or(0);
    if rows.is_empty() || outputs == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != outputs {
            return Vec::new();
        }
        let normalized = normalize_probability_weights(row);
        if normalized.len() != outputs {
            return Vec::new();
        }
        out.push(normalized);
    }
    out
}

/// Normalize each row over any [`Real`] carrier.
///
/// This checks only host-visible rectangular shape; numeric validation remains
/// the responsibility of the caller.
pub fn normalize_probability_rows_real<R: Real>(rows: &[Vec<R>]) -> Vec<Vec<R>> {
    let outputs = rows.first().map(Vec::len).unwrap_or(0);
    if rows.is_empty() || outputs == 0 || rows.iter().any(|row| row.len() != outputs) {
        return Vec::new();
    }
    rows.iter()
        .map(|row| normalize_probability_weights_real(row))
        .collect()
}

/// Log score `ln p_observed` for a categorical forecast.
///
/// Higher is better. Returns `-inf` for a zero-probability observed category and
/// `NaN` for malformed forecasts or an out-of-range observation.
pub fn log_score(p: &[f64], observed: usize) -> f64 {
    if !is_probability_vector(p) || observed >= p.len() {
        return f64::NAN;
    }
    log_score_real(p, observed)
}

/// Log score `ln p_observed` over any [`Real`] carrier.
///
/// Returns `NaN` for an out-of-range observation. Probability-vector validation stays in the
/// eager `f64` wrapper.
pub fn log_score_real<R: Real>(p: &[R], observed: usize) -> R {
    p.get(observed).cloned().map_or_else(R::nan, |pi| pi.ln())
}

/// Log loss `-ln p_observed` for a categorical forecast.
///
/// Lower is better. Returns `+inf` for a zero-probability observed category.
pub fn log_loss(p: &[f64], observed: usize) -> f64 {
    if !is_probability_vector(p) || observed >= p.len() {
        return f64::NAN;
    }
    log_loss_real(p, observed)
}

/// Log loss `-ln p_observed` over any [`Real`] carrier.
pub fn log_loss_real<R: Real>(p: &[R], observed: usize) -> R {
    -log_score_real(p, observed)
}

/// Multi-class Brier score `sum_i (p_i - 1{i=observed})^2`.
///
/// Lower is better; a perfect forecast scores `0`.
pub fn brier_score(p: &[f64], observed: usize) -> f64 {
    if !is_probability_vector(p) || observed >= p.len() {
        return f64::NAN;
    }
    brier_score_real(p, observed)
}

/// Multi-class Brier score over any [`Real`] carrier.
///
/// Returns `NaN` for an out-of-range observation. Probability-vector validation stays in the
/// eager `f64` wrapper.
pub fn brier_score_real<R: Real>(p: &[R], observed: usize) -> R {
    if observed >= p.len() {
        return R::nan();
    }
    p.iter()
        .enumerate()
        .map(|(i, pi)| {
            let truth = if i == observed { R::one() } else { R::zero() };
            let delta = pi.clone() - truth;
            delta.clone() * delta
        })
        .fold(R::zero(), |sum, term| sum + term)
}

/// Spherical score `p_observed / sqrt(sum_i p_i^2)`.
///
/// Higher is better. The score is `1` for a correct point-mass forecast and
/// `1/sqrt(k)` for a uniform forecast over `k` categories.
pub fn spherical_score(p: &[f64], observed: usize) -> f64 {
    if !is_probability_vector(p) || observed >= p.len() {
        return f64::NAN;
    }
    spherical_score_real(p, observed)
}

/// Spherical score over any [`Real`] carrier.
///
/// Returns `NaN` for an out-of-range observation. A zero norm is represented by the carrier's
/// ordinary division semantics.
pub fn spherical_score_real<R: Real>(p: &[R], observed: usize) -> R {
    if observed >= p.len() {
        return R::nan();
    }
    let norm = p
        .iter()
        .cloned()
        .map(|pi| pi.clone() * pi)
        .fold(R::zero(), |sum, pi2| sum + pi2)
        .sqrt();
    p[observed].clone() / norm
}

/// Ranked probability score for ordered categories.
///
/// `sum_k (F(k) - 1{observed <= k})^2`, where `F` is the forecast CDF. Lower is
/// better; this rewards near misses less harshly than far misses.
pub fn ranked_probability_score(p: &[f64], observed: usize) -> f64 {
    if !is_probability_vector(p) || observed >= p.len() {
        return f64::NAN;
    }
    ranked_probability_score_real(p, observed)
}

/// Ranked probability score over any [`Real`] carrier.
///
/// Returns `NaN` for an out-of-range observation. Probability-vector validation stays in the
/// eager `f64` wrapper.
pub fn ranked_probability_score_real<R: Real>(p: &[R], observed: usize) -> R {
    if observed >= p.len() {
        return R::nan();
    }
    let mut cdf = R::zero();
    let mut score = R::zero();
    for (k, pi) in p.iter().enumerate() {
        cdf = cdf + pi.clone();
        let truth_cdf = if observed <= k { R::one() } else { R::zero() };
        let delta = cdf.clone() - truth_cdf;
        score = score + delta.clone() * delta;
    }
    score
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::assert_close;

    #[test]
    fn probability_vector_validation() {
        assert!(is_probability_vector(&[0.2, 0.3, 0.5]));
        assert!(is_probability_vector(&[0.2, 0.3, 0.5 + 5e-10]));
        assert!(!is_probability_vector(&[]));
        assert!(!is_probability_vector(&[0.2, -1e-12, 0.8]));
        assert!(!is_probability_vector(&[0.2, 0.3, 0.6]));
        assert!(validate_probability_vector(&[0.2, 0.3, 0.5]).is_ok());
        assert!(validate_probability_vector(&[f64::NAN]).is_err());
    }

    #[test]
    fn probability_weight_normalization() {
        let p = normalize_probability_weights(&[2.0, 6.0, 2.0]);
        assert_close!(p.iter().sum(), 1.0, 1e-12);
        assert_close!(p[1], 0.6, 1e-12);
        assert!(normalize_probability_weights(&[0.0, 0.0]).is_empty());
        assert!(normalize_probability_weights(&[1.0, f64::INFINITY]).is_empty());
    }

    #[test]
    fn row_stochastic_validation_and_normalization() {
        let rows = vec![vec![0.25, 0.75], vec![1.0, 0.0]];
        assert!(is_row_stochastic(&rows));
        assert!(validate_row_stochastic(&rows).is_ok());
        assert!(!is_row_stochastic(&[vec![0.25, 0.75], vec![1.0]]));
        assert!(!is_row_stochastic(&[vec![0.25, 0.8]]));

        let normalized = normalize_probability_rows(&[vec![2.0, 8.0], vec![3.0, 3.0]]);
        assert_close!(normalized[0][1], 0.8, 1e-12);
        assert_close!(normalized[1][0], 0.5, 1e-12);
        assert!(normalize_probability_rows(&[vec![0.0, 0.0]]).is_empty());
    }

    #[test]
    fn proper_scores() {
        let p = [0.2, 0.5, 0.3];
        assert_close!(log_score(&p, 1), 0.5_f64.ln(), 1e-12);
        assert_close!(log_loss(&p, 1), -0.5_f64.ln(), 1e-12);
        assert_close!(brier_score(&p, 1), 0.04 + 0.25 + 0.09, 1e-12);
        assert_close!(
            spherical_score(&p, 1),
            0.5 / (0.04_f64 + 0.25 + 0.09).sqrt(),
            1e-12,
        );
        assert_close!(ranked_probability_score(&p, 1), 0.04 + 0.09, 1e-12);
        assert_eq!(log_loss(&[1.0, 0.0], 1), f64::INFINITY);
        assert!(brier_score(&[0.5, 0.6], 1).is_nan());
        assert!(log_score(&[0.5], 2).is_nan());
        assert!(ranked_probability_score(&[0.5, -0.5, 1.0], 0).is_nan());
    }
}
