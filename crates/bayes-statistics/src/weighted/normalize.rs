//! Weight normalization and the effective sample size.
//!
//! [`normalized_log_weights`] is the one that matters for importance sampling:
//! raw importance weights span hundreds of orders of magnitude, so normalizing
//! them in the linear domain underflows the small ones to zero and loses the
//! tail entirely. Normalizing through a log-sum-exp keeps every weight.
//!
//! [`effective_sample_size`] is the diagnostic that says whether the weights are
//! usable at all: equal weights give exactly `n`, and all the mass on one point
//! gives exactly 1.
use euclid_numeric::Real;

pub(super) fn valid(data: &[f64], weights: &[f64]) -> bool {
    data.len() == weights.len()
        && !data.is_empty()
        && data.iter().all(|x| x.is_finite())
        && weights.iter().all(|w| w.is_finite() && *w >= 0.0)
        && weights.iter().sum::<f64>() > 0.0
}

fn log_sum_exp(xs: &[f64]) -> Option<f64> {
    let (&first, rest) = xs.split_first()?;
    let max = rest.iter().fold(first, |acc, &x| acc.max(x));
    let shifted_sum = xs.iter().map(|&x| (x - max).exp()).sum::<f64>();
    Some(max + shifted_sum.ln())
}

fn softmax(xs: &[f64]) -> Option<Vec<f64>> {
    let log_z = log_sum_exp(xs)?;
    Some(xs.iter().map(|&x| (x - log_z).exp()).collect())
}

/// Sum of weights. `NaN` for malformed input.
pub fn weight_sum(weights: &[f64]) -> f64 {
    if weights.is_empty() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return f64::NAN;
    }
    let sum = weights.iter().sum::<f64>();
    if sum > 0.0 { sum } else { f64::NAN }
}

/// Sum of weights over any [`Real`] carrier.
pub fn weight_sum_real<R: Real>(weights: &[R]) -> R {
    if weights.is_empty() {
        return R::nan();
    }
    weights.iter().cloned().fold(R::zero(), |acc, w| acc + w)
}

/// Normalize nonnegative raw weights to a probability vector.
///
/// This is the raw-weight counterpart of riemann `normWeights_isDistribution`:
/// when the return value is non-empty, its entries are nonnegative and sum to
/// one up to floating-point roundoff. Returns an empty vector for malformed,
/// all-zero, or overflowed inputs.
pub fn normalized_weights(weights: &[f64]) -> Vec<f64> {
    let sum = weight_sum(weights);
    if !sum.is_finite() {
        return Vec::new();
    }
    weights.iter().map(|&w| w / sum).collect()
}

/// Normalize raw weights over any [`Real`] carrier.
pub fn normalized_weights_real<R: Real>(weights: &[R]) -> Vec<R> {
    if weights.is_empty() {
        return Vec::new();
    }
    let sum = weight_sum_real(weights);
    weights.iter().cloned().map(|w| w / sum.clone()).collect()
}

/// Normalize log-weights to a probability vector using stable softmax.
///
/// This is the log-domain form used by importance weights, mixture
/// responsibilities, and other "weights known only up to a log constant"
/// calculations. It inherits the shift-invariance of the stable `softmax`, so adding a
/// constant to every log-weight does not change the result. Returns an empty vector on
/// empty input.
pub fn normalized_log_weights(log_weights: &[f64]) -> Vec<f64> {
    // The "empty on non-normalizable input" policy belongs here: a non-normalizable
    // input (empty, all `−∞`, `±∞`, or `NaN`) yields a non-finite vector, which we drop.
    softmax(log_weights)
        .filter(|p| p.iter().all(|x| x.is_finite()))
        .unwrap_or_default()
}

/// Effective sample size `(Σw)^2 / Σw²`.
///
/// For normalized nonnegative weights this equals the Kish ESS
/// `1 / Σᵢ pᵢ²`, and riemann proves the exact interpretation range
/// `1 ≤ ESS ≤ n` (`RiemannLean/Markov/ImportanceSampling.lean`: `one_le_ess`,
/// `ess_le_card`). The formula is scale-invariant, so raw weights may be passed
/// directly.
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    let sum = weight_sum(weights);
    if sum.is_nan() {
        return f64::NAN;
    }
    let sum_sq = weights.iter().map(|w| w * w).sum::<f64>();
    if sum_sq > 0.0 {
        sum * sum / sum_sq
    } else {
        f64::NAN
    }
}

/// Effective sample size `(Σw)^2 / Σw²` over any [`Real`] carrier.
pub fn effective_sample_size_real<R: Real>(weights: &[R]) -> R {
    if weights.is_empty() {
        return R::nan();
    }
    let sum = weight_sum_real(weights);
    let sum_sq = weights
        .iter()
        .cloned()
        .map(|w| w.clone() * w)
        .fold(R::zero(), |acc, w| acc + w);
    sum.clone() * sum / sum_sq
}

/// Whether an ESS value is attainable for `n` nonnegative normalized weights.
///
/// A false result flags a diagnostic outside the riemann-backed range
/// `1 ≤ ESS ≤ n`. The tolerance absorbs floating-point slack at the degenerate
/// (`ESS = 1`) and uniform (`ESS = n`) boundaries.
pub fn is_effective_sample_size_admissible(ess: f64, n: usize) -> bool {
    if n == 0 || !ess.is_finite() {
        return false;
    }
    let upper = n as f64;
    let tol = 1e-9 * upper.max(1.0);
    ess + tol >= 1.0 && ess <= upper + tol
}
