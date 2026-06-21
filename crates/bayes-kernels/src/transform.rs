use super::*;

// --- reparameterization transforms (constrained ↔ unconstrained ℝ) ---
//
// The bijections an inference layer differentiates through: a constrained value is
// mapped to ℝ, worked on there, and mapped back, with the density corrected by the
// transform's log-Jacobian. Each is pure analysis on the interior (boundary `±∞`
// is the caller's branch), so the whole reparameterization traces to a backend.

/// Positive support `(0, ∞)` → ℝ: `ln(x)`.
pub fn positive_unconstrain<T: LazyReal>(x: T) -> T {
    x.ln()
}

/// ℝ → positive support: `exp(y)`.
pub fn positive_constrain<T: LazyReal>(y: T) -> T {
    y.exp()
}

/// `ln|d exp(y)/dy| = y`, the positive-support Jacobian correction.
pub fn positive_constrain_log_jacobian<T: LazyReal>(y: T) -> T {
    y
}

/// Unit interval `(0, 1)` → ℝ: the logit `ln(x) − ln(1 − x)`.
pub fn unit_interval_unconstrain<T: LazyReal>(x: T) -> T {
    logit(x)
}

/// ℝ → unit interval: the sigmoid `1/(1 + exp(−y))`.
pub fn unit_interval_constrain<T: LazyReal>(y: T) -> T {
    sigmoid(y)
}

/// `ln|d σ(y)/dy| = log σ(y) + log(1 − σ(y)) = −softplus(−y) − softplus(y)`.
pub fn unit_interval_constrain_log_jacobian<T: LazyReal>(y: T) -> T {
    -(-y.clone()).ln_1p_exp() - y.ln_1p_exp()
}

/// Bounded interval `(lo, hi)` → ℝ: `logit((x − lo)/(hi − lo))`.
pub fn interval_unconstrain<T: LazyReal>(x: T, lo: T, hi: T) -> T {
    logit((x - lo.clone()) / (hi - lo))
}

/// ℝ → bounded interval: `lo + (hi − lo)·σ(y)`.
pub fn interval_constrain<T: LazyReal>(y: T, lo: T, hi: T) -> T {
    lo.clone() + (hi - lo) * sigmoid(y)
}

/// `ln|d constrain(y)/dy| = ln(hi − lo) + log σ(y) + log(1 − σ(y))`.
pub fn interval_constrain_log_jacobian<T: LazyReal>(y: T, lo: T, hi: T) -> T {
    (hi - lo).ln() + (-(-y.clone()).ln_1p_exp() - y.ln_1p_exp())
}
