use super::*;

// --- analytic higher moments (skewness, excess kurtosis, non-trivial mean/var) ---
//
// The closed-form moments that are genuine functions of the shape parameters —
// pure algebra. (The constant ones — Normal skew 0, exponential skew 2 — need no
// kernel; the trivial location/scale means are the parameters themselves.)

/// Log-normal mean `exp(µ + σ²/2)`.
pub fn lognormal_mean<T: LazyReal>(mean_log: T, std_log: T) -> T {
    (mean_log + std_log.clone() * std_log * T::half()).exp()
}

/// Log-normal variance `(exp(σ²) − 1)·exp(2µ + σ²)`.
pub fn lognormal_variance<T: LazyReal>(mean_log: T, std_log: T) -> T {
    let var_log = std_log.clone() * std_log;
    (var_log.clone().exp() - T::one()) * (T::two() * mean_log + var_log).exp()
}

/// Gamma variance `shape / rate²`.
pub fn gamma_variance<T: LazyReal>(shape: T, rate: T) -> T {
    shape / (rate.clone() * rate)
}

/// Beta variance `αβ / ((α+β)²·(α+β+1))`.
pub fn beta_variance<T: LazyReal>(alpha: T, beta: T) -> T {
    let sum = alpha.clone() + beta.clone();
    (alpha * beta) / (sum.clone() * sum.clone() * (sum + T::one()))
}

/// Gamma skewness `2/√shape`.
pub fn gamma_skewness<T: LazyReal>(shape: T) -> T {
    T::two() / shape.sqrt()
}

/// Gamma excess kurtosis `6/shape`.
pub fn gamma_excess_kurtosis<T: LazyReal>(shape: T) -> T {
    T::scalar(6.0) / shape
}

/// Beta skewness `2(β−α)·√(α+β+1) / ((α+β+2)·√(αβ))`.
pub fn beta_skewness<T: LazyReal>(alpha: T, beta: T) -> T {
    let sum = alpha.clone() + beta.clone();
    T::two() * (beta.clone() - alpha.clone()) * (sum.clone() + T::one()).sqrt()
        / ((sum + T::two()) * (alpha * beta).sqrt())
}

/// Chi-squared skewness `√(8/freedom)`.
pub fn chi_squared_skewness<T: LazyReal>(freedom: T) -> T {
    (T::scalar(8.0) / freedom).sqrt()
}

/// Chi-squared excess kurtosis `12/freedom`.
pub fn chi_squared_excess_kurtosis<T: LazyReal>(freedom: T) -> T {
    T::scalar(12.0) / freedom
}

/// Wald / inverse-Gaussian skewness `3√(µ/λ)`.
pub fn wald_skewness<T: LazyReal>(mean: T, shape: T) -> T {
    T::scalar(3.0) * (mean / shape).sqrt()
}

/// Wald / inverse-Gaussian excess kurtosis `15µ/λ`.
pub fn wald_excess_kurtosis<T: LazyReal>(mean: T, shape: T) -> T {
    T::scalar(15.0) * mean / shape
}

/// Poisson skewness `1/√λ`.
pub fn poisson_skewness<T: LazyReal>(rate: T) -> T {
    T::one() / rate.sqrt()
}

/// Poisson excess kurtosis `1/λ`.
pub fn poisson_excess_kurtosis<T: LazyReal>(rate: T) -> T {
    T::one() / rate
}

/// Bernoulli skewness `(q − p)/√(p·q)`, `q = 1 − p`.
pub fn bernoulli_skewness<T: LazyReal>(p: T) -> T {
    let q = T::one() - p.clone();
    (q.clone() - p.clone()) / (p * q).sqrt()
}

/// Binomial skewness `(1 − 2p)/√(n·p·(1−p))`.
pub fn binomial_skewness<T: LazyReal>(trials: T, p: T) -> T {
    let variance = trials * p.clone() * (T::one() - p.clone());
    (T::one() - T::two() * p) / variance.sqrt()
}

/// Geometric skewness `(2 − p)/√(1 − p)`.
pub fn geometric_skewness<T: LazyReal>(p: T) -> T {
    (T::two() - p.clone()) / (T::one() - p).sqrt()
}

/// Negative-binomial skewness `(2 − p)/√(r·(1 − p))`.
pub fn negative_binomial_skewness<T: LazyReal>(successes: T, p: T) -> T {
    (T::two() - p.clone()) / (successes * (T::one() - p)).sqrt()
}
