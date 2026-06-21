use super::*;

// --- closed-form differential / Shannon entropies ---
//
// Pure algebra for most families; the gamma/beta-shaped ones reach for the lazy
// `ln_gamma`/`digamma` rather than a caller-supplied normalizer. All are branch-free
// and lower unchanged to a tensor backend.

/// Normal entropy `½·ln(2π·e·σ²)`.
pub fn normal_entropy<T: LazyReal>(std_dev: T) -> T {
    T::half() * (T::tau() * T::e() * std_dev.clone() * std_dev).ln()
}

/// Log-normal entropy `µ_log + ½·ln(2π·e·σ_log²)`.
pub fn lognormal_entropy<T: LazyReal>(mean_log: T, std_log: T) -> T {
    mean_log + T::half() * (T::tau() * T::e() * std_log.clone() * std_log).ln()
}

/// Exponential entropy `1 − ln(rate)`.
pub fn exponential_entropy<T: LazyReal>(rate: T) -> T {
    T::one() - rate.ln()
}

/// Laplace entropy `ln(2·e·scale)`.
pub fn laplace_entropy<T: LazyReal>(scale: T) -> T {
    (T::two() * T::e() * scale).ln()
}

/// Logistic entropy `ln(scale) + 2`.
pub fn logistic_entropy<T: LazyReal>(scale: T) -> T {
    scale.ln() + T::two()
}

/// Uniform entropy `ln(width)`.
pub fn uniform_entropy<T: LazyReal>(width: T) -> T {
    width.ln()
}

/// Discrete-uniform entropy `ln(count)`.
pub fn discrete_uniform_entropy<T: LazyReal>(count: T) -> T {
    count.ln()
}

/// Cauchy entropy `ln(4π·scale)`.
pub fn cauchy_entropy<T: LazyReal>(scale: T) -> T {
    (T::scalar(4.0) * T::pi() * scale).ln()
}

/// Gumbel entropy `ln(scale) + γ + 1`.
pub fn gumbel_entropy<T: LazyReal>(scale: T) -> T {
    scale.ln() + T::euler_gamma() + T::one()
}

/// Rayleigh entropy `1 + ln(scale/√2) + γ/2`.
pub fn rayleigh_entropy<T: LazyReal>(scale: T) -> T {
    T::one() + (scale / T::sqrt_2()).ln() + T::euler_gamma() * T::half()
}

/// Weibull entropy `γ·(1 − 1/k) + ln(scale/k) + 1`.
pub fn weibull_entropy<T: LazyReal>(scale: T, shape: T) -> T {
    T::euler_gamma() * (T::one() - T::one() / shape.clone()) + (scale / shape).ln() + T::one()
}

/// Pareto entropy `ln(minimum/shape) + 1/shape + 1`.
pub fn pareto_entropy<T: LazyReal>(minimum: T, shape: T) -> T {
    (minimum / shape.clone()).ln() + T::one() / shape + T::one()
}

/// Triangular entropy `½ + ln((high − low)/2)`.
pub fn triangular_entropy<T: LazyReal>(low: T, high: T) -> T {
    T::half() + ((high - low) / T::two()).ln()
}

/// Half-normal entropy `½·ln(π·scale²/2) + ½`.
pub fn half_normal_entropy<T: LazyReal>(scale: T) -> T {
    T::half() * (T::pi() * scale.clone() * scale / T::two()).ln() + T::half()
}

/// Gamma differential entropy:
/// `shape − ln(rate) + ln Γ(shape) + (1 − shape)·ψ(shape)`.
///
/// Computed directly from the lazy `ln_gamma`/`digamma`, with no caller-supplied
/// normalizer.
pub fn gamma_entropy<T: LazyReal>(shape: T, rate: T) -> T {
    let one = T::one();
    shape.clone() - rate.ln() + shape.clone().ln_gamma() + (one - shape.clone()) * shape.digamma()
}

/// Chi-squared entropy — the gamma entropy at `shape = freedom/2`, `rate = ½`.
pub fn chi_squared_entropy<T: LazyReal>(half_freedom: T) -> T {
    gamma_entropy(half_freedom, T::half())
}

/// Inverse-gamma entropy:
/// `shape + ln(scale) + ln Γ(shape) − (1 + shape)·ψ(shape)`.
pub fn inverse_gamma_entropy<T: LazyReal>(shape: T, scale: T) -> T {
    shape.clone() + scale.ln() + shape.clone().ln_gamma()
        - (T::one() + shape.clone()) * shape.digamma()
}

/// Beta entropy:
/// `ln B(α,β) − (α−1)·ψ(α) − (β−1)·ψ(β) + (α+β−2)·ψ(α+β)`.
pub fn beta_entropy<T: LazyReal>(alpha: T, beta: T) -> T {
    let one = T::one();
    let sum = alpha.clone() + beta.clone();
    let ln_beta = alpha.clone().ln_gamma() + beta.clone().ln_gamma() - sum.clone().ln_gamma();
    ln_beta
        - (alpha.clone() - one.clone()) * alpha.digamma()
        - (beta.clone() - one) * beta.digamma()
        + (sum.clone() - T::two()) * sum.digamma()
}

/// Student-t entropy (location/scale):
/// `ln(scale) + ½·ln(νπ) + ln Γ(ν/2) − ln Γ((ν+1)/2) + (ν+1)/2·(ψ((ν+1)/2) − ψ(ν/2))`.
pub fn students_t_entropy<T: LazyReal>(scale: T, freedom: T) -> T {
    let half_nu = freedom.clone() * T::half();
    let half_nu_plus_one = (freedom.clone() + T::one()) * T::half();
    scale.ln() + T::half() * (freedom * T::pi()).ln() + half_nu.clone().ln_gamma()
        - half_nu_plus_one.clone().ln_gamma()
        + half_nu_plus_one.clone() * (half_nu_plus_one.digamma() - half_nu.digamma())
}

/// von Mises entropy: `ln(2π) + ln I₀(κ) − κ·A(κ)`, `A = I₁/I₀` the mean resultant
/// length — over the scaled Bessel surface (`ln I₀ = ln I₀ᵉ + |κ|`, `A = I₁ᵉ/I₀ᵉ`).
pub fn von_mises_entropy<T: LazyReal>(kappa: T) -> T {
    let resultant = kappa.clone().bessel_i1e() / kappa.clone().bessel_i0e();
    T::ln_tau() + ln_i0(kappa.clone()) - kappa * resultant
}
