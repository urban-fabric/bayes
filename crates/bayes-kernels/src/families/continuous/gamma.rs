use crate::families::common::gamma_bc;
use euclid::numeric::Real;

/// Gamma log-density on `x > 0`.
pub fn gamma_log_pdf_on_support<T: Real>(x: T, shape: T, rate: T, ln_gamma_shape: T) -> T {
    let one = T::one();
    shape.clone() * rate.clone().ln() - ln_gamma_shape + (shape - one) * x.clone().ln() - rate * x
}

/// Gamma CDF on `x > 0`.
pub fn gamma_cdf_on_support<T: Real>(x: T, shape: T, rate: T) -> T {
    (rate * x).gamma_lr(shape)
}

/// Gamma CDF with the below-support floor folded in.
pub fn gamma_cdf<T: Real>(x: T, shape: T, rate: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    gamma_cdf_on_support(safe_x, shape, rate).mask_invalid(in_support, T::zero())
}

/// Gamma variance `shape / rate^2`.
pub fn gamma_variance<T: Real>(shape: T, rate: T) -> T {
    shape / (rate.clone() * rate)
}

/// Gamma skewness `2 / sqrt(shape)`.
pub fn gamma_skewness<T: Real>(shape: T) -> T {
    T::two() / shape.sqrt()
}

/// Gamma excess kurtosis `6 / shape`.
pub fn gamma_excess_kurtosis<T: Real>(shape: T) -> T {
    T::scalar(6.0) / shape
}

/// Gamma differential entropy.
pub fn gamma_entropy<T: Real>(shape: T, rate: T) -> T {
    let one = T::one();
    shape.clone() - rate.ln() + shape.clone().ln_gamma() + (one - shape.clone()) * shape.digamma()
}

/// Gamma KL divergence.
pub fn kl_gamma<T: Real>(shape_self: T, rate_self: T, shape_other: T, rate_other: T) -> T {
    (shape_self.clone() - shape_other.clone()) * shape_self.clone().digamma()
        - shape_self.clone().ln_gamma()
        + shape_other.clone().ln_gamma()
        + shape_other * (rate_self.clone().ln() - rate_other.clone().ln())
        + shape_self * (rate_other - rate_self.clone()) / rate_self
}

/// Gamma squared Hellinger distance.
pub fn hellinger_sq_gamma<T: Real>(
    shape_self: T,
    rate_self: T,
    shape_other: T,
    rate_other: T,
) -> T {
    crate::hellinger_sq_from_bc(gamma_bc(shape_self, rate_self, shape_other, rate_other))
}

/// Gamma chi-squared divergence interior.
pub fn chi_squared_gamma_on_support<T: Real>(
    shape_self: T,
    rate_self: T,
    shape_other: T,
    rate_other: T,
) -> T {
    let (a0, b0, a1, b1) = (shape_self, rate_self, shape_other, rate_other);
    let a = T::two() * a0.clone() - a1.clone();
    let b = T::two() * b0.clone() - b1.clone();
    let ln_term = T::two() * a0.clone() * b0.clone().ln() - T::two() * a0.ln_gamma()
        + a1.clone().ln_gamma()
        - a1 * b1.ln()
        + a.clone().ln_gamma()
        - a * b.ln();
    ln_term.exp() - T::one()
}

/// Gamma Renyi divergence interior.
pub fn renyi_gamma_on_support<T: Real>(
    shape_self: T,
    rate_self: T,
    shape_other: T,
    rate_other: T,
    alpha: T,
) -> T {
    let (a0, b0, a1, b1) = (shape_self, rate_self, shape_other, rate_other);
    let one_minus_alpha = T::one() - alpha.clone();
    let a_alpha = alpha.clone() * a0.clone() + one_minus_alpha.clone() * a1.clone();
    let b_alpha = alpha.clone() * b0.clone() + one_minus_alpha.clone() * b1.clone();
    let num = alpha.clone() * (a0.clone() * b0.ln() - a0.ln_gamma())
        + one_minus_alpha * (a1.clone() * b1.ln() - a1.ln_gamma())
        + a_alpha.clone().ln_gamma()
        - a_alpha * b_alpha.ln();
    num / (alpha - T::one())
}

/// Gamma-Poisson posterior-predictive log-mass.
pub fn gamma_poisson_log_predictive<T: Real>(count: T, shape: T, rate: T, exposure: T) -> T {
    (shape.clone() + count.clone()).ln_gamma()
        - shape.clone().ln_gamma()
        - (count.clone() + T::one()).ln_gamma()
        + shape.clone() * rate.clone().ln()
        + count.clone() * exposure.clone().ln()
        - (shape + count) * (rate + exposure).ln()
}

/// Gamma-Exponential posterior-predictive log-density.
pub fn gamma_exponential_log_predictive<T: Real>(x: T, shape: T, rate: T) -> T {
    shape.clone().ln() + shape.clone() * rate.clone().ln() - (shape + T::one()) * (rate + x).ln()
}

/// Gamma-Poisson log marginal likelihood.
pub fn gamma_poisson_log_marginal_likelihood<T: Real>(
    shape: T,
    rate: T,
    count_sum: T,
    count_n: T,
    ln_factorial_sum: T,
) -> T {
    shape.clone() * rate.clone().ln() - shape.clone().ln_gamma() - ln_factorial_sum
        + (shape.clone() + count_sum.clone()).ln_gamma()
        - (shape + count_sum) * (rate + count_n).ln()
}

/// Gamma-Exponential log marginal likelihood.
pub fn gamma_exponential_log_marginal_likelihood<T: Real>(
    shape: T,
    rate: T,
    count_n: T,
    sum: T,
) -> T {
    shape.clone() * rate.clone().ln() - shape.clone().ln_gamma()
        + (shape.clone() + count_n.clone()).ln_gamma()
        - (shape + count_n) * (rate + sum).ln()
}
