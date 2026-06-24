use euclid::numeric::Real;

use super::gamma::{gamma_cdf, gamma_cdf_on_support, gamma_entropy, gamma_log_pdf_on_support};

/// Chi-squared log-density on `x > 0`.
pub fn chi_squared_log_pdf_on_support<T: Real>(
    x: T,
    half_freedom: T,
    ln_gamma_half_freedom: T,
) -> T {
    gamma_log_pdf_on_support(x, half_freedom, T::half(), ln_gamma_half_freedom)
}

/// Chi-squared CDF on `x > 0`.
pub fn chi_squared_cdf_on_support<T: Real>(x: T, half_freedom: T) -> T {
    gamma_cdf_on_support(x, half_freedom, T::half())
}

/// Chi-squared CDF with the below-support floor folded in.
pub fn chi_squared_cdf<T: Real>(x: T, half_freedom: T) -> T {
    gamma_cdf(x, half_freedom, T::half())
}

/// Chi-squared skewness `sqrt(8 / freedom)`.
pub fn chi_squared_skewness<T: Real>(freedom: T) -> T {
    (T::scalar(8.0) / freedom).sqrt()
}

/// Chi-squared excess kurtosis `12 / freedom`.
pub fn chi_squared_excess_kurtosis<T: Real>(freedom: T) -> T {
    T::scalar(12.0) / freedom
}

/// Chi-squared entropy.
pub fn chi_squared_entropy<T: Real>(half_freedom: T) -> T {
    gamma_entropy(half_freedom, T::half())
}
