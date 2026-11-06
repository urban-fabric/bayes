use euclid_numeric::Real;

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

/// Chi-squared log-survival: the **upper** regularized incomplete gamma,
/// `ln Q(k/2, x/2)`, rather than the log of one minus the lower one.
///
/// This is the p-value of a chi-squared statistic, so the difference is not
/// academic: computed as `1 − cdf` it is exactly zero for every statistic past
/// about the `1 − 1e-16` point, which for 1 degree of freedom is `x ≈ 70`.
///
/// The branch is not optional. A survival has two ends and they need opposite
/// treatments: where it is **small** the complementary special function is the
/// accurate one, and where it is **near 1** that function returns `1` and
/// throws the deficit away — there `ln_1p(−cdf)` keeps it. Taking only the
/// first route made `cdf + sf` come to `1 + 1e-9` for `FisherSnedecor(1, 1)`
/// just above zero.
/// Takes **half** the degrees of freedom, like every other kernel in this
/// file: the shape of the equivalent gamma. Passing the full count instead
/// silently doubles the degrees of freedom, which is a plausible-looking
/// distribution rather than an error.
pub fn chi_squared_ln_sf<T: Real>(x: T, half_freedom: T) -> T {
    let in_support = x.ge(&T::zero());
    let safe_x = T::select(&in_support, x, T::zero());
    let z = safe_x * T::half();
    let shape = half_freedom;
    let survival = z.clone().gamma_ur(shape.clone());
    let cumulative = z.gamma_lr(shape);
    T::select(
        &survival.gt(&T::half()),
        (-cumulative).ln_1p(),
        survival.ln(),
    )
    .mask_invalid(in_support, T::zero())
}
