use euclid_numeric::Real;

pub use super::gamma::{
    gamma_exponential_log_marginal_likelihood, gamma_exponential_log_predictive,
};

/// Exponential log-density on `x >= 0`.
pub fn exponential_log_pdf_on_support<T: Real>(x: T, rate: T) -> T {
    rate.clone().ln() - rate * x
}

/// Exponential log-survival on `x >= 0`.
pub fn exponential_log_survival_on_support<T: Real>(x: T, rate: T) -> T {
    -(rate * x)
}

/// Exponential CDF on `x >= 0`.
/// `−expm1(−t)` rather than `1 − exp(−t)`: for a small `t` the exponential is
/// `1 − t + …`, and subtracting it from 1 throws away every digit of `t` below
/// the ulp of 1. `exponential_cdf(4e-9, 2.5)` came back `1.000000005e-8` where
/// the answer is `9.99999995e-9` — eight digits lost at the bottom of the
/// range, which is exactly where a quantile has to invert it.
pub fn exponential_cdf_on_support<T: Real>(x: T, rate: T) -> T {
    -((-(rate * x)).exp_m1())
}

/// Exponential CDF with the below-support floor folded in.
pub fn exponential_cdf<T: Real>(x: T, rate: T) -> T {
    let in_support = x.ge(&T::zero());
    T::select(&in_support, exponential_cdf_on_support(x, rate), T::zero())
}

/// Exponential log-CDF on `x >= 0`.
pub fn exponential_ln_cdf_on_support<T: Real>(x: T, rate: T) -> T {
    (-(rate * x)).ln_1m_exp()
}

/// Exponential log-CDF with the below-support floor folded in.
pub fn exponential_ln_cdf<T: Real>(x: T, rate: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    exponential_ln_cdf_on_support(safe_x, rate).mask_invalid(in_support, T::neg_infinity())
}

/// Exponential log-survival with the below-support branch folded in.
pub fn exponential_ln_sf<T: Real>(x: T, rate: T) -> T {
    let in_support = x.gt(&T::zero());
    T::select(
        &in_support,
        exponential_log_survival_on_support(x, rate),
        T::zero(),
    )
}

/// Exponential quantile.
/// `ln(1 − p)` is written `ln_1p(−p)`: for a small `p` the subtraction rounds
/// `1 − p` to the nearest `f64` below 1 and quantises the answer in steps of
/// `1.1e-16`, so the lower tail came back about 11% high at `p = 1e-16`.
pub fn exponential_quantile<T: Real>(p: T, rate: T) -> T {
    -((-p).ln_1p()) / rate
}

/// Exponential entropy `1 - ln(rate)`.
pub fn exponential_entropy<T: Real>(rate: T) -> T {
    T::one() - rate.ln()
}

/// Exponential KL divergence.
pub fn kl_exponential<T: Real>(rate_self: T, rate_other: T) -> T {
    (rate_self.clone() / rate_other.clone()).ln() + rate_other / rate_self - T::one()
}

/// Exponential squared Hellinger distance.
pub fn hellinger_sq_exponential<T: Real>(rate_self: T, rate_other: T) -> T {
    let (l0, l1) = (rate_self, rate_other);
    crate::hellinger_sq_from_bc(T::two() * (l0.clone() * l1.clone()).sqrt() / (l0 + l1))
}

/// Exponential chi-squared divergence interior.
pub fn chi_squared_exponential_on_support<T: Real>(rate_self: T, rate_other: T) -> T {
    let (l0, l1) = (rate_self, rate_other);
    l0.clone() * l0.clone() / (l1.clone() * (T::two() * l0 - l1)) - T::one()
}

/// Exponential inverse survival `−ln(p)/λ`, exact for every representable `p`.
///
/// The generic `inverse_cdf(1 − p)` route returns infinity for any `p` below
/// about `1e-16`, because `1 − p` is then `1`.
pub fn exponential_isf<T: Real>(p: T, rate: T) -> T {
    -(p.ln()) / rate
}
