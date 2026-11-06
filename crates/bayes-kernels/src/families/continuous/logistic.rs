use euclid_numeric::Real;

use crate::glm::sigmoid;

/// Logistic-distribution log-density.
pub fn logistic_log_pdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale.clone();
    let log_denom = (-z.clone()).ln_1p_exp();
    -scale.ln() - z - log_denom.clone() - log_denom
}

/// Logistic CDF.
pub fn logistic_cdf<T: Real>(x: T, location: T, scale: T) -> T {
    sigmoid((x - location) / scale)
}

/// Logistic log-CDF.
pub fn logistic_ln_cdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    -(-z).ln_1p_exp()
}

/// Logistic log-survival.
pub fn logistic_ln_sf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    -z.ln_1p_exp()
}

/// Logistic quantile.
pub fn logistic_quantile<T: Real>(p: T, location: T, scale: T) -> T {
    // `ln p - ln_1p(-p)` rather than `ln(p/(1 - p))`, which loses the lower
    // tail to the same subtraction described in `exponential_quantile`.
    location + scale * (p.clone().ln() - (-p).ln_1p())
}

/// Logistic entropy.
pub fn logistic_entropy<T: Real>(scale: T) -> T {
    scale.ln() + T::two()
}
