use euclid::numeric::Real;

use super::normal::{normal_cdf, normal_quantile};

/// Log-normal log-density on `x > 0`.
pub fn lognormal_log_pdf_on_support<T: Real>(x: T, mean_log: T, std_log: T) -> T {
    let log_x = x.clone().ln();
    let z = (log_x.clone() - mean_log) / std_log.clone();
    -(T::half() * z.clone() * z) - log_x - std_log.ln() - T::half_ln_tau()
}

/// Log-normal CDF with the below-support floor folded in.
pub fn lognormal_cdf<T: Real>(x: T, mean_log: T, std_log: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    normal_cdf(safe_x.ln(), mean_log, std_log).mask_invalid(in_support, T::zero())
}

/// Log-normal quantile.
pub fn lognormal_quantile<T: Real>(p: T, mean_log: T, std_log: T) -> T {
    normal_quantile(p, mean_log, std_log).exp()
}

/// Log-normal mean.
pub fn lognormal_mean<T: Real>(mean_log: T, std_log: T) -> T {
    (mean_log + std_log.clone() * std_log * T::half()).exp()
}

/// Log-normal variance.
pub fn lognormal_variance<T: Real>(mean_log: T, std_log: T) -> T {
    let var_log = std_log.clone() * std_log;
    (var_log.clone().exp() - T::one()) * (T::two() * mean_log + var_log).exp()
}

/// Log-normal entropy.
pub fn lognormal_entropy<T: Real>(mean_log: T, std_log: T) -> T {
    mean_log + T::half() * (T::tau() * T::e() * std_log.clone() * std_log).ln()
}
