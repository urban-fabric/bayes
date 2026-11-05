use euclid_numeric::Real;

pub use crate::families::continuous::beta::beta_binomial_log_predictive;

/// Beta-binomial log-mass.
pub fn beta_binomial_log_pmf<T: Real>(ln_binomial: T, ln_beta_posterior: T, ln_beta_prior: T) -> T {
    ln_binomial + ln_beta_posterior - ln_beta_prior
}
