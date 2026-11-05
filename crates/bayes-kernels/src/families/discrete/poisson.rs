use euclid_numeric::{Integer, Real};

pub use crate::families::continuous::gamma::{
    gamma_poisson_log_marginal_likelihood, gamma_poisson_log_predictive,
};
pub use crate::glm::{poisson_log_curvature, poisson_log_score};

/// Poisson log-mass.
pub fn poisson_log_pmf<I, T>(k: I, lambda: T, ln_factorial_k: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    k.to_real() * lambda.clone().ln() - lambda - ln_factorial_k
}

/// Poisson CDF at integer `k`.
pub fn poisson_cdf<I, T>(k: I, rate: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    rate.gamma_ur(k.to_real() + T::one())
}

/// Poisson skewness.
pub fn poisson_skewness<T: Real>(rate: T) -> T {
    T::one() / rate.sqrt()
}

/// Poisson excess kurtosis.
pub fn poisson_excess_kurtosis<T: Real>(rate: T) -> T {
    T::one() / rate
}

/// Poisson KL divergence.
pub fn kl_poisson<T: Real>(rate_self: T, rate_other: T) -> T {
    rate_self.clone() * (rate_self.clone() / rate_other.clone()).ln() - rate_self + rate_other
}

/// Poisson squared Hellinger distance.
pub fn hellinger_sq_poisson<T: Real>(rate_self: T, rate_other: T) -> T {
    let d = rate_self.sqrt() - rate_other.sqrt();
    crate::hellinger_sq_from_bc((-T::half() * d.clone() * d).exp())
}

/// Poisson chi-squared divergence.
pub fn chi_squared_poisson<T: Real>(rate_self: T, rate_other: T) -> T {
    let diff = rate_self - rate_other.clone();
    (diff.clone() * diff / rate_other).exp() - T::one()
}
