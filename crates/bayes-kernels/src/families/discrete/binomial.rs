use euclid::numeric::{Integer, Real};

pub use crate::families::continuous::beta::beta_binomial_log_predictive;
pub use crate::glm::{binomial_logit_curvature, binomial_logit_score};

use super::bernoulli::kl_bernoulli_on_unit;

/// Binomial log-mass for supplied success/failure counts.
pub fn binomial_log_pmf<I, T>(successes: I, failures: I, p: T, ln_binomial: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    let one = T::one();
    ln_binomial + successes.to_real() * p.clone().ln() + failures.to_real() * (one - p).ln()
}

/// Binomial CDF at `k < n`.
pub fn binomial_cdf_on_support<I, T>(successes: I, trials: I, p: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    let s = successes.to_real();
    (T::one() - p).beta_reg(trials.to_real() - s.clone(), s + T::one())
}

/// Binomial skewness.
pub fn binomial_skewness<T: Real>(trials: T, p: T) -> T {
    let variance = trials * p.clone() * (T::one() - p.clone());
    (T::one() - T::two() * p) / variance.sqrt()
}

/// Binomial KL for a matched trial count.
pub fn kl_binomial<T: Real>(trials: T, p_self: T, p_other: T) -> T {
    trials * kl_bernoulli_on_unit(p_self, p_other)
}

/// Binomial squared Hellinger distance for a matched trial count.
pub fn hellinger_sq_binomial<T: Real>(trials: T, p_self: T, p_other: T) -> T {
    let one = T::one();
    let bc_bernoulli = (p_self.clone() * p_other.clone()).sqrt()
        + ((one.clone() - p_self) * (one - p_other)).sqrt();
    crate::hellinger_sq_from_bc((trials * bc_bernoulli.ln()).exp())
}

/// Binomial chi-squared divergence for a matched trial count.
pub fn chi_squared_binomial<T: Real>(trials: T, p_self: T, p_other: T) -> T {
    let diff = p_self - p_other.clone();
    let chi_bernoulli = diff.clone() * diff / (p_other.clone() * (T::one() - p_other));
    (trials * (T::one() + chi_bernoulli).ln()).exp() - T::one()
}

/// Binomial Renyi divergence for a matched trial count.
pub fn renyi_binomial<T: Real>(trials: T, p_self: T, p_other: T, alpha: T) -> T {
    let one = T::one();
    let one_minus_alpha = one.clone() - alpha.clone();
    let term =
        |p: T, q: T| (alpha.clone() * p.ln()).exp() * (one_minus_alpha.clone() * q.ln()).exp();
    let s = term(p_self.clone(), p_other.clone()) + term(one.clone() - p_self, one - p_other);
    trials * s.ln() / (alpha - T::one())
}
