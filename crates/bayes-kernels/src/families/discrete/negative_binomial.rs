use euclid::numeric::{Integer, Real};

/// Negative-binomial log-mass in failures-before-r-successes parameterization.
pub fn negative_binomial_log_pmf<I, T>(
    failures: I,
    successes_until_stop: T,
    p: T,
    ln_coefficient: T,
) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    let one = T::one();
    ln_coefficient + successes_until_stop * p.clone().ln() + failures.to_real() * (one - p).ln()
}

/// Negative-binomial CDF at `k` failures.
pub fn negative_binomial_cdf<I, T>(failures: I, successes_until_stop: T, p: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    p.beta_reg(successes_until_stop, failures.to_real() + T::one())
}

/// Negative-binomial skewness.
pub fn negative_binomial_skewness<T: Real>(successes: T, p: T) -> T {
    (T::two() - p.clone()) / (successes * (T::one() - p)).sqrt()
}

/// Negative-binomial KL for a matched number of successes.
pub fn kl_negative_binomial<T: Real>(successes: T, p_self: T, p_other: T) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let q_other = one - p_other.clone();
    successes.clone() * (p_self.clone() / p_other).ln()
        + successes * (q_self.clone() / p_self) * (q_self / q_other).ln()
}

/// Negative-binomial squared Hellinger distance.
pub fn hellinger_sq_negative_binomial<T: Real>(successes: T, p_self: T, p_other: T) -> T {
    let one = T::one();
    let bc = (p_self.clone() * p_other.clone()).sqrt()
        / (one.clone() - ((one.clone() - p_self) * (one - p_other)).sqrt());
    crate::hellinger_sq_from_bc((successes * bc.ln()).exp())
}

/// Negative-binomial chi-squared divergence interior.
pub fn chi_squared_negative_binomial_on_support<T: Real>(successes: T, p_self: T, p_other: T) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let z = q_self.clone() * q_self / (one.clone() - p_other.clone());
    (successes * (T::two() * p_self.ln() - p_other.ln() - (one - z).ln())).exp() - T::one()
}

/// Negative-binomial Renyi divergence interior.
pub fn renyi_negative_binomial_on_support<T: Real>(
    successes: T,
    p_self: T,
    p_other: T,
    alpha: T,
) -> T {
    let one = T::one();
    let one_minus_alpha = one.clone() - alpha.clone();
    let w = (alpha.clone() * (one.clone() - p_self.clone()).ln()).exp()
        * (one_minus_alpha.clone() * (one.clone() - p_other.clone()).ln()).exp();
    successes * (alpha.clone() * p_self.ln() + one_minus_alpha * p_other.ln() - (one - w).ln())
        / (alpha - T::one())
}
