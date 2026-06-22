use euclid::numeric::Real;

/// Discrete-uniform log-mass on its support.
pub fn discrete_uniform_log_pmf_on_support<T: Real>(count: T) -> T {
    -count.ln()
}

/// Discrete-uniform entropy.
pub fn discrete_uniform_entropy<T: Real>(count: T) -> T {
    count.ln()
}

/// Discrete-uniform KL on nested support.
pub fn kl_discrete_uniform_on_support<T: Real>(count_self: T, count_other: T) -> T {
    (count_other / count_self).ln()
}

/// Discrete-uniform squared Hellinger distance.
pub fn hellinger_sq_discrete_uniform<T: Real>(
    lo_self: T,
    hi_self: T,
    lo_other: T,
    hi_other: T,
) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone())
        + T::one())
    .max(T::zero());
    let n_self = hi_self - lo_self + T::one();
    let n_other = hi_other - lo_other + T::one();
    crate::hellinger_sq_from_bc(overlap / (n_self * n_other).sqrt())
}

/// Discrete-uniform chi-squared divergence on nested support.
pub fn chi_squared_discrete_uniform_on_support<T: Real>(count_self: T, count_other: T) -> T {
    count_other / count_self - T::one()
}

/// Discrete-uniform Renyi divergence interior.
pub fn renyi_discrete_uniform_on_support<T: Real>(
    lo_self: T,
    hi_self: T,
    lo_other: T,
    hi_other: T,
    alpha: T,
) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone())
        + T::one())
    .max(T::zero());
    let n_self = hi_self - lo_self + T::one();
    let n_other = hi_other - lo_other + T::one();
    let one_minus_alpha = T::one() - alpha.clone();
    let s =
        overlap * (-alpha.clone() * n_self.ln()).exp() * (-one_minus_alpha * n_other.ln()).exp();
    s.ln() / (alpha - T::one())
}
