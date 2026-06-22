use euclid::numeric::{Integer, Real};

/// Geometric log-mass for trial-count parameterization.
pub fn geometric_log_pmf<I, T>(trials: I, p: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    let one = T::one();
    (trials.to_real() - one.clone()) * (one - p.clone()).ln() + p.ln()
}

/// Geometric log-mass for failures before the first success.
pub fn geometric_failures_log_pmf<I, T>(failures: I, p: T) -> T
where
    I: Integer<Real = T>,
    T: Real,
{
    let one = T::one();
    failures.to_real() * (one - p.clone()).ln() + p.ln()
}

/// Geometric skewness.
pub fn geometric_skewness<T: Real>(p: T) -> T {
    (T::two() - p.clone()) / (T::one() - p).sqrt()
}

/// Geometric KL divergence.
pub fn kl_geometric<T: Real>(p_self: T, p_other: T) -> T {
    let one = T::one();
    let q_self = one.clone() - p_self.clone();
    let q_other = one - p_other.clone();
    (p_self.clone() / p_other).ln() + (q_self.clone() / p_self) * (q_self / q_other).ln()
}
