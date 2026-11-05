use euclid_numeric::Real;

/// k-th order-statistic CDF from a base CDF value.
pub fn order_statistic_cdf<T: Real>(base_cdf: T, rank: T, sample_size: T) -> T {
    base_cdf.beta_reg(rank.clone(), sample_size - rank + T::one())
}
