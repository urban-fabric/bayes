use euclid::numeric::Real;

/// Hypergeometric log-mass for drawing `k` successes.
pub fn hypergeometric_log_pmf<T: Real>(
    ln_binomial_successes: T,
    ln_binomial_failures: T,
    ln_binomial_population: T,
) -> T {
    ln_binomial_successes + ln_binomial_failures - ln_binomial_population
}
