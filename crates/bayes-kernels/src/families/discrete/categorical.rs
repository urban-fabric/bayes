use euclid::numeric::Real;

/// Categorical log-mass for the selected class probability.
pub fn categorical_log_pmf<T: Real>(selected_probability: T) -> T {
    selected_probability.ln()
}
