use euclid::numeric::Real;

/// Student-t log-density for a location/scale t distribution.
pub fn student_t_log_pdf<T: Real>(x: T, location: T, scale: T, freedom: T, log_normalizer: T) -> T {
    let z = (x - location) / scale.clone();
    let one = T::one();
    let inner = one.clone() + (z.clone() * z) / freedom.clone();
    -scale.ln() - log_normalizer - T::half() * (freedom + one) * inner.ln()
}

/// Student-t CDF.
pub fn students_t_cdf<T: Real>(x: T, location: T, scale: T, freedom: T) -> T {
    let z = (x - location) / scale;
    let w = freedom.clone() / (freedom.clone() + z.clone() * z.clone());
    let ix = w.beta_reg(freedom * T::half(), T::half());
    T::half() + T::half() * z.signum() * (T::one() - ix)
}

/// Student-t entropy.
pub fn students_t_entropy<T: Real>(scale: T, freedom: T) -> T {
    let half_nu = freedom.clone() * T::half();
    let half_nu_plus_one = (freedom.clone() + T::one()) * T::half();
    scale.ln() + T::half() * (freedom * T::pi()).ln() + half_nu.clone().ln_gamma()
        - half_nu_plus_one.clone().ln_gamma()
        + half_nu_plus_one.clone() * (half_nu_plus_one.digamma() - half_nu.digamma())
}
