use euclid::numeric::Real;

/// Half-normal log-density on `x >= 0`.
pub fn half_normal_log_pdf_on_support<T: Real>(x: T, scale: T) -> T {
    let scale_sq = scale.clone() * scale.clone();
    let log_sqrt_2_over_pi = T::half() * (T::two() / T::pi()).ln();
    log_sqrt_2_over_pi - scale.ln() - T::half() * (x.clone() * x) / scale_sq
}

/// Half-normal CDF on `x >= 0`.
pub fn half_normal_cdf_on_support<T: Real>(x: T, scale: T) -> T {
    (x / (scale * T::sqrt_2())).erf()
}

/// Half-normal CDF with the below-support floor folded in.
pub fn half_normal_cdf<T: Real>(x: T, scale: T) -> T {
    let in_support = x.ge(&T::zero());
    T::select(&in_support, half_normal_cdf_on_support(x, scale), T::zero())
}

/// Half-normal quantile.
pub fn half_normal_quantile<T: Real>(p: T, scale: T) -> T {
    scale * T::sqrt_2() * p.erf_inv()
}

/// Half-normal entropy.
pub fn half_normal_entropy<T: Real>(scale: T) -> T {
    T::half() * (T::pi() * scale.clone() * scale / T::two()).ln() + T::half()
}
