use euclid_numeric::Real;

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

/// Half-normal skewness `√2(4−π) / (π−2)^{3/2}` — a scale-free constant
/// (≈ 0.995).
pub fn half_normal_skewness<T: Real>() -> T {
    let pi = T::pi();
    let d = pi.clone() - T::two();
    T::sqrt_2() * (T::scalar(4.0) - pi) / (d.clone() * d.clone() * d).sqrt()
}

/// Half-normal excess kurtosis `8(π−3) / (π−2)²` — a scale-free constant
/// (≈ 0.869).
pub fn half_normal_excess_kurtosis<T: Real>() -> T {
    let pi = T::pi();
    let d = pi.clone() - T::two();
    T::scalar(8.0) * (pi - T::scalar(3.0)) / (d.clone() * d)
}

/// Half-normal quantile.
pub fn half_normal_quantile<T: Real>(p: T, scale: T) -> T {
    scale * T::sqrt_2() * p.erf_inv()
}

/// Half-normal entropy.
pub fn half_normal_entropy<T: Real>(scale: T) -> T {
    T::half() * (T::pi() * scale.clone() * scale / T::two()).ln() + T::half()
}

/// Half-normal log-survival on `x >= 0`: `ln erfc(t)` for `t = x/(σ√2)`.
///
/// Written as `ln erfcx(t) − t²`, because `erfc` itself underflows to zero near
/// `t = 27` while the scaled complementary error function stays `O(1/t)` — the
/// same device the normal's `ln_sf` uses.
pub fn half_normal_ln_sf<T: Real>(x: T, scale: T) -> T {
    let in_support = x.ge(&T::zero());
    let safe_x = T::select(&in_support, x, T::zero());
    let t = safe_x / (scale * T::sqrt_2());
    (t.clone().erfcx().ln() - t.clone() * t).mask_invalid(in_support, T::zero())
}

/// Half-normal inverse survival `σ√2·erfc_inv(p)`.
///
/// The generic `inverse_cdf(1 − p)` route goes through `erf_inv` evaluated a
/// hair below 1, where it is near-vertical: at `p = 1e-16` that returned a
/// point whose actual survival was `9e-14`, out by a factor of 900. `erfc_inv`
/// takes the tail probability as its own argument and has no such problem.
pub fn half_normal_isf<T: Real>(p: T, scale: T) -> T {
    scale * T::sqrt_2() * p.erfc_inv()
}
