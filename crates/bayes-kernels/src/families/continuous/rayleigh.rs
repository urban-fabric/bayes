use euclid_numeric::Real;

/// Rayleigh log-density on `x >= 0`.
pub fn rayleigh_log_pdf_on_support<T: Real>(x: T, scale: T) -> T {
    let scale_sq = scale.clone() * scale.clone();
    x.clone().ln() - scale.clone().ln() - scale.ln() - T::half() * (x.clone() * x) / scale_sq
}

/// Rayleigh CDF on `x >= 0`.
pub fn rayleigh_cdf_on_support<T: Real>(x: T, scale: T) -> T {
    // `expm1`, for the reason given on `exponential_cdf_on_support`.
    -((-(x.clone() * x) / (T::two() * scale.clone() * scale)).exp_m1())
}

/// Rayleigh CDF with the below-support floor folded in.
pub fn rayleigh_cdf<T: Real>(x: T, scale: T) -> T {
    let in_support = x.ge(&T::zero());
    T::select(&in_support, rayleigh_cdf_on_support(x, scale), T::zero())
}

/// Rayleigh quantile.
pub fn rayleigh_quantile<T: Real>(p: T, scale: T) -> T {
    // `ln_1p(-p)` rather than `ln(1 - p)`: see `exponential_quantile`.
    scale * (-T::two() * (-p).ln_1p()).sqrt()
}

/// Rayleigh entropy.
pub fn rayleigh_entropy<T: Real>(scale: T) -> T {
    T::one() + (scale / T::sqrt_2()).ln() + T::euler_gamma() * T::half()
}

/// Rayleigh skewness `2√π(π−3) / (4−π)^{3/2}` — a scale-free constant
/// (≈ 0.631).
pub fn rayleigh_skewness<T: Real>() -> T {
    let pi = T::pi();
    let d = T::scalar(4.0) - pi.clone();
    T::two() * pi.clone().sqrt() * (pi - T::scalar(3.0)) / (d.clone() * d.clone() * d).sqrt()
}

/// Rayleigh excess kurtosis `−(6π² − 24π + 16) / (4−π)²` — a scale-free
/// constant (≈ 0.245).
pub fn rayleigh_excess_kurtosis<T: Real>() -> T {
    let pi = T::pi();
    let d = T::scalar(4.0) - pi.clone();
    -(T::scalar(6.0) * pi.clone() * pi.clone() - T::scalar(24.0) * pi + T::scalar(16.0))
        / (d.clone() * d)
}

/// Rayleigh KL divergence.
pub fn kl_rayleigh<T: Real>(scale_self: T, scale_other: T) -> T {
    let ratio = scale_self.clone() / scale_other.clone();
    T::two() * (scale_other / scale_self).ln() + ratio.clone() * ratio - T::one()
}

/// Rayleigh log-survival on `x >= 0`: `−x²/(2σ²)`.
pub fn rayleigh_ln_sf<T: Real>(x: T, scale: T) -> T {
    let in_support = x.ge(&T::zero());
    let safe_x = T::select(&in_support, x, T::zero());
    let z = safe_x / scale;
    (-T::half() * z.clone() * z).mask_invalid(in_support, T::zero())
}

/// Rayleigh inverse survival `σ√(−2 ln p)`.
pub fn rayleigh_isf<T: Real>(p: T, scale: T) -> T {
    scale * (-T::two() * p.ln()).sqrt()
}
