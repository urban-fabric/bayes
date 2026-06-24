use euclid::numeric::Real;

/// Rayleigh log-density on `x >= 0`.
pub fn rayleigh_log_pdf_on_support<T: Real>(x: T, scale: T) -> T {
    let scale_sq = scale.clone() * scale.clone();
    x.clone().ln() - scale.clone().ln() - scale.ln() - T::half() * (x.clone() * x) / scale_sq
}

/// Rayleigh CDF on `x >= 0`.
pub fn rayleigh_cdf_on_support<T: Real>(x: T, scale: T) -> T {
    T::one() - (-(x.clone() * x) / (T::two() * scale.clone() * scale)).exp()
}

/// Rayleigh CDF with the below-support floor folded in.
pub fn rayleigh_cdf<T: Real>(x: T, scale: T) -> T {
    let in_support = x.ge(&T::zero());
    T::select(&in_support, rayleigh_cdf_on_support(x, scale), T::zero())
}

/// Rayleigh quantile.
pub fn rayleigh_quantile<T: Real>(p: T, scale: T) -> T {
    scale * (-T::two() * (T::one() - p).ln()).sqrt()
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
