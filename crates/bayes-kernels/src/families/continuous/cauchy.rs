use euclid::numeric::Real;

/// Cauchy log-density.
pub fn cauchy_log_pdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale.clone();
    -T::ln_pi() - scale.ln() - (T::one() + z.clone() * z).ln()
}

/// Cauchy CDF.
pub fn cauchy_cdf<T: Real>(x: T, location: T, scale: T) -> T {
    T::half() + ((x - location) / scale).atan() / T::pi()
}

/// Cauchy quantile.
pub fn cauchy_quantile<T: Real>(p: T, location: T, scale: T) -> T {
    location + scale * (T::pi() * (p - T::half())).tan()
}

/// Cauchy entropy.
pub fn cauchy_entropy<T: Real>(scale: T) -> T {
    (T::scalar(4.0) * T::pi() * scale).ln()
}

/// Cauchy KL divergence.
pub fn kl_cauchy<T: Real>(loc_self: T, scale_self: T, loc_other: T, scale_other: T) -> T {
    let scale_sum = scale_self.clone() + scale_other.clone();
    let dm = loc_self - loc_other;
    ((scale_sum.clone() * scale_sum + dm.clone() * dm)
        / (T::scalar(4.0) * scale_self * scale_other))
        .ln()
}
