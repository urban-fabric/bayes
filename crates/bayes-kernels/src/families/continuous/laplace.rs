use euclid::numeric::Real;

/// Laplace log-density.
pub fn laplace_log_pdf<T: Real>(x: T, location: T, scale: T) -> T {
    T::half().ln() - scale.clone().ln() - (x - location).abs() / scale
}

/// Laplace CDF.
pub fn laplace_cdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x.clone() - location.clone()) / scale;
    let left = T::half() * z.clone().exp();
    let right = T::one() - T::half() * (-z).exp();
    T::select(&x.lt(&location), left, right)
}

/// Laplace quantile.
pub fn laplace_quantile<T: Real>(p: T, location: T, scale: T) -> T {
    let q = p - T::half();
    location - scale * q.clone().signum() * (T::one() - T::two() * q.abs()).ln()
}

/// Laplace entropy.
pub fn laplace_entropy<T: Real>(scale: T) -> T {
    (T::two() * T::e() * scale).ln()
}

/// Laplace KL divergence.
pub fn kl_laplace<T: Real>(loc_self: T, scale_self: T, loc_other: T, scale_other: T) -> T {
    let d = (loc_self - loc_other).abs();
    (scale_other.clone() / scale_self.clone()).ln()
        + (scale_self.clone() * (-(d.clone() / scale_self)).exp() + d) / scale_other
        - T::one()
}
