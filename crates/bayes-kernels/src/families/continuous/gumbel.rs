use euclid::numeric::Real;

/// Gumbel log-density.
pub fn gumbel_log_pdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale.clone();
    -scale.ln() - z.clone() - (-z).exp()
}

/// Gumbel CDF.
pub fn gumbel_cdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    (-((-z).exp())).exp()
}

/// Gumbel log-CDF.
pub fn gumbel_ln_cdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    -(-z).exp()
}

/// Gumbel quantile.
pub fn gumbel_quantile<T: Real>(p: T, location: T, scale: T) -> T {
    location - scale * (-(p.ln())).ln()
}

/// Gumbel entropy.
pub fn gumbel_entropy<T: Real>(scale: T) -> T {
    scale.ln() + T::euler_gamma() + T::one()
}

/// Gumbel excess kurtosis — the scale-free constant `12/5`.
pub fn gumbel_excess_kurtosis<T: Real>() -> T {
    T::scalar(12.0) / T::scalar(5.0)
}

/// Gumbel KL divergence.
pub fn kl_gumbel<T: Real>(loc_self: T, scale_self: T, loc_other: T, scale_other: T) -> T {
    let (mu0, b0, mu1, b1) = (loc_self, scale_self, loc_other, scale_other);
    (b1.clone() / b0.clone()).ln()
        + T::euler_gamma() * (b0.clone() / b1.clone() - T::one())
        + (mu0.clone() - mu1.clone()) / b1.clone()
        + ((mu1 - mu0) / b1.clone()).exp() * (T::one() + b0 / b1).ln_gamma().exp()
        - T::one()
}
