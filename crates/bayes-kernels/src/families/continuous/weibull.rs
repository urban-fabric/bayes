use euclid::numeric::Real;

/// Weibull log-density on `x >= 0`.
pub fn weibull_log_pdf_on_support<T: Real>(x: T, scale: T, shape: T) -> T {
    let y = x / scale.clone();
    let log_y = y.ln();
    shape.clone().ln() - scale.ln() + (shape.clone() - T::one()) * log_y.clone()
        - (shape * log_y).exp()
}

/// Weibull CDF on `x >= 0`.
pub fn weibull_cdf_on_support<T: Real>(x: T, scale: T, shape: T) -> T {
    let powered = (shape * (x / scale).ln()).exp();
    T::one() - (-powered).exp()
}

/// Weibull CDF with the below-support floor folded in.
pub fn weibull_cdf<T: Real>(x: T, scale: T, shape: T) -> T {
    let in_support = x.ge(&T::zero());
    let safe_x = T::select(&in_support, x, scale.clone());
    weibull_cdf_on_support(safe_x, scale, shape).mask_invalid(in_support, T::zero())
}

/// Weibull quantile.
pub fn weibull_quantile<T: Real>(p: T, scale: T, shape: T) -> T {
    let base = -((T::one() - p).ln());
    scale * ((T::one() / shape) * base.ln()).exp()
}

/// Weibull entropy.
pub fn weibull_entropy<T: Real>(scale: T, shape: T) -> T {
    T::euler_gamma() * (T::one() - T::one() / shape.clone()) + (scale / shape).ln() + T::one()
}

/// Weibull KL divergence.
pub fn kl_weibull<T: Real>(shape_self: T, scale_self: T, shape_other: T, scale_other: T) -> T {
    let (k0, l0, k1, l1) = (shape_self, scale_self, shape_other, scale_other);
    let term0 = k0.clone().ln() - k0.clone() * l0.clone().ln();
    let term1 = k1.clone().ln() - k1.clone() * l1.clone().ln();
    let cross = (k0.clone() - k1.clone()) * (l0.clone().ln() - T::euler_gamma() / k0.clone());
    let ratio_pow = (k1.clone() * (l0 / l1).ln()).exp();
    let gamma_term = (T::one() + k1 / k0).ln_gamma().exp();
    term0 - term1 + cross + ratio_pow * gamma_term - T::one()
}
