use euclid_numeric::Real;

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
    // `expm1`, for the reason given on `exponential_cdf_on_support`.
    -((-powered).exp_m1())
}

/// Weibull CDF with the below-support floor folded in.
pub fn weibull_cdf<T: Real>(x: T, scale: T, shape: T) -> T {
    let in_support = x.ge(&T::zero());
    let safe_x = T::select(&in_support, x, scale.clone());
    weibull_cdf_on_support(safe_x, scale, shape).mask_invalid(in_support, T::zero())
}

/// Weibull quantile.
pub fn weibull_quantile<T: Real>(p: T, scale: T, shape: T) -> T {
    // `ln_1p(-p)` rather than `ln(1 - p)`: see `exponential_quantile`.
    let base = -((-p).ln_1p());
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

/// Weibull log-survival on `x >= 0`: `−(x/λ)^k`.
///
/// Exact for every representable argument, where `ln(1 − cdf)` has already
/// collapsed to `−inf` — a Weibull with unit scale and shape has survival
/// `4.2e-18` at `x = 40`, which the subtraction cannot see.
pub fn weibull_ln_sf<T: Real>(x: T, scale: T, shape: T) -> T {
    let in_support = x.ge(&T::zero());
    let safe_x = T::select(&in_support, x, scale.clone());
    (-(shape * (safe_x / scale).ln()).exp()).mask_invalid(in_support, T::zero())
}

/// Weibull inverse survival `λ(−ln p)^{1/k}`.
///
/// Takes `p` directly rather than inverting at `1 − p`, so it keeps working
/// below `p = 1e-16` where that subtraction rounds to 1.
pub fn weibull_isf<T: Real>(p: T, scale: T, shape: T) -> T {
    scale * ((-(p.ln())).ln() / shape).exp()
}
