use euclid::numeric::Real;

/// Lomax log-density on `x >= 0`.
pub fn lomax_log_pdf_on_support<T: Real>(x: T, shape: T, scale: T) -> T {
    let one = T::one();
    shape.clone().ln() - scale.clone().ln() - (shape + one.clone()) * (one + x / scale).ln()
}

/// Lomax CDF on `x >= 0`.
pub fn lomax_cdf_on_support<T: Real>(x: T, shape: T, scale: T) -> T {
    T::one() - (-(shape * (T::one() + x / scale).ln())).exp()
}

/// Lomax CDF with the below-support floor folded in.
pub fn lomax_cdf<T: Real>(x: T, shape: T, scale: T) -> T {
    let in_support = x.ge(&T::zero());
    let safe_x = T::select(&in_support, x, T::zero());
    lomax_cdf_on_support(safe_x, shape, scale).mask_invalid(in_support, T::zero())
}

/// Lomax log-survival with the below-support branch folded in.
pub fn lomax_ln_sf<T: Real>(x: T, shape: T, scale: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::zero());
    let ln_sf = -(shape * (T::one() + safe_x / scale).ln());
    T::select(&in_support, ln_sf, T::zero())
}

/// Lomax quantile.
pub fn lomax_quantile<T: Real>(p: T, scale: T, shape: T) -> T {
    let base = T::one() - p;
    scale * (((-(T::one() / shape)) * base.ln()).exp() - T::one())
}
