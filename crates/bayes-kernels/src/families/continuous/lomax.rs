use euclid_numeric::Real;

/// Lomax log-density on `x >= 0`.
pub fn lomax_log_pdf_on_support<T: Real>(x: T, shape: T, scale: T) -> T {
    let one = T::one();
    shape.clone().ln() - scale.clone().ln() - (shape + one.clone()) * (one + x / scale).ln()
}

/// Lomax CDF on `x >= 0`.
pub fn lomax_cdf_on_support<T: Real>(x: T, shape: T, scale: T) -> T {
    // Two corrections to `1 - (1 + x/scale)^-shape`, both at the bottom of
    // the range: `ln_1p(x/scale)` keeps the ratio when it is tiny, and
    // `expm1` keeps the result after the power. Together they take the CDF
    // near zero from eight significant digits to full precision.
    -((-(shape * (x / scale).ln_1p())).exp_m1())
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
/// Takes `(shape, scale)`, matching [`lomax_cdf`] and [`lomax_ln_sf`].
///
/// It used to take them the other way round — the only kernel in the family
/// that did. Nothing caught it because both are positive reals, so calling
/// it by analogy with its own CDF silently returned a Lomax with the shape
/// and scale exchanged.
///
/// `expm1` for the final step: near `p = 0` the quantile is
/// `scale·(e^t − 1)` for a tiny `t`, and the subtraction loses it.
pub fn lomax_quantile<T: Real>(p: T, shape: T, scale: T) -> T {
    // `ln_1p(-p)` rather than `ln(1 - p)`: see `exponential_quantile`.
    let log_base = (-p).ln_1p();
    scale * ((-(T::one() / shape)) * log_base).exp_m1()
}
