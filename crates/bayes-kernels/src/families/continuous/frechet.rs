use euclid::numeric::Real;

/// Frechet log-density on `x > 0`.
pub fn frechet_log_pdf_on_support<T: Real>(x: T, scale: T, shape: T) -> T {
    let y = x / scale.clone();
    let log_y = y.ln();
    let y_neg_shape = (-(shape.clone()) * log_y.clone()).exp();
    shape.clone().ln() - scale.ln() - (shape + T::one()) * log_y - y_neg_shape
}

/// Frechet CDF on `x > 0`.
pub fn frechet_cdf_on_support<T: Real>(x: T, scale: T, shape: T) -> T {
    (-((shape * (scale / x).ln()).exp())).exp()
}

/// Frechet CDF with the below-support floor folded in.
pub fn frechet_cdf<T: Real>(x: T, scale: T, shape: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, scale.clone());
    frechet_cdf_on_support(safe_x, scale, shape).mask_invalid(in_support, T::zero())
}

/// Frechet log-CDF with the below-support floor folded in.
pub fn frechet_ln_cdf<T: Real>(x: T, scale: T, shape: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, scale.clone());
    let ln_cdf = -((shape * (scale / safe_x).ln()).exp());
    ln_cdf.mask_invalid(in_support, T::neg_infinity())
}

/// Frechet quantile.
pub fn frechet_quantile<T: Real>(p: T, scale: T, shape: T) -> T {
    let base = -(p.ln());
    scale * ((-(T::one() / shape)) * base.ln()).exp()
}
