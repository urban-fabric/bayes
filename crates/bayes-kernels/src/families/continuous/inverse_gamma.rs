use euclid::numeric::Real;

/// Inverse-gamma log-density on `x > 0`.
pub fn inverse_gamma_log_pdf_on_support<T: Real>(x: T, shape: T, scale: T, ln_gamma_shape: T) -> T {
    let one = T::one();
    shape.clone() * scale.clone().ln() - ln_gamma_shape - (shape + one) * x.clone().ln() - scale / x
}

/// Inverse-gamma CDF with the below-support floor folded in.
pub fn inverse_gamma_cdf<T: Real>(x: T, shape: T, scale: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    (scale / safe_x)
        .gamma_ur(shape)
        .mask_invalid(in_support, T::zero())
}

/// Inverse-gamma entropy.
pub fn inverse_gamma_entropy<T: Real>(shape: T, scale: T) -> T {
    shape.clone() + scale.ln() + shape.clone().ln_gamma()
        - (T::one() + shape.clone()) * shape.digamma()
}
