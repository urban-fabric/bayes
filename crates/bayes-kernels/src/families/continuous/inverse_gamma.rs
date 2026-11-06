use euclid_numeric::Real;

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

/// Inverse-gamma log-survival `ln P(a, β/x)`.
///
/// Note which incomplete gamma: an inverse-gamma's *CDF* is the **upper**
/// regularized function `Q(a, β/x)`, so its survival is the **lower** one. The
/// two are not interchangeable, and picking the wrong one produces a plausible
/// decreasing function that is not this distribution's tail.
///
/// The branch is not optional. A survival has two ends and they need opposite
/// treatments: where it is **small** the complementary special function is the
/// accurate one, and where it is **near 1** that function returns `1` and
/// throws the deficit away — there `ln_1p(−cdf)` keeps it. Taking only the
/// first route made `cdf + sf` come to `1 + 1e-9` for `FisherSnedecor(1, 1)`
/// just above zero.
pub fn inverse_gamma_ln_sf<T: Real>(x: T, shape: T, scale: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    let z = scale / safe_x;
    let survival = z.clone().gamma_lr(shape.clone());
    let cumulative = z.gamma_ur(shape);
    T::select(
        &survival.gt(&T::half()),
        (-cumulative).ln_1p(),
        survival.ln(),
    )
    .mask_invalid(in_support, T::zero())
}
