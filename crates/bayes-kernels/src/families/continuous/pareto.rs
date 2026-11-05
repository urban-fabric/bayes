use euclid_numeric::Real;

/// Pareto log-density on `x >= minimum`.
pub fn pareto_log_pdf_on_support<T: Real>(x: T, minimum: T, shape: T) -> T {
    let one = T::one();
    shape.clone().ln() + shape.clone() * minimum.ln() - (shape + one) * x.ln()
}

/// Pareto CDF on `x >= minimum`, with both subtractions removed.
///
/// Just above the minimum the answer is tiny and the textbook form
/// `1 − (x_m/x)^α` loses it **twice**. `expm1` handles the outer subtraction.
/// The inner one is `ln(x_m/x)`: for `x` a hair above `x_m` that ratio is
/// `1 − ε` and its logarithm keeps only the digits the ratio had left. Written
/// as `−α·ln_1p((x − x_m)/x_m)` the relative offset is formed exactly, so the
/// CDF is accurate from `x_m` outward rather than from a few ulps above it.
pub fn pareto_cdf_on_support<T: Real>(x: T, minimum: T, shape: T) -> T {
    let offset = (x - minimum.clone()) / minimum;
    -((-(shape * offset.ln_1p())).exp_m1())
}

/// Pareto CDF with the below-support floor folded in.
pub fn pareto_cdf<T: Real>(x: T, minimum: T, shape: T) -> T {
    let in_support = x.ge(&minimum);
    let safe_x = T::select(&in_support, x, minimum.clone());
    pareto_cdf_on_support(safe_x, minimum, shape).mask_invalid(in_support, T::zero())
}

/// Pareto quantile.
pub fn pareto_quantile<T: Real>(p: T, minimum: T, shape: T) -> T {
    // `ln_1p(-p)` rather than `ln(1 - p)`: see `exponential_quantile`.
    let log_base = (-p).ln_1p();
    minimum * ((-(T::one() / shape)) * log_base).exp()
}

/// Pareto entropy.
pub fn pareto_entropy<T: Real>(minimum: T, shape: T) -> T {
    (minimum / shape.clone()).ln() + T::one() / shape + T::one()
}

/// Pareto KL on nested support.
pub fn kl_pareto_on_support<T: Real>(
    minimum_self: T,
    shape_self: T,
    minimum_other: T,
    shape_other: T,
) -> T {
    let (xm0, a0, xm1, a1) = (minimum_self, shape_self, minimum_other, shape_other);
    (a0.clone() / a1.clone()).ln() + a1.clone() * (xm0 / xm1).ln() + a1 / a0 - T::one()
}

/// Pareto log-survival on `x >= x_m`: `−α·ln_1p((x − x_m)/x_m)`.
///
/// The survival of a Pareto is a pure power law, so its logarithm is linear in
/// `ln x` — exact at every scale, unlike `ln(1 − cdf)`.
///
/// Written through `ln_1p` rather than as the difference `α(ln x_m − ln x)`.
/// The two agree far from the minimum, but near it that difference is a
/// cancellation between two values both close to `ln x_m`, while the relative
/// offset `(x − x_m)/x_m` is formed exactly.
pub fn pareto_ln_sf<T: Real>(x: T, scale: T, shape: T) -> T {
    let in_support = x.ge(&scale);
    let safe_x = T::select(&in_support, x, scale.clone());
    let offset = (safe_x - scale.clone()) / scale;
    (-(shape * offset.ln_1p())).mask_invalid(in_support, T::zero())
}

/// Pareto inverse survival `x_m p^{−1/α}`.
pub fn pareto_isf<T: Real>(p: T, scale: T, shape: T) -> T {
    scale * (-(p.ln()) / shape).exp()
}
