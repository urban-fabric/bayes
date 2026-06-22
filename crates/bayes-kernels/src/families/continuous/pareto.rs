use euclid::numeric::Real;

/// Pareto log-density on `x >= minimum`.
pub fn pareto_log_pdf_on_support<T: Real>(x: T, minimum: T, shape: T) -> T {
    let one = T::one();
    shape.clone().ln() + shape.clone() * minimum.ln() - (shape + one) * x.ln()
}

/// Pareto CDF on `x >= minimum`.
pub fn pareto_cdf_on_support<T: Real>(x: T, minimum: T, shape: T) -> T {
    T::one() - (shape * (minimum / x).ln()).exp()
}

/// Pareto CDF with the below-support floor folded in.
pub fn pareto_cdf<T: Real>(x: T, minimum: T, shape: T) -> T {
    let in_support = x.ge(&minimum);
    let safe_x = T::select(&in_support, x, minimum.clone());
    pareto_cdf_on_support(safe_x, minimum, shape).mask_invalid(in_support, T::zero())
}

/// Pareto quantile.
pub fn pareto_quantile<T: Real>(p: T, minimum: T, shape: T) -> T {
    let base = T::one() - p;
    minimum * ((-(T::one() / shape)) * base.ln()).exp()
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
