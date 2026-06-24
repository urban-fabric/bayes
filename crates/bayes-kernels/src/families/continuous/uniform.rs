use euclid::numeric::Real;

/// Uniform log-density on its support.
pub fn uniform_log_pdf_on_support<T: Real>(width: T) -> T {
    -width.ln()
}

/// Uniform CDF on `[lo, lo + width]`.
pub fn uniform_cdf<T: Real>(x: T, lo: T, width: T) -> T {
    ((x - lo) / width).clamp(T::zero(), T::one())
}

/// Uniform quantile.
pub fn uniform_quantile<T: Real>(p: T, lo: T, width: T) -> T {
    lo + width * p
}

/// Uniform entropy.
pub fn uniform_entropy<T: Real>(width: T) -> T {
    width.ln()
}

/// Uniform KL on nested support.
pub fn kl_uniform_on_support<T: Real>(width_self: T, width_other: T) -> T {
    (width_other / width_self).ln()
}

/// Uniform squared Hellinger distance.
pub fn hellinger_sq_uniform<T: Real>(lo_self: T, hi_self: T, lo_other: T, hi_other: T) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone()))
        .max(T::zero());
    let width_self = hi_self - lo_self;
    let width_other = hi_other - lo_other;
    crate::hellinger_sq_from_bc(overlap / (width_self * width_other).sqrt())
}

/// Uniform chi-squared divergence on nested support.
pub fn chi_squared_uniform_on_support<T: Real>(width_self: T, width_other: T) -> T {
    width_other / width_self - T::one()
}

/// Uniform Renyi divergence interior.
pub fn renyi_uniform_on_support<T: Real>(
    lo_self: T,
    hi_self: T,
    lo_other: T,
    hi_other: T,
    alpha: T,
) -> T {
    let overlap = (hi_self.clone().min(hi_other.clone()) - lo_self.clone().max(lo_other.clone()))
        .max(T::zero());
    let width_self = hi_self - lo_self;
    let width_other = hi_other - lo_other;
    let one_minus_alpha = T::one() - alpha.clone();
    let s = overlap
        * (-alpha.clone() * width_self.ln()).exp()
        * (-one_minus_alpha * width_other.ln()).exp();
    s.ln() / (alpha - T::one())
}
