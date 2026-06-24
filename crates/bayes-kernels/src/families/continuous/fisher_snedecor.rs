use euclid::numeric::Real;

/// Fisher-Snedecor F log-density on `x > 0`.
pub fn fisher_snedecor_log_pdf_on_support<T: Real>(
    x: T,
    d1: T,
    d2: T,
    half_d1: T,
    half_d2: T,
    ln_beta_half_degrees: T,
) -> T {
    let d1x = d1.clone() * x.clone();
    half_d1.clone() * d1x.clone().ln() + half_d2.clone() * d2.clone().ln()
        - (half_d1 + half_d2) * (d1x + d2).ln()
        - x.ln()
        - ln_beta_half_degrees
}

/// Fisher-Snedecor F CDF on `x > 0`.
pub fn fisher_snedecor_cdf_on_support<T: Real>(x: T, d1: T, d2: T) -> T {
    let d1x = d1.clone() * x;
    (d1x.clone() / (d1x + d2.clone())).beta_reg(d1 * T::half(), d2 * T::half())
}

/// Fisher-Snedecor F CDF with the below-support floor folded in.
pub fn fisher_snedecor_cdf<T: Real>(x: T, d1: T, d2: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    fisher_snedecor_cdf_on_support(safe_x, d1, d2).mask_invalid(in_support, T::zero())
}
