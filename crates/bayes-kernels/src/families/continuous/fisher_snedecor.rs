use euclid_numeric::Real;

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
/// The F CDF on its support, `I_z(d1/2, d2/2)` with `z = d1·x / (d1·x + d2)`.
///
/// Forming `z` directly loses the upper tail. As `x` grows, `z → 1`, and `1 − z` is exactly the
/// quantity `beta_reg` needs — but it has been destroyed by rounding, since `z` only carries `1`
/// to within an ulp. At `d1 = d2 = 1` and `x ≈ 1e15` that made the CDF **non-monotone**: it fell
/// from `0.99999998103` to `0.99999997988` as `x` rose, a 6% error in the survival probability.
/// At `x = ∞` it was worse than wrong: `∞/∞` is `NaN`, so `cdf(∞)` returned `NaN` rather than 1.
///
/// So the complement `1 − z = d2 / (d1·x + d2)` is formed directly, where no cancellation occurs,
/// and the reflection `I_z(a, b) = 1 − I_{1−z}(b, a)` is used above the midpoint. Both branches
/// agree at `z = ½`, and each one feeds `beta_reg` the argument that is accurately known.
pub fn fisher_snedecor_cdf_on_support<T: Real>(x: T, d1: T, d2: T) -> T {
    let d1x = d1.clone() * x;
    let denom = d1x.clone() + d2.clone();
    let z = d1x / denom.clone();
    let z_complement = d2.clone() / denom;
    let (a, b) = (d1 * T::half(), d2 * T::half());
    let lower = z.clone().beta_reg(a.clone(), b.clone());
    let upper = T::one() - z_complement.beta_reg(b, a);
    T::select(&z.le(&T::half()), lower, upper)
}

/// Fisher-Snedecor F CDF with the below-support floor folded in.
pub fn fisher_snedecor_cdf<T: Real>(x: T, d1: T, d2: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    fisher_snedecor_cdf_on_support(safe_x, d1, d2).mask_invalid(in_support, T::zero())
}

/// F log-survival `ln I_{d2/(d1x + d2)}(d2/2, d1/2)`.
///
/// The reflected incomplete beta *is* the survival, so this needs no
/// subtraction at all — and the complement `d2/(d1x + d2)` is formed directly
/// rather than as `1 − d1x/(d1x + d2)`, which is the same reason
/// [`fisher_snedecor_cdf_on_support`] switches branches at the midpoint.
///
/// The branch is not optional. A survival has two ends and they need opposite
/// treatments: where it is **small** the complementary special function is the
/// accurate one, and where it is **near 1** that function returns `1` and
/// throws the deficit away — there `ln_1p(−cdf)` keeps it. Taking only the
/// first route made `cdf + sf` come to `1 + 1e-9` for `FisherSnedecor(1, 1)`
/// just above zero.
pub fn fisher_snedecor_ln_sf<T: Real>(x: T, d1: T, d2: T) -> T {
    let in_support = x.gt(&T::zero());
    let safe_x = T::select(&in_support, x, T::one());
    let d1x = d1.clone() * safe_x;
    let denom = d1x.clone() + d2.clone();
    let y = d1x / denom.clone();
    let complement = d2.clone() / denom;
    let (a, b) = (d1 * T::half(), d2 * T::half());
    let survival = complement.beta_reg(b.clone(), a.clone());
    let cumulative = y.beta_reg(a, b);
    T::select(
        &survival.gt(&T::half()),
        (-cumulative).ln_1p(),
        survival.ln(),
    )
    .mask_invalid(in_support, T::zero())
}
