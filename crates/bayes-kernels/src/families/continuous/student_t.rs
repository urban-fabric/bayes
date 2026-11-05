use euclid_numeric::Real;

/// Student-t log-density for a location/scale t distribution.
pub fn student_t_log_pdf<T: Real>(x: T, location: T, scale: T, freedom: T, log_normalizer: T) -> T {
    let z = (x - location) / scale.clone();
    let one = T::one();
    let inner = one.clone() + (z.clone() * z) / freedom.clone();
    -scale.ln() - log_normalizer - T::half() * (freedom + one) * inner.ln()
}

/// Student-t CDF, taken on whichever side has no subtraction.
///
/// Below the location the answer is just `½·I_w(ν/2, ½)`; the symmetric form
/// `½ + ½·sgn(z)·(1 − I_w)` instead subtracts a tiny `I_w` from 1 and then
/// subtracts the result from a half, which quantises the lower tail into steps
/// of `1.1e-16`. That floor is what stopped the quantile's bisection resolving
/// past `t ≈ −3212` for five degrees of freedom, since below it the CDF had
/// nothing left to distinguish one `t` from another.
pub fn students_t_cdf<T: Real>(x: T, location: T, scale: T, freedom: T) -> T {
    let z = (x - location) / scale;
    let w = freedom.clone() / (freedom.clone() + z.clone() * z.clone());
    let ix = w.beta_reg(freedom * T::half(), T::half());
    let lower = T::half() * ix.clone();
    let upper = T::one() - T::half() * ix;
    T::select(&z.le(&T::zero()), lower, upper)
}

/// Student-t entropy.
pub fn students_t_entropy<T: Real>(scale: T, freedom: T) -> T {
    let half_nu = freedom.clone() * T::half();
    let half_nu_plus_one = (freedom.clone() + T::one()) * T::half();
    scale.ln() + T::half() * (freedom * T::pi()).ln() + half_nu.clone().ln_gamma()
        - half_nu_plus_one.clone().ln_gamma()
        + half_nu_plus_one.clone() * (half_nu_plus_one.digamma() - half_nu.digamma())
}

/// Student-t log-survival, read **directly** off the incomplete beta.
///
/// The CDF is `½ + ½·sgn(z)·(1 − I_w(ν/2, ½))` with `w = ν/(ν + z²)`, so above
/// the location the survival is just `½·I_w(ν/2, ½)` — the subtraction cancels
/// out of the algebra rather than being performed. That gives the correct
/// `z^{-ν}` polynomial tail at every scale; `1 − cdf` gives zero past `1e-16`.
pub fn students_t_ln_sf<T: Real>(x: T, location: T, scale: T, freedom: T) -> T {
    let z = (x - location) / scale;
    let w = freedom.clone() / (freedom.clone() + z.clone() * z.clone());
    let ix = w.beta_reg(freedom * T::half(), T::half());
    let upper = (T::half() * ix.clone()).ln();
    let lower = (-(T::half() * ix)).ln_1p();
    T::select(&z.ge(&T::zero()), upper, lower)
}
