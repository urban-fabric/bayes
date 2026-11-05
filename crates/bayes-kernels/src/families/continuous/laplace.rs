use euclid_numeric::Real;

/// Laplace log-density.
pub fn laplace_log_pdf<T: Real>(x: T, location: T, scale: T) -> T {
    T::half().ln() - scale.clone().ln() - (x - location).abs() / scale
}

/// Laplace CDF.
pub fn laplace_cdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x.clone() - location.clone()) / scale;
    let left = T::half() * z.clone().exp();
    let right = T::one() - T::half() * (-z).exp();
    T::select(&x.lt(&location), left, right)
}

/// Laplace quantile.
/// Each half is taken in the form that has no subtraction. Below the median
/// `Q(p) = μ + b·ln(2p)` exactly; the symmetric expression
/// `1 − 2|p − ½|` instead rounds `p − ½` to `−½` for any `p` under `1.1e-16`
/// and returned a quantile 1.8× too close to the median.
pub fn laplace_quantile<T: Real>(p: T, location: T, scale: T) -> T {
    let lower = location.clone() + scale.clone() * (T::two() * p.clone()).ln();
    let upper = location - scale * (T::two() * (T::one() - p.clone())).ln();
    T::select(&p.le(&T::half()), lower, upper)
}

/// Laplace entropy.
pub fn laplace_entropy<T: Real>(scale: T) -> T {
    (T::two() * T::e() * scale).ln()
}

/// Laplace KL divergence.
pub fn kl_laplace<T: Real>(loc_self: T, scale_self: T, loc_other: T, scale_other: T) -> T {
    let d = (loc_self - loc_other).abs();
    (scale_other.clone() / scale_self.clone()).ln()
        + (scale_self.clone() * (-(d.clone() / scale_self)).exp() + d) / scale_other
        - T::one()
}

/// Laplace log-survival, which is **linear** above the location.
///
/// `ln P(X > x) = −ln 2 − (x − μ)/b` exactly on the upper half, so the tail is
/// available at every representable argument. Below the location the survival
/// is near 1 and `ln_1p` carries the small part.
pub fn laplace_ln_sf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    let upper = -T::ln_2() - z.clone();
    let lower = (-(T::half() * z.clone().exp())).ln_1p();
    T::select(&z.ge(&T::zero()), upper, lower)
}
