use euclid_numeric::Real;

/// Cauchy log-density.
pub fn cauchy_log_pdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale.clone();
    -T::ln_pi() - scale.ln() - (T::one() + z.clone() * z).ln()
}

/// Cauchy CDF.
/// Below the location the identity `atan(w) + atan(1/w) = π/2` turns
/// `½ − atan(|z|)/π` into `atan(1/|z|)/π`, which has no subtraction. Without
/// it the lower tail is lost: at `z = −3.2e15` the true CDF is `1e-16` and
/// `½ + atan(z)/π` cancels to roughly half of it.
pub fn cauchy_cdf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    let lower = (-z.clone()).recip().atan() / T::pi();
    let body = T::half() + z.clone().atan() / T::pi();
    T::select(&z.lt(&T::zero()), lower, body)
}

/// Cauchy quantile.
///
/// `tan(π(p − ½))` is accurate through the body of the distribution but unusable in the tails.
/// `π/2` is not representable, so `tan` at the nearest `f64` returns `≈1.633e16` rather than an
/// infinity — and because `π(p − ½)` rounds to that same argument for every `p` below about
/// `1e-16`, the whole lower tail collapsed onto one value instead of following
/// `Q(p) ≈ −1/(πp)`. `Q(0)` came back as `−1.633e16` instead of `−∞`.
///
/// The reciprocal (cotangent) form `Q = location − scale·cot(πp)` has no such problem: `πp` is
/// computed without cancellation, `tan(πp) → πp` as `p → 0`, and the division delivers both the
/// correct `−1/(πp)` asymptote and a true infinity at `p = 0`. It is used on each tail with the
/// argument that stays away from `π/2`, keeping the body on the direct form where it is exact.
pub fn cauchy_quantile<T: Real>(p: T, location: T, scale: T) -> T {
    let body = (T::pi() * (p.clone() - T::half())).tan();
    let lower = -(T::pi() * p.clone()).tan().recip();
    let upper = (T::pi() * (T::one() - p.clone())).tan().recip();
    let tail = T::select(&p.le(&T::half()), lower, upper);
    let in_body = (p - T::half()).abs().le(&T::scalar(0.25));
    location + scale * T::select(&in_body, body, tail)
}

/// Cauchy entropy.
pub fn cauchy_entropy<T: Real>(scale: T) -> T {
    (T::scalar(4.0) * T::pi() * scale).ln()
}

/// Cauchy KL divergence.
pub fn kl_cauchy<T: Real>(loc_self: T, scale_self: T, loc_other: T, scale_other: T) -> T {
    let scale_sum = scale_self.clone() + scale_other.clone();
    let dm = loc_self - loc_other;
    ((scale_sum.clone() * scale_sum + dm.clone() * dm)
        / (T::scalar(4.0) * scale_self * scale_other))
        .ln()
}

/// Cauchy log-survival, taken on whichever side keeps `atan` out of its own
/// cancellation.
///
/// Above the location the identity `atan(z) + atan(1/z) = π/2` turns
/// `½ − atan(z)/π` into `atan(1/z)/π`, which has no subtraction at all and
/// delivers the correct `1/(πz)` tail. Below it the complement is small and
/// `ln_1p` carries it.
pub fn cauchy_ln_sf<T: Real>(x: T, location: T, scale: T) -> T {
    let z = (x - location) / scale;
    let upper = (z.clone().recip().atan() / T::pi()).ln();
    let lower = (-((-z.clone()).recip().atan() / T::pi())).ln_1p();
    T::select(&z.ge(&T::zero()), upper, lower)
}
