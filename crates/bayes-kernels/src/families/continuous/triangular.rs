use euclid::numeric::{Boolean, Real};

/// Triangular PDF on `[lo, hi]` with mode `mode`.
pub fn triangular_pdf<T: Real>(x: T, lo: T, hi: T, mode: T) -> T {
    let width = hi.clone() - lo.clone();
    let up = T::two() * (x.clone() - lo.clone()) / (width.clone() * (mode.clone() - lo.clone()));
    let down = T::two() * (hi.clone() - x.clone()) / (width * (hi.clone() - mode.clone()));
    let ramp = T::select(&x.lt(&mode), up, down);
    let inside = x.ge(&lo).and(&x.le(&hi));
    T::select(&inside, ramp, T::zero())
}

/// Triangular CDF on `[lo, hi]` with mode `mode`.
pub fn triangular_cdf<T: Real>(x: T, lo: T, hi: T, mode: T) -> T {
    let width = hi.clone() - lo.clone();
    let lower = (x.clone() - lo.clone()) * (x.clone() - lo.clone())
        / (width.clone() * (mode.clone() - lo.clone()));
    let upper = T::one()
        - (hi.clone() - x.clone()) * (hi.clone() - x.clone())
            / (width * (hi.clone() - mode.clone()));
    let body = T::select(&x.lt(&mode), lower, upper);
    let clamped = T::select(&x.ge(&hi), T::one(), body);
    T::select(&x.le(&lo), T::zero(), clamped)
}

/// Triangular quantile on `[lo, hi]` with mode `mode`.
pub fn triangular_quantile<T: Real>(p: T, lo: T, hi: T, mode: T) -> T {
    let width = hi.clone() - lo.clone();
    let split = (mode.clone() - lo.clone()) / width.clone();
    let left = lo.clone() + (width.clone() * (mode.clone() - lo.clone()) * p.clone()).sqrt();
    let right = hi.clone() - (width * (hi - mode) * (T::one() - p.clone())).sqrt();
    T::select(&p.lt(&split), left, right)
}

/// Triangular entropy.
pub fn triangular_entropy<T: Real>(low: T, high: T) -> T {
    T::half() + ((high - low) / T::two()).ln()
}
