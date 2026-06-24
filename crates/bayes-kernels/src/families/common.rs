use euclid::numeric::Real;

pub(super) fn gaussian_bc<T: Real>(m0: T, s0: T, m1: T, s1: T) -> T {
    let v = s0.clone() * s0.clone() + s1.clone() * s1.clone();
    let dm = m0 - m1;
    (T::two() * s0 * s1 / v.clone()).sqrt() * (-T::scalar(0.25) * dm.clone() * dm / v).exp()
}

pub(super) fn gamma_bc<T: Real>(a0: T, b0: T, a1: T, b1: T) -> T {
    let a = T::half() * (a0.clone() + a1.clone());
    let b = T::half() * (b0.clone() + b1.clone());
    let ln_bc = a.clone().ln_gamma() + T::half() * (a0.clone() * b0.ln() + a1.clone() * b1.ln())
        - T::half() * (a0.ln_gamma() + a1.ln_gamma())
        - a * b.ln();
    ln_bc.exp()
}

pub(super) fn ln_beta<T: Real>(a: T, b: T) -> T {
    a.clone().ln_gamma() + b.clone().ln_gamma() - (a + b).ln_gamma()
}
