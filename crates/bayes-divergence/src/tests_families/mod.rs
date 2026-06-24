//! Numerical validation of the closed-form divergences for the extended set of
//! families. Every closed form is checked against a trapezoid integral (continuous)
//! or a finite pmf sum (discrete), plus the algebraic Rényi-limit identities
//! (`α→1` is KL, `α=½` is `−2 ln(1−H²)`, `α=2` is `ln(1+χ²)`).

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::{
    Bernoulli, Binomial, ChiSquared, Continuous, Discrete, DiscreteUniform, Frechet, Gamma, Gumbel,
    HalfNormal, InverseGamma, LogNormal, NegativeBinomial, Normal, Uniform, VonMises, Wald,
    Weibull,
};
#[allow(unused_imports)]
use std::f64::consts::PI;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

/// Trapezoid integral of `f` over `[lo, hi]` with `n` panels.
fn integ(lo: f64, hi: f64, n: usize, f: impl Fn(f64) -> f64) -> f64 {
    let h = (hi - lo) / n as f64;
    let mut s = 0.0;
    for k in 0..=n {
        let x = lo + k as f64 * h;
        let w = if k == 0 || k == n { 0.5 } else { 1.0 };
        s += w * f(x);
    }
    s * h
}

/// Numerical `D_KL(p ‖ q) = ∫ p ln(p/q)` for two continuous densities.
fn kl_num(p: &impl Continuous, q: &impl Continuous, lo: f64, hi: f64, n: usize) -> f64 {
    integ(lo, hi, n, |x| {
        let pv = p.pdf(x);
        if pv > 0.0 {
            pv * (pv / q.pdf(x)).ln()
        } else {
            0.0
        }
    })
}

/// Numerical squared Hellinger `1 − ∫√(pq)`.
fn hellinger_num(p: &impl Continuous, q: &impl Continuous, lo: f64, hi: f64, n: usize) -> f64 {
    1.0 - integ(lo, hi, n, |x| (p.pdf(x) * q.pdf(x)).max(0.0).sqrt())
}

/// Numerical χ² `∫ p²/q − 1`.
fn chi2_num(p: &impl Continuous, q: &impl Continuous, lo: f64, hi: f64, n: usize) -> f64 {
    integ(lo, hi, n, |x| {
        let pv = p.pdf(x);
        if pv > 0.0 { pv * pv / q.pdf(x) } else { 0.0 }
    }) - 1.0
}

/// Numerical Rényi `ln(∫ pᵅ q¹⁻ᵅ)/(α−1)`. Cells where either density has
/// underflowed to `0` contribute `0` (avoids `0·∞` at the support edges).
fn renyi_num(p: &impl Continuous, q: &impl Continuous, lo: f64, hi: f64, n: usize, a: f64) -> f64 {
    integ(lo, hi, n, |x| {
        let (pv, qv) = (p.pdf(x), q.pdf(x));
        if pv > 0.0 && qv > 0.0 {
            pv.powf(a) * qv.powf(1.0 - a)
        } else {
            0.0
        }
    })
    .ln()
        / (a - 1.0)
}

/// The three Rényi-limit identities, exact between the closed forms.
fn check_renyi_limits<D>(a: &D, b: &D)
where
    D: KlDivergence<Output = f64>
        + HellingerDistance<Output = f64>
        + ChiSquaredDivergence<Output = f64>
        + RenyiDivergence<Output = f64>,
{
    close(a.renyi_divergence(b, 1.0), a.kl_divergence(b), 1e-9);
    close(
        a.renyi_divergence(b, 0.5),
        -2.0 * (1.0 - a.hellinger_sq(b)).ln(),
        1e-9,
    );
    close(
        a.renyi_divergence(b, 2.0),
        (1.0 + a.chi_squared_divergence(b)).ln(),
        1e-9,
    );
}

// ----------------------------------------------------------------------------
// Gamma family: Gamma (new Rényi/χ²), ChiSquared, InverseGamma.
// ----------------------------------------------------------------------------

mod part1;
mod part2;
