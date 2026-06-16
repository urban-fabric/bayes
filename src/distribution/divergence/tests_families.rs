//! Numerical validation of the closed-form divergences for the extended set of
//! families. Every closed form is checked against a trapezoid integral (continuous)
//! or a finite pmf sum (discrete), plus the algebraic Rényi-limit identities
//! (`α→1` is KL, `α=½` is `−2 ln(1−H²)`, `α=2` is `ln(1+χ²)`).

use super::*;
use crate::distribution::{
    Bernoulli, Binomial, ChiSquared, Continuous, Discrete, DiscreteUniform, Frechet, Gamma, Gumbel,
    HalfNormal, InverseGamma, LogNormal, NegativeBinomial, Normal, Uniform, VonMises, Wald,
    Weibull,
};
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
    D: KlDivergence + HellingerDistance + ChiSquaredDivergence + RenyiDivergence,
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

#[test]
fn gamma_renyi_and_chi_squared_match_numeric() {
    let a = Gamma::new(2.0, 1.5).unwrap();
    let b = Gamma::new(3.0, 0.7).unwrap();
    close(
        a.chi_squared_divergence(&b),
        chi2_num(&a, &b, 0.0, 80.0, 400_000),
        2e-3,
    );
    for al in [0.3, 0.7, 2.0] {
        close(
            a.renyi_divergence(&b, al),
            renyi_num(&a, &b, 0.0, 80.0, 400_000, al),
            2e-3,
        );
    }
    check_renyi_limits(&a, &b);
    close(a.chi_squared_divergence(&a), 0.0, 1e-10);
    // χ² diverges when the numerator outweighs the denominator tail (2a1 ≤ a2).
    assert!(
        Gamma::new(1.0, 1.0)
            .unwrap()
            .chi_squared_divergence(&Gamma::new(3.0, 1.0).unwrap())
            .is_infinite()
    );
}

#[test]
fn chi_squared_dist_reuses_gamma_and_matches_numeric() {
    let a = ChiSquared::new(4.0).unwrap();
    let b = ChiSquared::new(7.0).unwrap();
    // χ²(k) ≡ Gamma(k/2, ½).
    let (ga, gb) = (Gamma::new(2.0, 0.5).unwrap(), Gamma::new(3.5, 0.5).unwrap());
    close(a.kl_divergence(&b), ga.kl_divergence(&gb), 1e-12);
    close(a.hellinger_sq(&b), ga.hellinger_sq(&gb), 1e-12);
    close(
        a.chi_squared_divergence(&b),
        ga.chi_squared_divergence(&gb),
        1e-12,
    );
    close(
        a.renyi_divergence(&b, 1.7),
        ga.renyi_divergence(&gb, 1.7),
        1e-12,
    );
    // direct numeric KL
    close(
        a.kl_divergence(&b),
        kl_num(&a, &b, 0.0, 90.0, 400_000),
        2e-3,
    );
    check_renyi_limits(&a, &b);
    close(a.kl_divergence(&a), 0.0, 1e-10);
}

#[test]
fn inverse_gamma_reuses_gamma_and_matches_numeric() {
    let a = InverseGamma::new(5.0, 3.0).unwrap();
    let b = InverseGamma::new(6.0, 4.0).unwrap();
    // f-divergences are invariant under x ↦ 1/x, so IG(α,β) ≡ Gamma(α, rate β).
    let (ga, gb) = (Gamma::new(5.0, 3.0).unwrap(), Gamma::new(6.0, 4.0).unwrap());
    close(a.kl_divergence(&b), ga.kl_divergence(&gb), 1e-12);
    close(a.hellinger_sq(&b), ga.hellinger_sq(&gb), 1e-12);
    close(
        a.chi_squared_divergence(&b),
        ga.chi_squared_divergence(&gb),
        1e-12,
    );
    close(
        a.renyi_divergence(&b, 1.7),
        ga.renyi_divergence(&gb, 1.7),
        1e-12,
    );
    // direct numeric KL and Hellinger over the inverse-gamma densities
    close(
        a.kl_divergence(&b),
        kl_num(&a, &b, 1e-3, 20.0, 400_000),
        3e-3,
    );
    close(
        a.hellinger_sq(&b),
        hellinger_num(&a, &b, 1e-3, 20.0, 400_000),
        3e-3,
    );
    check_renyi_limits(&a, &b);
}

// ----------------------------------------------------------------------------
// Gaussian family: Normal (new χ²), HalfNormal, LogNormal (new Hellinger/χ²).
// ----------------------------------------------------------------------------

#[test]
fn normal_chi_squared_matches_numeric_and_diverges() {
    let a = Normal::new(0.0, 1.0).unwrap();
    let b = Normal::new(0.5, 1.5).unwrap();
    close(
        a.chi_squared_divergence(&b),
        chi2_num(&a, &b, -40.0, 40.0, 400_000),
        2e-3,
    );
    close(
        a.renyi_divergence(&b, 2.0),
        (1.0 + a.chi_squared_divergence(&b)).ln(),
        1e-9,
    );
    // Diverges when s0² ≥ 2 s1².
    assert!(
        Normal::new(0.0, 2.0)
            .unwrap()
            .chi_squared_divergence(&Normal::new(0.0, 1.0).unwrap())
            .is_infinite()
    );
}

#[test]
fn half_normal_divergences_match_numeric_and_zero_mean_normal() {
    let a = HalfNormal::new(1.0).unwrap();
    let b = HalfNormal::new(1.5).unwrap();
    close(
        a.kl_divergence(&b),
        kl_num(&a, &b, 0.0, 30.0, 400_000),
        2e-3,
    );
    close(
        a.hellinger_sq(&b),
        hellinger_num(&a, &b, 0.0, 30.0, 400_000),
        2e-3,
    );
    close(
        a.chi_squared_divergence(&b),
        chi2_num(&a, &b, 0.0, 30.0, 400_000),
        2e-3,
    );
    for al in [0.3, 0.7, 2.0] {
        close(
            a.renyi_divergence(&b, al),
            renyi_num(&a, &b, 0.0, 30.0, 400_000, al),
            2e-3,
        );
    }
    check_renyi_limits(&a, &b);
    close(a.kl_divergence(&a), 0.0, 1e-10);
    // Folding cancels: equals the zero-mean Gaussian on the scales.
    close(
        a.kl_divergence(&b),
        Normal::new(0.0, 1.0)
            .unwrap()
            .kl_divergence(&Normal::new(0.0, 1.5).unwrap()),
        1e-12,
    );
}

#[test]
fn lognormal_hellinger_and_chi_squared_match_underlying_normal() {
    let a = LogNormal::new(0.0, 1.0).unwrap();
    let b = LogNormal::new(0.5, 1.2).unwrap();
    let (na, nb) = (
        Normal::new(0.0, 1.0).unwrap(),
        Normal::new(0.5, 1.2).unwrap(),
    );
    // H² and χ² are invariant under the monotone exp map.
    close(a.hellinger_sq(&b), na.hellinger_sq(&nb), 1e-12);
    close(
        a.chi_squared_divergence(&b),
        na.chi_squared_divergence(&nb),
        1e-12,
    );
    // direct numeric Hellinger over the log-normal densities
    close(
        a.hellinger_sq(&b),
        hellinger_num(&a, &b, 1e-4, 80.0, 800_000),
        3e-3,
    );
}

// ----------------------------------------------------------------------------
// Von Mises (directional): all four via Bessel I₀.
// ----------------------------------------------------------------------------

#[test]
fn von_mises_divergences_match_numeric() {
    let a = VonMises::new(0.0, 2.0).unwrap();
    let b = VonMises::new(0.7, 1.5).unwrap();
    let (lo, hi, n) = (-PI, PI, 300_000);
    close(a.kl_divergence(&b), kl_num(&a, &b, lo, hi, n), 2e-3);
    close(a.hellinger_sq(&b), hellinger_num(&a, &b, lo, hi, n), 2e-3);
    close(
        a.chi_squared_divergence(&b),
        chi2_num(&a, &b, lo, hi, n),
        2e-3,
    );
    for al in [0.3, 0.7, 2.0, 3.0] {
        close(
            a.renyi_divergence(&b, al),
            renyi_num(&a, &b, lo, hi, n, al),
            2e-3,
        );
    }
    check_renyi_limits(&a, &b);
    close(a.kl_divergence(&a), 0.0, 1e-10);
    close(a.hellinger_sq(&a), 0.0, 1e-10);
}

// ----------------------------------------------------------------------------
// Wald / inverse Gaussian: all four.
// ----------------------------------------------------------------------------

#[test]
fn wald_divergences_match_numeric() {
    let a = Wald::new(1.0, 3.0).unwrap();
    let b = Wald::new(1.3, 2.0).unwrap();
    let (lo, hi, n) = (1e-3, 30.0, 700_000);
    close(a.kl_divergence(&b), kl_num(&a, &b, lo, hi, n), 2e-3);
    close(a.hellinger_sq(&b), hellinger_num(&a, &b, lo, hi, n), 2e-3);
    close(
        a.chi_squared_divergence(&b),
        chi2_num(&a, &b, lo, hi, n),
        3e-3,
    );
    for al in [0.3, 0.7, 2.0] {
        close(
            a.renyi_divergence(&b, al),
            renyi_num(&a, &b, lo, hi, n, al),
            3e-3,
        );
    }
    check_renyi_limits(&a, &b);
    close(a.kl_divergence(&a), 0.0, 1e-10);
}

// ----------------------------------------------------------------------------
// Gumbel and Fréchet: KL only.
// ----------------------------------------------------------------------------

#[test]
fn gumbel_kl_matches_numeric() {
    let a = Gumbel::new(0.0, 1.0).unwrap();
    let b = Gumbel::new(0.5, 1.5).unwrap();
    close(
        a.kl_divergence(&b),
        kl_num(&a, &b, -20.0, 60.0, 600_000),
        2e-3,
    );
    close(a.kl_divergence(&a), 0.0, 1e-10);
    assert!(a.kl_divergence(&b) > 0.0);
}

#[test]
fn frechet_kl_matches_weibull_reparam_and_numeric() {
    let a = Frechet::new(3.0, 1.0).unwrap();
    let b = Frechet::new(2.0, 1.5).unwrap();
    // X ~ Fréchet(α, s) ⟺ 1/X ~ Weibull(shape α, scale 1/s); `Weibull::new` takes
    // (scale, shape).
    let wa = Weibull::new(1.0, 3.0).unwrap();
    let wb = Weibull::new(1.0 / 1.5, 2.0).unwrap();
    close(a.kl_divergence(&b), wa.kl_divergence(&wb), 1e-12);
    close(
        a.kl_divergence(&b),
        kl_num(&a, &b, 1e-3, 500.0, 800_000),
        2e-2,
    );
    close(a.kl_divergence(&a), 0.0, 1e-10);
}

// ----------------------------------------------------------------------------
// Uniform / DiscreteUniform: support-based closed forms.
// ----------------------------------------------------------------------------

#[test]
fn uniform_divergences_closed_form() {
    // [0,1] ⊆ [-0.5, 2]: widths 1 and 2.5, overlap 1.
    let a = Uniform::new(0.0, 1.0).unwrap();
    let b = Uniform::new(-0.5, 2.0).unwrap();
    close(a.kl_divergence(&b), 2.5_f64.ln(), 1e-12);
    close(a.chi_squared_divergence(&b), 1.5, 1e-12);
    close(a.hellinger_sq(&b), 1.0 - 1.0 / 2.5_f64.sqrt(), 1e-12);
    check_renyi_limits(&a, &b);
    close(a.kl_divergence(&a), 0.0, 1e-12);
    close(a.hellinger_sq(&a), 0.0, 1e-12);
    // Partial overlap: KL/χ² diverge, Hellinger stays finite.
    let c = Uniform::new(0.0, 2.0).unwrap();
    let d = Uniform::new(1.0, 3.0).unwrap();
    assert!(c.kl_divergence(&d).is_infinite());
    assert!(c.chi_squared_divergence(&d).is_infinite());
    close(c.hellinger_sq(&d), 0.5, 1e-12); // BC = 1/√(2·2)
    assert!(c.renyi_divergence(&d, 0.5).is_finite());
    assert!(c.renyi_divergence(&d, 2.0).is_infinite());
}

#[test]
fn discrete_uniform_divergences_closed_form() {
    // {2..5} (n=4) ⊆ {0..9} (n=10), overlap 4.
    let a = DiscreteUniform::new(2, 5).unwrap();
    let b = DiscreteUniform::new(0, 9).unwrap();
    close(a.kl_divergence(&b), (10.0_f64 / 4.0).ln(), 1e-12);
    close(a.chi_squared_divergence(&b), 10.0 / 4.0 - 1.0, 1e-12);
    close(a.hellinger_sq(&b), 1.0 - 4.0 / 40.0_f64.sqrt(), 1e-12);
    check_renyi_limits(&a, &b);
    close(a.kl_divergence(&a), 0.0, 1e-12);
    // Partial overlap {3,4} of {0..4} and {3..7}.
    let c = DiscreteUniform::new(0, 4).unwrap();
    let d = DiscreteUniform::new(3, 7).unwrap();
    assert!(c.kl_divergence(&d).is_infinite());
    close(c.hellinger_sq(&d), 0.6, 1e-12); // BC = 2/√(5·5)
}

// ----------------------------------------------------------------------------
// Binomial / NegativeBinomial: all four via the Bernoulli/series identities.
// ----------------------------------------------------------------------------

#[test]
fn binomial_divergences_scale_with_n_and_match_sums() {
    let n = 12u64;
    let a = Binomial::new(n, 0.3).unwrap();
    let b = Binomial::new(n, 0.55).unwrap();
    // KL = n · per-trial Bernoulli KL.
    let bern = Bernoulli::new(0.3)
        .unwrap()
        .kl_divergence(&Bernoulli::new(0.55).unwrap());
    close(a.kl_divergence(&b), n as f64 * bern, 1e-12);
    // direct pmf sums
    let kl: f64 = (0..=n)
        .map(|k| {
            let p = a.pmf(k);
            if p > 0.0 {
                p * (p / b.pmf(k)).ln()
            } else {
                0.0
            }
        })
        .sum();
    close(a.kl_divergence(&b), kl, 1e-9);
    let bc: f64 = (0..=n).map(|k| (a.pmf(k) * b.pmf(k)).sqrt()).sum();
    close(a.hellinger_sq(&b), 1.0 - bc, 1e-9);
    let chi: f64 = (0..=n)
        .map(|k| {
            let p = a.pmf(k);
            if b.pmf(k) > 0.0 {
                p * p / b.pmf(k)
            } else {
                0.0
            }
        })
        .sum::<f64>()
        - 1.0;
    close(a.chi_squared_divergence(&b), chi, 1e-9);
    for al in [0.4, 2.0, 3.0] {
        let s: f64 = (0..=n)
            .map(|k| a.pmf(k).powf(al) * b.pmf(k).powf(1.0 - al))
            .sum();
        close(a.renyi_divergence(&b, al), s.ln() / (al - 1.0), 1e-9);
    }
    check_renyi_limits(&a, &b);
    close(a.kl_divergence(&a), 0.0, 1e-12);
    // Differing trial counts are out-of-family.
    assert!(
        a.kl_divergence(&Binomial::new(11, 0.3).unwrap())
            .is_infinite()
    );
    assert_eq!(a.hellinger_sq(&Binomial::new(11, 0.3).unwrap()), 1.0);
}

#[test]
fn negative_binomial_divergences_match_sums() {
    let r = 5.0;
    let a = NegativeBinomial::new(r, 0.4).unwrap();
    let b = NegativeBinomial::new(r, 0.6).unwrap();
    // Stay in the well-conditioned range: past ~250 one pmf underflows to 0 while
    // the other is still positive, and the series have long since converged.
    let kmax = 250u64;
    let kl: f64 = (0..=kmax)
        .map(|k| {
            let p = a.pmf(k);
            if p > 0.0 {
                p * (p / b.pmf(k)).ln()
            } else {
                0.0
            }
        })
        .sum();
    close(a.kl_divergence(&b), kl, 1e-7);
    let bc: f64 = (0..=kmax).map(|k| (a.pmf(k) * b.pmf(k)).sqrt()).sum();
    close(a.hellinger_sq(&b), 1.0 - bc, 1e-7);
    // χ² is finite here: (1−p₁)² = 0.36 < 1−p₂ = 0.4.
    let chi: f64 = (0..=kmax)
        .map(|k| {
            let p = a.pmf(k);
            if b.pmf(k) > 0.0 {
                p * p / b.pmf(k)
            } else {
                0.0
            }
        })
        .sum::<f64>()
        - 1.0;
    close(a.chi_squared_divergence(&b), chi, 1e-6);
    for al in [0.4, 0.7, 2.0] {
        let s: f64 = (0..=kmax)
            .map(|k| {
                let (pk, qk) = (a.pmf(k), b.pmf(k));
                if pk > 0.0 && qk > 0.0 {
                    pk.powf(al) * qk.powf(1.0 - al)
                } else {
                    0.0
                }
            })
            .sum();
        close(a.renyi_divergence(&b, al), s.ln() / (al - 1.0), 1e-6);
    }
    check_renyi_limits(&a, &b);
    close(a.kl_divergence(&a), 0.0, 1e-10);
    // χ² diverges when (1−p₁)² ≥ 1−p₂.
    let c = NegativeBinomial::new(r, 0.1).unwrap();
    let d = NegativeBinomial::new(r, 0.5).unwrap();
    assert!(c.chi_squared_divergence(&d).is_infinite());
    // Differing successes are out-of-family.
    assert!(
        a.kl_divergence(&NegativeBinomial::new(4.0, 0.4).unwrap())
            .is_infinite()
    );
}
