use super::*;
use bayes_core::assert_close_scaled;

#[test]
fn wald_divergences_match_numeric() {
    let a = Wald::new(1.0, 3.0);
    let b = Wald::new(1.3, 2.0);
    let (lo, hi, n) = (1e-3, 30.0, 700_000);
    assert_close_scaled!(a.kl_divergence(&b), kl_num(&a, &b, lo, hi, n), 2e-3);
    assert_close_scaled!(a.hellinger_sq(&b), hellinger_num(&a, &b, lo, hi, n), 2e-3);
    assert_close_scaled!(
        a.chi_squared_divergence(&b),
        chi2_num(&a, &b, lo, hi, n),
        3e-3,
    );
    for al in [0.3, 0.7, 2.0] {
        assert_close_scaled!(
            a.renyi_divergence(&b, al),
            renyi_num(&a, &b, lo, hi, n, al),
            3e-3,
        );
    }
    check_renyi_limits(&a, &b);
    assert_close_scaled!(a.kl_divergence(&a), 0.0, 1e-10);
}

// ----------------------------------------------------------------------------
// Gumbel and Fréchet: KL only.
// ----------------------------------------------------------------------------

#[test]
fn gumbel_kl_matches_numeric() {
    let a = Gumbel::new(0.0, 1.0);
    let b = Gumbel::new(0.5, 1.5);
    assert_close_scaled!(
        a.kl_divergence(&b),
        kl_num(&a, &b, -20.0, 60.0, 600_000),
        2e-3,
    );
    assert_close_scaled!(a.kl_divergence(&a), 0.0, 1e-10);
    assert!(a.kl_divergence(&b) > 0.0);
}

#[test]
fn frechet_kl_matches_weibull_reparam_and_numeric() {
    let a = Frechet::new(3.0, 1.0);
    let b = Frechet::new(2.0, 1.5);
    // X ~ Fréchet(α, s) ⟺ 1/X ~ Weibull(shape α, scale 1/s); `Weibull::new` takes
    // (scale, shape).
    let wa = Weibull::new(1.0, 3.0);
    let wb = Weibull::new(1.0 / 1.5, 2.0);
    assert_close_scaled!(a.kl_divergence(&b), wa.kl_divergence(&wb), 1e-12);
    assert_close_scaled!(
        a.kl_divergence(&b),
        kl_num(&a, &b, 1e-3, 500.0, 800_000),
        2e-2,
    );
    assert_close_scaled!(a.kl_divergence(&a), 0.0, 1e-10);
}

// ----------------------------------------------------------------------------
// Uniform / DiscreteUniform: support-based closed forms.
// ----------------------------------------------------------------------------

#[test]
fn uniform_divergences_closed_form() {
    // [0,1] ⊆ [-0.5, 2]: widths 1 and 2.5, overlap 1.
    let a = Uniform::new(0.0, 1.0);
    let b = Uniform::new(-0.5, 2.0);
    assert_close_scaled!(a.kl_divergence(&b), 2.5_f64.ln(), 1e-12);
    assert_close_scaled!(a.chi_squared_divergence(&b), 1.5, 1e-12);
    assert_close_scaled!(a.hellinger_sq(&b), 1.0 - 1.0 / 2.5_f64.sqrt(), 1e-12);
    check_renyi_limits(&a, &b);
    assert_close_scaled!(a.kl_divergence(&a), 0.0, 1e-12);
    assert_close_scaled!(a.hellinger_sq(&a), 0.0, 1e-12);
    // Partial overlap: KL/χ² diverge, Hellinger stays finite.
    let c = Uniform::new(0.0_f64, 2.0);
    let d = Uniform::new(1.0_f64, 3.0);
    assert!(c.kl_divergence(&d).is_infinite());
    assert!(c.chi_squared_divergence(&d).is_infinite());
    assert_close_scaled!(c.hellinger_sq(&d), 0.5, 1e-12); // BC = 1/√(2·2)
    assert!(c.renyi_divergence(&d, 0.5).is_finite());
    assert!(c.renyi_divergence(&d, 2.0).is_infinite());
}

#[test]
fn discrete_uniform_divergences_closed_form() {
    // {2..5} (n=4) ⊆ {0..9} (n=10), overlap 4.
    let a = DiscreteUniform::new(2_i64, 5);
    let b = DiscreteUniform::new(0_i64, 9);
    assert_close_scaled!(a.kl_divergence(&b), (10.0_f64 / 4.0).ln(), 1e-12);
    assert_close_scaled!(a.chi_squared_divergence(&b), 10.0 / 4.0 - 1.0, 1e-12);
    assert_close_scaled!(a.hellinger_sq(&b), 1.0 - 4.0 / 40.0_f64.sqrt(), 1e-12);
    check_renyi_limits(&a, &b);
    assert_close_scaled!(a.kl_divergence(&a), 0.0, 1e-12);
    // Partial overlap {3,4} of {0..4} and {3..7}.
    let c = DiscreteUniform::new(0_i64, 4);
    let d = DiscreteUniform::new(3_i64, 7);
    assert!(c.kl_divergence(&d).is_infinite());
    assert_close_scaled!(c.hellinger_sq(&d), 0.6, 1e-12); // BC = 2/√(5·5)
}

// ----------------------------------------------------------------------------
// Binomial / NegativeBinomial: all four via the Bernoulli/series identities.
// ----------------------------------------------------------------------------

#[test]
fn binomial_divergences_scale_with_n_and_match_sums() {
    let n = 12u64;
    let a = Binomial::new(n, 0.3);
    let b = Binomial::new(n, 0.55);
    // KL = n · per-trial Bernoulli KL.
    let bern = Bernoulli::new(0.3).kl_divergence(&Bernoulli::new(0.55));
    assert_close_scaled!(a.kl_divergence(&b), n as f64 * bern, 1e-12);
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
    assert_close_scaled!(a.kl_divergence(&b), kl, 1e-9);
    let bc: f64 = (0..=n).map(|k| (a.pmf(k) * b.pmf(k)).sqrt()).sum();
    assert_close_scaled!(a.hellinger_sq(&b), 1.0 - bc, 1e-9);
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
    assert_close_scaled!(a.chi_squared_divergence(&b), chi, 1e-9);
    for al in [0.4, 2.0, 3.0] {
        let s: f64 = (0..=n)
            .map(|k| a.pmf(k).powf(al) * b.pmf(k).powf(1.0 - al))
            .sum();
        assert_close_scaled!(a.renyi_divergence(&b, al), s.ln() / (al - 1.0), 1e-9);
    }
    check_renyi_limits(&a, &b);
    assert_close_scaled!(a.kl_divergence(&a), 0.0, 1e-12);
    // Differing trial counts are out-of-family.
    assert!(a.kl_divergence(&Binomial::new(11, 0.3)).is_infinite());
    assert_eq!(a.hellinger_sq(&Binomial::new(11, 0.3)), 1.0);
}

#[test]
fn negative_binomial_divergences_match_sums() {
    let r = 5.0;
    let a = NegativeBinomial::new(r, 0.4);
    let b = NegativeBinomial::new(r, 0.6);
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
    assert_close_scaled!(a.kl_divergence(&b), kl, 1e-7);
    let bc: f64 = (0..=kmax).map(|k| (a.pmf(k) * b.pmf(k)).sqrt()).sum();
    assert_close_scaled!(a.hellinger_sq(&b), 1.0 - bc, 1e-7);
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
    assert_close_scaled!(a.chi_squared_divergence(&b), chi, 1e-6);
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
        assert_close_scaled!(a.renyi_divergence(&b, al), s.ln() / (al - 1.0), 1e-6);
    }
    check_renyi_limits(&a, &b);
    assert_close_scaled!(a.kl_divergence(&a), 0.0, 1e-10);
    // χ² diverges when (1−p₁)² ≥ 1−p₂.
    let c = NegativeBinomial::new(r, 0.1);
    let d = NegativeBinomial::new(r, 0.5);
    assert!(c.chi_squared_divergence(&d).is_infinite());
    // Differing successes are out-of-family.
    assert!(
        a.kl_divergence(&NegativeBinomial::new(4.0, 0.4))
            .is_infinite()
    );
}
