use super::*;
use bayes_core::assert_close_scaled;

#[test]
fn gamma_renyi_and_chi_squared_match_numeric() {
    let a = Gamma::new(2.0, 1.5);
    let b = Gamma::new(3.0, 0.7);
    assert_close_scaled!(
        a.chi_squared_divergence(&b),
        chi2_num(&a, &b, 0.0, 80.0, 400_000),
        2e-3,
    );
    for al in [0.3, 0.7, 2.0] {
        assert_close_scaled!(
            a.renyi_divergence(&b, al),
            renyi_num(&a, &b, 0.0, 80.0, 400_000, al),
            2e-3,
        );
    }
    check_renyi_limits(&a, &b);
    assert_close_scaled!(a.chi_squared_divergence(&a), 0.0, 1e-10);
    // χ² diverges when the numerator outweighs the denominator tail (2a1 ≤ a2).
    assert!(
        Gamma::new(1.0_f64, 1.0)
            .chi_squared_divergence(&Gamma::new(3.0_f64, 1.0))
            .is_infinite()
    );
}

#[test]
fn chi_squared_dist_reuses_gamma_and_matches_numeric() {
    let a = ChiSquared::new(4.0);
    let b = ChiSquared::new(7.0);
    // χ²(k) ≡ Gamma(k/2, ½).
    let (ga, gb) = (Gamma::new(2.0, 0.5), Gamma::new(3.5, 0.5));
    assert_close_scaled!(a.kl_divergence(&b), ga.kl_divergence(&gb), 1e-12);
    assert_close_scaled!(a.hellinger_sq(&b), ga.hellinger_sq(&gb), 1e-12);
    assert_close_scaled!(
        a.chi_squared_divergence(&b),
        ga.chi_squared_divergence(&gb),
        1e-12,
    );
    assert_close_scaled!(
        a.renyi_divergence(&b, 1.7),
        ga.renyi_divergence(&gb, 1.7),
        1e-12,
    );
    // direct numeric KL
    assert_close_scaled!(
        a.kl_divergence(&b),
        kl_num(&a, &b, 0.0, 90.0, 400_000),
        2e-3,
    );
    check_renyi_limits(&a, &b);
    assert_close_scaled!(a.kl_divergence(&a), 0.0, 1e-10);
}

#[test]
fn inverse_gamma_reuses_gamma_and_matches_numeric() {
    let a = InverseGamma::new(5.0, 3.0);
    let b = InverseGamma::new(6.0, 4.0);
    // f-divergences are invariant under x ↦ 1/x, so IG(α,β) ≡ Gamma(α, rate β).
    let (ga, gb) = (Gamma::new(5.0, 3.0), Gamma::new(6.0, 4.0));
    assert_close_scaled!(a.kl_divergence(&b), ga.kl_divergence(&gb), 1e-12);
    assert_close_scaled!(a.hellinger_sq(&b), ga.hellinger_sq(&gb), 1e-12);
    assert_close_scaled!(
        a.chi_squared_divergence(&b),
        ga.chi_squared_divergence(&gb),
        1e-12,
    );
    assert_close_scaled!(
        a.renyi_divergence(&b, 1.7),
        ga.renyi_divergence(&gb, 1.7),
        1e-12,
    );
    // direct numeric KL and Hellinger over the inverse-gamma densities
    assert_close_scaled!(
        a.kl_divergence(&b),
        kl_num(&a, &b, 1e-3, 20.0, 400_000),
        3e-3,
    );
    assert_close_scaled!(
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
    let a = Normal::new(0.0, 1.0);
    let b = Normal::new(0.5, 1.5);
    assert_close_scaled!(
        a.chi_squared_divergence(&b),
        chi2_num(&a, &b, -40.0, 40.0, 400_000),
        2e-3,
    );
    assert_close_scaled!(
        a.renyi_divergence(&b, 2.0),
        (1.0 + a.chi_squared_divergence(&b)).ln(),
        1e-9,
    );
    // Diverges when s0² ≥ 2 s1².
    assert!(
        Normal::new(0.0_f64, 2.0)
            .chi_squared_divergence(&Normal::new(0.0_f64, 1.0))
            .is_infinite()
    );
}

#[test]
fn half_normal_divergences_match_numeric_and_zero_mean_normal() {
    let a = HalfNormal::new(1.0);
    let b = HalfNormal::new(1.5);
    assert_close_scaled!(
        a.kl_divergence(&b),
        kl_num(&a, &b, 0.0, 30.0, 400_000),
        2e-3,
    );
    assert_close_scaled!(
        a.hellinger_sq(&b),
        hellinger_num(&a, &b, 0.0, 30.0, 400_000),
        2e-3,
    );
    assert_close_scaled!(
        a.chi_squared_divergence(&b),
        chi2_num(&a, &b, 0.0, 30.0, 400_000),
        2e-3,
    );
    for al in [0.3, 0.7, 2.0] {
        assert_close_scaled!(
            a.renyi_divergence(&b, al),
            renyi_num(&a, &b, 0.0, 30.0, 400_000, al),
            2e-3,
        );
    }
    check_renyi_limits(&a, &b);
    assert_close_scaled!(a.kl_divergence(&a), 0.0, 1e-10);
    // Folding cancels: equals the zero-mean Gaussian on the scales.
    assert_close_scaled!(
        a.kl_divergence(&b),
        Normal::new(0.0, 1.0).kl_divergence(&Normal::new(0.0, 1.5)),
        1e-12,
    );
}

#[test]
fn lognormal_hellinger_and_chi_squared_match_underlying_normal() {
    let a = LogNormal::new(0.0, 1.0);
    let b = LogNormal::new(0.5, 1.2);
    let (na, nb) = (Normal::new(0.0, 1.0), Normal::new(0.5, 1.2));
    // H² and χ² are invariant under the monotone exp map.
    assert_close_scaled!(a.hellinger_sq(&b), na.hellinger_sq(&nb), 1e-12);
    assert_close_scaled!(
        a.chi_squared_divergence(&b),
        na.chi_squared_divergence(&nb),
        1e-12,
    );
    // direct numeric Hellinger over the log-normal densities
    assert_close_scaled!(
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
    let a = VonMises::new(0.0, 2.0);
    let b = VonMises::new(0.7, 1.5);
    let (lo, hi, n) = (-PI, PI, 300_000);
    assert_close_scaled!(a.kl_divergence(&b), kl_num(&a, &b, lo, hi, n), 2e-3);
    assert_close_scaled!(a.hellinger_sq(&b), hellinger_num(&a, &b, lo, hi, n), 2e-3);
    assert_close_scaled!(
        a.chi_squared_divergence(&b),
        chi2_num(&a, &b, lo, hi, n),
        2e-3,
    );
    for al in [0.3, 0.7, 2.0, 3.0] {
        assert_close_scaled!(
            a.renyi_divergence(&b, al),
            renyi_num(&a, &b, lo, hi, n, al),
            2e-3,
        );
    }
    check_renyi_limits(&a, &b);
    assert_close_scaled!(a.kl_divergence(&a), 0.0, 1e-10);
    assert_close_scaled!(a.hellinger_sq(&a), 0.0, 1e-10);
}

// ----------------------------------------------------------------------------
// Wald / inverse Gaussian: all four.
// ----------------------------------------------------------------------------
