use super::*;

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
