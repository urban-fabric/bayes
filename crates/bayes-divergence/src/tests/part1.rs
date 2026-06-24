use super::*;

#[test]
fn self_divergence_is_zero() {
    close(
        Normal::new(1.0, 2.0).kl_divergence(&Normal::new(1.0, 2.0)),
        0.0,
        1e-12,
    );
    close(
        Exponential::new(3.0).kl_divergence(&Exponential::new(3.0)),
        0.0,
        1e-12,
    );
    close(
        Gamma::new(2.0, 1.5).kl_divergence(&Gamma::new(2.0, 1.5)),
        0.0,
        1e-10,
    );
    close(
        Beta::new(2.0, 3.0).kl_divergence(&Beta::new(2.0, 3.0)),
        0.0,
        1e-10,
    );
    close(
        Poisson::new(4.0).kl_divergence(&Poisson::new(4.0)),
        0.0,
        1e-12,
    );
}

#[test]
fn kl_is_non_negative_and_asymmetric() {
    let p = Normal::new(0.0, 1.0);
    let q = Normal::new(2.0, 3.0);
    assert!(p.kl_divergence(&q) > 0.0);
    assert!(q.kl_divergence(&p) > 0.0);
    assert_ne!(p.kl_divergence(&q), q.kl_divergence(&p));
}

#[test]
fn normal_kl_known_value() {
    close(
        Normal::new(0.0, 1.0).kl_divergence(&Normal::new(1.0, 1.0)),
        0.5,
        1e-12,
    );
}

#[test]
fn bernoulli_kl_matches_definition() {
    let kl = Bernoulli::new(0.3).kl_divergence(&Bernoulli::new(0.5));
    let expect = 0.3 * (0.3f64 / 0.5).ln() + 0.7 * (0.7f64 / 0.5).ln();
    close(kl, expect, 1e-12);
}

#[test]
fn reverse_kl_is_the_swapped_direction() {
    let p = Normal::new(0.0, 1.0);
    let q = Normal::new(2.0, 3.0);
    close(p.reverse_kl(&q), q.kl_divergence(&p), 1e-12);
    // KL is asymmetric, so reverse ≠ forward here.
    assert_ne!(p.reverse_kl(&q), p.kl_divergence(&q));
}

#[test]
fn jeffreys_is_symmetric_sum_of_both_directions_and_nonnegative() {
    for (p, q) in [
        (Normal::new(0.0, 1.0), Normal::new(2.0, 3.0)),
        (Normal::new(-1.0, 2.0), Normal::new(1.0, 0.5)),
    ] {
        let j = p.jeffreys(&q);
        close(j, p.kl_divergence(&q) + q.kl_divergence(&p), 1e-12);
        close(j, q.jeffreys(&p), 1e-12); // symmetric
        assert!(j > 0.0); // > 0 for distinct, ≥ 0 in general (Gibbs)
    }
    // Identical distributions: Jeffreys = 0.
    let b = Beta::new(2.0, 3.0);
    close(b.jeffreys(&Beta::new(2.0, 3.0)), 0.0, 1e-10);
}

#[test]
fn bernoulli_js_self_is_zero_symmetric_and_within_bounds() {
    let a = Bernoulli::new(0.3);
    let b = Bernoulli::new(0.9);
    close(a.js_divergence(&a), 0.0, 1e-12);
    close(a.js_divergence(&b), b.js_divergence(&a), 1e-12); // symmetric
    let js = a.js_divergence(&b);
    assert!(js > 0.0 && js <= std::f64::consts::LN_2 + 1e-12); // 0 ≤ JS ≤ ln 2
}

#[test]
fn bernoulli_js_matches_hand_expansion() {
    let (p, q) = (0.3, 0.5);
    let js = Bernoulli::new(p).js_divergence(&Bernoulli::new(q));
    let m1 = 0.5 * (p + q);
    let m0 = 0.5 * ((1.0 - p) + (1.0 - q));
    let expect = 0.5 * (p * (p / m1).ln() + (1.0 - p) * ((1.0 - p) / m0).ln())
        + 0.5 * (q * (q / m1).ln() + (1.0 - q) * ((1.0 - q) / m0).ln());
    close(js, expect, 1e-12);
}

#[test]
fn maximally_separated_bernoullis_saturate_the_log2_bound() {
    // JS(δ₀, δ₁) = ln 2, the upper bound (disjoint supports).
    let js = Bernoulli::new(0.0).js_divergence(&Bernoulli::new(1.0));
    close(js, std::f64::consts::LN_2, 1e-12);
}

#[test]
fn categorical_js_self_zero_symmetric_and_bounded() {
    let a = Categorical::new(&[0.2, 0.3, 0.5]).unwrap();
    let b = Categorical::new(&[0.6, 0.1, 0.3]).unwrap();
    close(a.js_divergence(&a), 0.0, 1e-12);
    close(a.js_divergence(&b), b.js_divergence(&a), 1e-12);
    let js = a.js_divergence(&b);
    assert!(js > 0.0 && js <= std::f64::consts::LN_2 + 1e-12);
}

#[test]
fn categorical_js_matches_bernoulli_for_two_categories() {
    // A 2-category Categorical is a Bernoulli; the two paths must agree.
    let cat = Categorical::new(&[0.7, 0.3])
        .unwrap()
        .js_divergence(&Categorical::new(&[0.4, 0.6]).unwrap());
    // Bernoulli p() is P(success); category 0 here plays "success".
    let bern = Bernoulli::new(0.7).js_divergence(&Bernoulli::new(0.4));
    close(cat, bern, 1e-12);
}

#[test]
fn hellinger_self_is_zero_and_symmetric() {
    macro_rules! check {
        ($a:expr, $b:expr) => {{
            close($a.hellinger_sq(&$a), 0.0, 1e-12);
            close($a.hellinger_sq(&$b), $b.hellinger_sq(&$a), 1e-12);
            let h2 = $a.hellinger_sq(&$b);
            assert!((0.0..=1.0).contains(&h2), "H² out of [0,1]: {h2}");
        }};
    }
    check!(Normal::new(0.0, 1.0), Normal::new(1.5, 2.0));
    check!(Exponential::new(2.0), Exponential::new(0.5));
    check!(Gamma::new(2.0, 1.5), Gamma::new(3.0, 0.7));
    check!(Beta::new(2.0, 3.0), Beta::new(4.0, 2.0));
    check!(Poisson::new(3.0), Poisson::new(7.0));
    check!(Bernoulli::new(0.3), Bernoulli::new(0.8));
}

#[test]
fn bernoulli_hellinger_matches_closed_form() {
    let (p, q) = (0.3, 0.8);
    let h2 = Bernoulli::new(p).hellinger_sq(&Bernoulli::new(q));
    let bc = (p * q).sqrt() + ((1.0 - p) * (1.0 - q)).sqrt();
    close(h2, 1.0 - bc, 1e-12);
}

#[test]
fn poisson_hellinger_matches_pmf_sum() {
    let (a, b) = (Poisson::new(3.0), Poisson::new(7.0));
    let bc: f64 = (0_u64..200).map(|k| (a.pmf(k) * b.pmf(k)).sqrt()).sum();
    close(a.hellinger_sq(&b), 1.0 - bc, 1e-9);
}

#[test]
fn continuous_hellinger_matches_numeric_integral() {
    let n1 = Normal::new(0.0, 1.0);
    let n2 = Normal::new(1.5, 2.0);
    close(
        n1.hellinger_sq(&n2),
        1.0 - bc_integral(|x| n1.pdf(x), |x| n2.pdf(x), -40.0, 40.0),
        2e-3,
    );
    let e1 = Exponential::new(2.0);
    let e2 = Exponential::new(0.5);
    close(
        e1.hellinger_sq(&e2),
        1.0 - bc_integral(|x| e1.pdf(x), |x| e2.pdf(x), 0.0, 80.0),
        2e-3,
    );
    let g1 = Gamma::new(2.0, 1.5);
    let g2 = Gamma::new(3.0, 0.7);
    close(
        g1.hellinger_sq(&g2),
        1.0 - bc_integral(|x| g1.pdf(x), |x| g2.pdf(x), 0.0, 80.0),
        2e-3,
    );
    let b1 = Beta::new(2.0, 3.0);
    let b2 = Beta::new(4.0, 2.0);
    close(
        b1.hellinger_sq(&b2),
        1.0 - bc_integral(|x| b1.pdf(x), |x| b2.pdf(x), 0.0, 1.0),
        2e-3,
    );
}

#[test]
fn total_variation_bernoulli_categorical_symmetric_bounded() {
    let a = Bernoulli::new(0.3);
    let b = Bernoulli::new(0.8);
    close(a.total_variation(&b), 0.5, 1e-12); // |0.3 − 0.8|
    close(a.total_variation(&a), 0.0, 1e-12);
    close(a.total_variation(&b), b.total_variation(&a), 1e-12);

    let c = Categorical::new(&[0.2, 0.3, 0.5]).unwrap();
    let d = Categorical::new(&[0.5, 0.1, 0.4]).unwrap();
    let tv = c.total_variation(&d);
    close(tv, 0.5 * (0.3 + 0.2 + 0.1), 1e-12);
    assert!((0.0..=1.0).contains(&tv));
    // 2-category Categorical agrees with Bernoulli.
    let c2 = Categorical::new(&[0.7, 0.3]).unwrap();
    let d2 = Categorical::new(&[0.4, 0.6]).unwrap();
    close(
        c2.total_variation(&d2),
        Bernoulli::new(0.7).total_variation(&Bernoulli::new(0.4)),
        1e-12,
    );
}

#[test]
fn pinsker_and_hellinger_comparisons_hold() {
    // The proved inequalities, checked numerically on Bernoulli pairs:
    // TV² ≤ KL (Pinsker), H² ≤ TV, and H² ≤ ½ KL.
    for (p, q) in [(0.3, 0.8), (0.1, 0.2), (0.5, 0.95)] {
        let a = Bernoulli::new(p);
        let b = Bernoulli::new(q);
        let tv = a.total_variation(&b);
        let kl = a.kl_divergence(&b);
        let h2 = a.hellinger_sq(&b);
        assert!(tv * tv <= kl + 1e-12, "Pinsker TV²≤KL failed: {p},{q}");
        assert!(h2 <= tv + 1e-12, "H²≤TV failed: {p},{q}");
        assert!(h2 <= 0.5 * kl + 1e-12, "H²≤½KL failed: {p},{q}");
    }
}

#[test]
fn chi_squared_self_zero_and_closed_forms() {
    // Self-divergence is zero.
    close(
        Bernoulli::new(0.3).chi_squared_divergence(&Bernoulli::new(0.3)),
        0.0,
        1e-12,
    );
    close(
        Poisson::new(4.0).chi_squared_divergence(&Poisson::new(4.0)),
        0.0,
        1e-12,
    );
    // Bernoulli closed form vs the two-term definition.
    let (p, q) = (0.3, 0.8);
    let chi = Bernoulli::new(p).chi_squared_divergence(&Bernoulli::new(q));
    let def = (p - q).powi(2) / q + ((1.0 - p) - (1.0 - q)).powi(2) / (1.0 - q);
    close(chi, def, 1e-12);
}

#[test]
fn poisson_chi_squared_matches_pmf_sum() {
    let (a, b) = (Poisson::new(3.0), Poisson::new(5.0));
    // χ² = Σ pₖ²/qₖ − 1.
    let s: f64 = (0_u64..400)
        .map(|k| {
            let (pk, qk) = (a.pmf(k), b.pmf(k));
            if qk > 0.0 { pk * pk / qk } else { 0.0 }
        })
        .sum();
    close(a.chi_squared_divergence(&b), s - 1.0, 1e-9);
}
