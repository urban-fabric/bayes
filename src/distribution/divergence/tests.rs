use super::*;
use crate::distribution::{
    Bernoulli, Beta, Categorical, Cauchy, Continuous, Dirichlet, Discrete, Exponential, Gamma,
    Geometric, Laplace, LogNormal, Moments, Multinomial, Normal, Pareto, Poisson, Rayleigh, Weibull,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

/// Numerical Bhattacharyya coefficient `∫ √(f·g)` by the trapezoid rule, for
/// validating the closed-form Hellinger affinities.
fn bc_integral(f: impl Fn(f64) -> f64, g: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let n = 100_000usize;
    let h = (hi - lo) / n as f64;
    let mut s = 0.0;
    for k in 0..=n {
        let x = lo + k as f64 * h;
        let w = if k == 0 || k == n { 0.5 } else { 1.0 };
        s += w * (f(x) * g(x)).max(0.0).sqrt();
    }
    s * h
}

#[test]
fn self_divergence_is_zero() {
    close(
        Normal::new(1.0, 2.0)
            .unwrap()
            .kl_divergence(&Normal::new(1.0, 2.0).unwrap()),
        0.0,
        1e-12,
    );
    close(
        Exponential::new(3.0)
            .unwrap()
            .kl_divergence(&Exponential::new(3.0).unwrap()),
        0.0,
        1e-12,
    );
    close(
        Gamma::new(2.0, 1.5)
            .unwrap()
            .kl_divergence(&Gamma::new(2.0, 1.5).unwrap()),
        0.0,
        1e-10,
    );
    close(
        Beta::new(2.0, 3.0)
            .unwrap()
            .kl_divergence(&Beta::new(2.0, 3.0).unwrap()),
        0.0,
        1e-10,
    );
    close(
        Poisson::new(4.0)
            .unwrap()
            .kl_divergence(&Poisson::new(4.0).unwrap()),
        0.0,
        1e-12,
    );
}

#[test]
fn kl_is_non_negative_and_asymmetric() {
    let p = Normal::new(0.0, 1.0).unwrap();
    let q = Normal::new(2.0, 3.0).unwrap();
    assert!(p.kl_divergence(&q) > 0.0);
    assert!(q.kl_divergence(&p) > 0.0);
    assert_ne!(p.kl_divergence(&q), q.kl_divergence(&p));
}

#[test]
fn normal_kl_known_value() {
    close(
        Normal::new(0.0, 1.0)
            .unwrap()
            .kl_divergence(&Normal::new(1.0, 1.0).unwrap()),
        0.5,
        1e-12,
    );
}

#[test]
fn bernoulli_kl_matches_definition() {
    let kl = Bernoulli::new(0.3)
        .unwrap()
        .kl_divergence(&Bernoulli::new(0.5).unwrap());
    let expect = 0.3 * (0.3f64 / 0.5).ln() + 0.7 * (0.7f64 / 0.5).ln();
    close(kl, expect, 1e-12);
}

#[test]
fn reverse_kl_is_the_swapped_direction() {
    let p = Normal::new(0.0, 1.0).unwrap();
    let q = Normal::new(2.0, 3.0).unwrap();
    close(p.reverse_kl(&q), q.kl_divergence(&p), 1e-12);
    // KL is asymmetric, so reverse ≠ forward here.
    assert_ne!(p.reverse_kl(&q), p.kl_divergence(&q));
}

#[test]
fn jeffreys_is_symmetric_sum_of_both_directions_and_nonnegative() {
    for (p, q) in [
        (
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(2.0, 3.0).unwrap(),
        ),
        (
            Normal::new(-1.0, 2.0).unwrap(),
            Normal::new(1.0, 0.5).unwrap(),
        ),
    ] {
        let j = p.jeffreys(&q);
        close(j, p.kl_divergence(&q) + q.kl_divergence(&p), 1e-12);
        close(j, q.jeffreys(&p), 1e-12); // symmetric
        assert!(j > 0.0); // > 0 for distinct, ≥ 0 in general (Gibbs)
    }
    // Identical distributions: Jeffreys = 0.
    let b = Beta::new(2.0, 3.0).unwrap();
    close(b.jeffreys(&Beta::new(2.0, 3.0).unwrap()), 0.0, 1e-10);
}

#[test]
fn bernoulli_js_self_is_zero_symmetric_and_within_bounds() {
    let a = Bernoulli::new(0.3).unwrap();
    let b = Bernoulli::new(0.9).unwrap();
    close(a.js_divergence(&a), 0.0, 1e-12);
    close(a.js_divergence(&b), b.js_divergence(&a), 1e-12); // symmetric
    let js = a.js_divergence(&b);
    assert!(js > 0.0 && js <= std::f64::consts::LN_2 + 1e-12); // 0 ≤ JS ≤ ln 2
}

#[test]
fn bernoulli_js_matches_hand_expansion() {
    let (p, q) = (0.3, 0.5);
    let js = Bernoulli::new(p)
        .unwrap()
        .js_divergence(&Bernoulli::new(q).unwrap());
    let m1 = 0.5 * (p + q);
    let m0 = 0.5 * ((1.0 - p) + (1.0 - q));
    let expect = 0.5 * (p * (p / m1).ln() + (1.0 - p) * ((1.0 - p) / m0).ln())
        + 0.5 * (q * (q / m1).ln() + (1.0 - q) * ((1.0 - q) / m0).ln());
    close(js, expect, 1e-12);
}

#[test]
fn maximally_separated_bernoullis_saturate_the_log2_bound() {
    // JS(δ₀, δ₁) = ln 2, the upper bound (disjoint supports).
    let js = Bernoulli::new(0.0)
        .unwrap()
        .js_divergence(&Bernoulli::new(1.0).unwrap());
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
    let bern = Bernoulli::new(0.7)
        .unwrap()
        .js_divergence(&Bernoulli::new(0.4).unwrap());
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
    check!(
        Normal::new(0.0, 1.0).unwrap(),
        Normal::new(1.5, 2.0).unwrap()
    );
    check!(
        Exponential::new(2.0).unwrap(),
        Exponential::new(0.5).unwrap()
    );
    check!(Gamma::new(2.0, 1.5).unwrap(), Gamma::new(3.0, 0.7).unwrap());
    check!(Beta::new(2.0, 3.0).unwrap(), Beta::new(4.0, 2.0).unwrap());
    check!(Poisson::new(3.0).unwrap(), Poisson::new(7.0).unwrap());
    check!(Bernoulli::new(0.3).unwrap(), Bernoulli::new(0.8).unwrap());
}

#[test]
fn bernoulli_hellinger_matches_closed_form() {
    let (p, q) = (0.3, 0.8);
    let h2 = Bernoulli::new(p)
        .unwrap()
        .hellinger_sq(&Bernoulli::new(q).unwrap());
    let bc = (p * q).sqrt() + ((1.0 - p) * (1.0 - q)).sqrt();
    close(h2, 1.0 - bc, 1e-12);
}

#[test]
fn poisson_hellinger_matches_pmf_sum() {
    let (a, b) = (Poisson::new(3.0).unwrap(), Poisson::new(7.0).unwrap());
    let bc: f64 = (0..200).map(|k| (a.pmf(k) * b.pmf(k)).sqrt()).sum();
    close(a.hellinger_sq(&b), 1.0 - bc, 1e-9);
}

#[test]
fn continuous_hellinger_matches_numeric_integral() {
    let n1 = Normal::new(0.0, 1.0).unwrap();
    let n2 = Normal::new(1.5, 2.0).unwrap();
    close(
        n1.hellinger_sq(&n2),
        1.0 - bc_integral(|x| n1.pdf(x), |x| n2.pdf(x), -40.0, 40.0),
        2e-3,
    );
    let e1 = Exponential::new(2.0).unwrap();
    let e2 = Exponential::new(0.5).unwrap();
    close(
        e1.hellinger_sq(&e2),
        1.0 - bc_integral(|x| e1.pdf(x), |x| e2.pdf(x), 0.0, 80.0),
        2e-3,
    );
    let g1 = Gamma::new(2.0, 1.5).unwrap();
    let g2 = Gamma::new(3.0, 0.7).unwrap();
    close(
        g1.hellinger_sq(&g2),
        1.0 - bc_integral(|x| g1.pdf(x), |x| g2.pdf(x), 0.0, 80.0),
        2e-3,
    );
    let b1 = Beta::new(2.0, 3.0).unwrap();
    let b2 = Beta::new(4.0, 2.0).unwrap();
    close(
        b1.hellinger_sq(&b2),
        1.0 - bc_integral(|x| b1.pdf(x), |x| b2.pdf(x), 0.0, 1.0),
        2e-3,
    );
}

#[test]
fn total_variation_bernoulli_categorical_symmetric_bounded() {
    let a = Bernoulli::new(0.3).unwrap();
    let b = Bernoulli::new(0.8).unwrap();
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
        Bernoulli::new(0.7)
            .unwrap()
            .total_variation(&Bernoulli::new(0.4).unwrap()),
        1e-12,
    );
}

#[test]
fn pinsker_and_hellinger_comparisons_hold() {
    // The proved inequalities, checked numerically on Bernoulli pairs:
    // TV² ≤ KL (Pinsker), H² ≤ TV, and H² ≤ ½ KL.
    for (p, q) in [(0.3, 0.8), (0.1, 0.2), (0.5, 0.95)] {
        let a = Bernoulli::new(p).unwrap();
        let b = Bernoulli::new(q).unwrap();
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
        Bernoulli::new(0.3)
            .unwrap()
            .chi_squared_divergence(&Bernoulli::new(0.3).unwrap()),
        0.0,
        1e-12,
    );
    close(
        Poisson::new(4.0)
            .unwrap()
            .chi_squared_divergence(&Poisson::new(4.0).unwrap()),
        0.0,
        1e-12,
    );
    // Bernoulli closed form vs the two-term definition.
    let (p, q) = (0.3, 0.8);
    let chi = Bernoulli::new(p)
        .unwrap()
        .chi_squared_divergence(&Bernoulli::new(q).unwrap());
    let def = (p - q).powi(2) / q + ((1.0 - p) - (1.0 - q)).powi(2) / (1.0 - q);
    close(chi, def, 1e-12);
}

#[test]
fn poisson_chi_squared_matches_pmf_sum() {
    let (a, b) = (Poisson::new(3.0).unwrap(), Poisson::new(5.0).unwrap());
    // χ² = Σ pₖ²/qₖ − 1.
    let s: f64 = (0..400)
        .map(|k| {
            let (pk, qk) = (a.pmf(k), b.pmf(k));
            if qk > 0.0 { pk * pk / qk } else { 0.0 }
        })
        .sum();
    close(a.chi_squared_divergence(&b), s - 1.0, 1e-9);
}

#[test]
fn categorical_chi_squared_matches_definition_and_handles_zero_support() {
    let c = Categorical::new(&[0.2, 0.3, 0.5]).unwrap();
    let d = Categorical::new(&[0.5, 0.1, 0.4]).unwrap();
    let def: f64 = [(0.2_f64, 0.5_f64), (0.3, 0.1), (0.5, 0.4)]
        .iter()
        .map(|(a, b)| (a - b).powi(2) / b)
        .sum();
    close(c.chi_squared_divergence(&d), def, 1e-12);
    // q has a zero where p does not → +∞.
    let p = Categorical::new(&[0.5, 0.5]).unwrap();
    let q = Categorical::new(&[1.0, 0.0]).unwrap();
    assert!(p.chi_squared_divergence(&q).is_infinite());
}

#[test]
fn exponential_chi_squared_converges_only_for_light_enough_denominator() {
    // Convergent case 2λ₁ > λ₂: validate against the numeric integral of p²/q.
    let a = Exponential::new(2.0).unwrap();
    let b = Exponential::new(1.0).unwrap();
    let integral = {
        let n = 200_000usize;
        let (lo, hi) = (0.0, 100.0);
        let h = (hi - lo) / n as f64;
        (0..=n)
            .map(|k| {
                let x = lo + k as f64 * h;
                let w = if k == 0 || k == n { 0.5 } else { 1.0 };
                w * a.pdf(x) * a.pdf(x) / b.pdf(x)
            })
            .sum::<f64>()
            * h
    };
    close(a.chi_squared_divergence(&b), integral - 1.0, 2e-3);
    // Divergent case 2λ₁ ≤ λ₂: heavier-tailed numerator → +∞.
    let c = Exponential::new(1.0).unwrap();
    let d = Exponential::new(3.0).unwrap();
    assert!(c.chi_squared_divergence(&d).is_infinite());
}

#[test]
fn kl_le_chi_squared_holds() {
    // The proved bound KL ≤ χ², checked on Bernoulli and Poisson pairs.
    for (p, q) in [(0.3, 0.8), (0.1, 0.2), (0.6, 0.55)] {
        let a = Bernoulli::new(p).unwrap();
        let b = Bernoulli::new(q).unwrap();
        assert!(
            a.kl_divergence(&b) <= a.chi_squared_divergence(&b) + 1e-12,
            "KL≤χ² (Bernoulli) failed: {p},{q}"
        );
    }
    let a = Poisson::new(3.0).unwrap();
    let b = Poisson::new(4.0).unwrap();
    assert!(a.kl_divergence(&b) <= a.chi_squared_divergence(&b) + 1e-12);
}

#[test]
fn renyi_self_zero_and_limits_tie_to_the_other_divergences() {
    let a = Bernoulli::new(0.3).unwrap();
    let b = Bernoulli::new(0.8).unwrap();
    // Self-divergence is zero at any order.
    for alpha in [0.5, 1.0, 2.0, 3.0] {
        close(a.renyi_divergence(&a, alpha), 0.0, 1e-12);
    }
    // α = 1 is exactly KL; just below/above α = 1 approaches it.
    close(a.renyi_divergence(&b, 1.0), a.kl_divergence(&b), 1e-12);
    close(
        a.renyi_divergence(&b, 1.0 + 1e-6),
        a.kl_divergence(&b),
        1e-4,
    );
    // α = ½ is −2·ln(1 − H²) (Hellinger).
    close(
        a.renyi_divergence(&b, 0.5),
        -2.0 * (1.0 - a.hellinger_sq(&b)).ln(),
        1e-12,
    );
    // α = 2 is ln(1 + χ²).
    close(
        a.renyi_divergence(&b, 2.0),
        (1.0 + a.chi_squared_divergence(&b)).ln(),
        1e-12,
    );
    // Monotone non-decreasing in α.
    assert!(a.renyi_divergence(&b, 0.5) <= a.renyi_divergence(&b, 2.0) + 1e-12);
}

#[test]
fn categorical_renyi_matches_bernoulli_for_two_categories() {
    for alpha in [0.5, 1.0, 2.0] {
        let cat = Categorical::new(&[0.7, 0.3])
            .unwrap()
            .renyi_divergence(&Categorical::new(&[0.4, 0.6]).unwrap(), alpha);
        let bern = Bernoulli::new(0.7)
            .unwrap()
            .renyi_divergence(&Bernoulli::new(0.4).unwrap(), alpha);
        close(cat, bern, 1e-12);
    }
}

#[test]
fn gaussian_renyi_matches_its_limits_and_the_integral() {
    let p = Normal::new(0.0, 1.0).unwrap();
    let q = Normal::new(1.5, 2.0).unwrap();
    // Self-divergence is zero at every order.
    for alpha in [0.3, 0.5, 1.0, 2.0, 3.0] {
        close(p.renyi_divergence(&p, alpha), 0.0, 1e-12);
    }
    // α = 1 is exactly KL.
    close(p.renyi_divergence(&q, 1.0), p.kl_divergence(&q), 1e-12);
    // α = ½ is −2·ln(1 − H²) (Hellinger), reusing the closed-form affinity.
    close(
        p.renyi_divergence(&q, 0.5),
        -2.0 * (1.0 - p.hellinger_sq(&q)).ln(),
        1e-12,
    );
    // Direct check against D_α = ln(∫ pᵅ q¹⁻ᵅ)/(α−1) by numeric integration.
    let renyi_integral = |alpha: f64| {
        let n = 200_000usize;
        let (lo, hi) = (-60.0, 60.0);
        let h = (hi - lo) / n as f64;
        let s: f64 = (0..=n)
            .map(|k| {
                let x = lo + k as f64 * h;
                let w = if k == 0 || k == n { 0.5 } else { 1.0 };
                w * p.pdf(x).powf(alpha) * q.pdf(x).powf(1.0 - alpha)
            })
            .sum::<f64>()
            * h;
        s.ln() / (alpha - 1.0)
    };
    for alpha in [0.3, 0.7, 2.0] {
        close(p.renyi_divergence(&q, alpha), renyi_integral(alpha), 2e-3);
    }
    // Monotone non-decreasing in α.
    assert!(p.renyi_divergence(&q, 0.5) <= p.renyi_divergence(&q, 2.0) + 1e-12);
    // Diverges when the order-α power is not integrable (σ_α² ≤ 0):
    // α = 2, σ₀ = 3, σ₁ = 1 ⇒ 2·1 − 1·9 < 0.
    assert!(
        Normal::new(0.0, 3.0)
            .unwrap()
            .renyi_divergence(&Normal::new(0.0, 1.0).unwrap(), 2.0)
            .is_infinite()
    );
    // LogNormal reuses the Normal formula on (μ, σ).
    for alpha in [0.4, 2.0] {
        close(
            LogNormal::new(0.0, 1.0)
                .unwrap()
                .renyi_divergence(&LogNormal::new(1.5, 2.0).unwrap(), alpha),
            p.renyi_divergence(&q, alpha),
            1e-12,
        );
    }
}

#[test]
fn mutual_information_zero_iff_independent_and_nonneg() {
    // Independent joint p(i,k) = pX(i)·pY(k): I = 0.
    let indep = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
    close(mutual_information(&indep), 0.0, 1e-12);
    // Perfectly correlated (diagonal) uniform: I = ln 2 (= marginal entropy).
    let correlated = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
    close(
        mutual_information(&correlated),
        std::f64::consts::LN_2,
        1e-12,
    );
    // A generic joint: I ≥ 0, and matches KL(joint ‖ pX⊗pY) by hand.
    let j = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
    let mi = mutual_information(&j);
    let (px, py) = ([0.3, 0.7], [0.4, 0.6]);
    let expect: f64 = [(0, 0, 0.1), (0, 1, 0.2), (1, 0, 0.3), (1, 1, 0.4)]
        .iter()
        .map(|&(i, k, v): &(usize, usize, f64)| v * (v / (px[i] * py[k])).ln())
        .sum();
    close(mi, expect, 1e-12);
    assert!(mi > 0.0);
}

// --- broadened KL coverage beyond the base six families ---

#[test]
fn lognormal_kl_equals_underlying_normal_kl() {
    // KL is invariant under the log reparametrization, so LN(0,1)‖LN(1,1)
    // matches N(0,1)‖N(1,1) = 0.5.
    close(
        LogNormal::new(0.0, 1.0)
            .unwrap()
            .kl_divergence(&LogNormal::new(1.0, 1.0).unwrap()),
        0.5,
        1e-12,
    );
}

#[test]
fn laplace_and_geometric_kl_known_values() {
    // Laplace, same location: ln(b₂/b₁) + b₁/b₂ − 1.
    close(
        Laplace::new(0.0, 1.0)
            .unwrap()
            .kl_divergence(&Laplace::new(0.0, 2.0).unwrap()),
        2.0_f64.ln() - 0.5,
        1e-12,
    );
    // Geometric: ln(p₁/p₂) + ((1−p₁)/p₁)·ln((1−p₁)/(1−p₂)) = ln(4/3).
    close(
        Geometric::new(0.5)
            .unwrap()
            .kl_divergence(&Geometric::new(0.25).unwrap()),
        (4.0_f64 / 3.0).ln(),
        1e-12,
    );
}

#[test]
fn categorical_multinomial_dirichlet_kl() {
    let p = Categorical::new(&[0.5, 0.5]).unwrap();
    let q = Categorical::new(&[0.25, 0.75]).unwrap();
    close(p.kl_divergence(&q), 0.5 * (4.0_f64 / 3.0).ln(), 1e-12);
    close(p.kl_divergence(&p), 0.0, 1e-12);
    assert!(
        p.kl_divergence(&Categorical::new(&[0.3, 0.3, 0.4]).unwrap())
            .is_infinite()
    );

    // Multinomial KL is `n ×` the categorical KL for matched trial counts.
    let m = Multinomial::new(10, &[0.5, 0.5]).unwrap();
    let mq = Multinomial::new(10, &[0.25, 0.75]).unwrap();
    close(m.kl_divergence(&mq), 10.0 * p.kl_divergence(&q), 1e-12);
    assert!(
        m.kl_divergence(&Multinomial::new(7, &[0.5, 0.5]).unwrap())
            .is_infinite()
    );

    // Dirichlet: nonneg, self-zero, asymmetric.
    let da = Dirichlet::new(vec![2.0, 3.0, 4.0]).unwrap();
    let db = Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap();
    close(da.kl_divergence(&da), 0.0, 1e-12);
    assert!(da.kl_divergence(&db) > 0.0);
    assert_ne!(da.kl_divergence(&db), db.kl_divergence(&da));
}

#[test]
fn cauchy_rayleigh_weibull_pareto_kl() {
    // Cauchy: ln[((γ₁+γ₂)² + Δ²)/(4γ₁γ₂)]; here ln(9/8).
    let c = Cauchy::new(0.0, 1.0).unwrap();
    close(
        c.kl_divergence(&Cauchy::new(0.0, 2.0).unwrap()),
        (9.0_f64 / 8.0).ln(),
        1e-12,
    );
    close(c.kl_divergence(&c), 0.0, 1e-12);

    // Rayleigh: 2 ln(σ₂/σ₁) + (σ₁/σ₂)² − 1.
    close(
        Rayleigh::new(1.0)
            .unwrap()
            .kl_divergence(&Rayleigh::new(2.0).unwrap()),
        2.0 * 2.0_f64.ln() - 0.75,
        1e-12,
    );

    // Weibull(scale, shape=1) is Exponential(rate 1/scale); the KLs agree.
    close(
        Weibull::new(2.0, 1.0)
            .unwrap()
            .kl_divergence(&Weibull::new(3.0, 1.0).unwrap()),
        Exponential::new(0.5)
            .unwrap()
            .kl_divergence(&Exponential::new(1.0 / 3.0).unwrap()),
        1e-12,
    );

    // Pareto: finite only when xₘ₁ ≥ xₘ₂.
    let p = Pareto::new(1.0, 3.0).unwrap();
    close(
        p.kl_divergence(&Pareto::new(1.0, 2.0).unwrap()),
        1.5_f64.ln() + 2.0 / 3.0 - 1.0,
        1e-12,
    );
    assert!(
        Pareto::new(2.0, 2.0)
            .unwrap()
            .kl_divergence(&Pareto::new(3.0, 2.0).unwrap())
            .is_infinite()
    );
}

#[test]
fn cross_entropy_decomposes_and_handles_missing_entropy() {
    // H(p, q) = H(p) + D_KL(p ‖ q), equal to the closed form for normals.
    let p = Normal::new(0.0, 1.0).unwrap();
    let q = Normal::new(1.0, 1.0).unwrap();
    close(
        p.cross_entropy(&q).unwrap(),
        p.entropy().unwrap() + p.kl_divergence(&q),
        1e-12,
    );
    close(
        p.cross_entropy(&q).unwrap(),
        1.0 + 0.5 * std::f64::consts::TAU.ln(),
        1e-12,
    );
    // Gibbs equality case: H(p, p) = H(p); nonneg KL only raises it.
    close(p.cross_entropy(&p).unwrap(), p.entropy().unwrap(), 1e-12);
    assert!(p.cross_entropy(&q).unwrap() > p.entropy().unwrap());
    // No analytic entropy (Poisson) → honest `None`.
    assert!(
        Poisson::new(2.0)
            .unwrap()
            .cross_entropy(&Poisson::new(3.0).unwrap())
            .is_none()
    );
}
