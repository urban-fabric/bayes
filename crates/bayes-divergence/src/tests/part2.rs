use super::*;

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
    let a = Exponential::new(2.0);
    let b = Exponential::new(1.0);
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
    let c = Exponential::new(1.0_f64);
    let d = Exponential::new(3.0_f64);
    assert!(c.chi_squared_divergence(&d).is_infinite());
}

#[test]
fn kl_le_chi_squared_holds() {
    // The proved bound KL ≤ χ², checked on Bernoulli and Poisson pairs.
    for (p, q) in [(0.3, 0.8), (0.1, 0.2), (0.6, 0.55)] {
        let a = Bernoulli::new(p);
        let b = Bernoulli::new(q);
        assert!(
            a.kl_divergence(&b) <= a.chi_squared_divergence(&b) + 1e-12,
            "KL≤χ² (Bernoulli) failed: {p},{q}"
        );
    }
    let a = Poisson::new(3.0);
    let b = Poisson::new(4.0);
    assert!(a.kl_divergence(&b) <= a.chi_squared_divergence(&b) + 1e-12);
}

#[test]
fn renyi_self_zero_and_limits_tie_to_the_other_divergences() {
    let a = Bernoulli::new(0.3);
    let b = Bernoulli::new(0.8);
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
        let bern = Bernoulli::new(0.7).renyi_divergence(&Bernoulli::new(0.4), alpha);
        close(cat, bern, 1e-12);
    }
}

#[test]
fn gaussian_renyi_matches_its_limits_and_the_integral() {
    let p = Normal::new(0.0, 1.0);
    let q = Normal::new(1.5, 2.0);
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
        Normal::new(0.0_f64, 3.0)
            .renyi_divergence(&Normal::new(0.0_f64, 1.0), 2.0)
            .is_infinite()
    );
    // LogNormal reuses the Normal formula on (μ, σ).
    for alpha in [0.4, 2.0] {
        close(
            LogNormal::new(0.0, 1.0).renyi_divergence(&LogNormal::new(1.5, 2.0), alpha),
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
        LogNormal::new(0.0, 1.0).kl_divergence(&LogNormal::new(1.0, 1.0)),
        0.5,
        1e-12,
    );
}

#[test]
fn laplace_and_geometric_kl_known_values() {
    // Laplace, same location: ln(b₂/b₁) + b₁/b₂ − 1.
    close(
        Laplace::new(0.0, 1.0).kl_divergence(&Laplace::new(0.0, 2.0)),
        2.0_f64.ln() - 0.5,
        1e-12,
    );
    // Geometric: ln(p₁/p₂) + ((1−p₁)/p₁)·ln((1−p₁)/(1−p₂)) = ln(4/3).
    close(
        Geometric::new(0.5).kl_divergence(&Geometric::new(0.25)),
        (4.0_f64 / 3.0).ln(),
        1e-12,
    );
}

#[test]
fn categorical_kl() {
    // The Multinomial / Dirichlet KL closed forms (which build on this categorical
    // KL) move with their distributions to the sibling crate `ito`.
    let p = Categorical::new(&[0.5, 0.5]).unwrap();
    let q = Categorical::new(&[0.25, 0.75]).unwrap();
    close(p.kl_divergence(&q), 0.5 * (4.0_f64 / 3.0).ln(), 1e-12);
    close(p.kl_divergence(&p), 0.0, 1e-12);
    assert!(
        p.kl_divergence(&Categorical::new(&[0.3, 0.3, 0.4]).unwrap())
            .is_infinite()
    );
}

#[test]
fn cauchy_rayleigh_weibull_pareto_kl() {
    // Cauchy: ln[((γ₁+γ₂)² + Δ²)/(4γ₁γ₂)]; here ln(9/8).
    let c = Cauchy::new(0.0, 1.0);
    close(
        c.kl_divergence(&Cauchy::new(0.0, 2.0)),
        (9.0_f64 / 8.0).ln(),
        1e-12,
    );
    close(c.kl_divergence(&c), 0.0, 1e-12);

    // Rayleigh: 2 ln(σ₂/σ₁) + (σ₁/σ₂)² − 1.
    close(
        Rayleigh::new(1.0).kl_divergence(&Rayleigh::new(2.0)),
        2.0 * 2.0_f64.ln() - 0.75,
        1e-12,
    );

    // Weibull(scale, shape=1) is Exponential(rate 1/scale); the KLs agree.
    close(
        Weibull::new(2.0, 1.0).kl_divergence(&Weibull::new(3.0, 1.0)),
        Exponential::new(0.5).kl_divergence(&Exponential::new(1.0 / 3.0)),
        1e-12,
    );

    // Pareto: finite only when xₘ₁ ≥ xₘ₂.
    let p = Pareto::new(1.0, 3.0);
    close(
        p.kl_divergence(&Pareto::new(1.0, 2.0)),
        1.5_f64.ln() + 2.0 / 3.0 - 1.0,
        1e-12,
    );
    assert!(
        Pareto::new(2.0_f64, 2.0)
            .kl_divergence(&Pareto::new(3.0_f64, 2.0))
            .is_infinite()
    );
}

#[test]
fn cross_entropy_decomposes_and_handles_missing_entropy() {
    // H(p, q) = H(p) + D_KL(p ‖ q), equal to the closed form for normals.
    let p = Normal::new(0.0, 1.0);
    let q = Normal::new(1.0, 1.0);
    close(
        p.cross_entropy(&q),
        p.entropy() + p.kl_divergence(&q),
        1e-12,
    );
    close(
        p.cross_entropy(&q),
        1.0 + 0.5 * std::f64::consts::TAU.ln(),
        1e-12,
    );
    // Gibbs equality case: H(p, p) = H(p); nonneg KL only raises it.
    close(p.cross_entropy(&p), p.entropy(), 1e-12);
    assert!(p.cross_entropy(&q) > p.entropy());
    // No analytic entropy (Poisson) -> explicit `NaN`.
    assert!(
        Poisson::new(2.0_f64)
            .cross_entropy(&Poisson::new(3.0_f64))
            .is_nan()
    );
}
