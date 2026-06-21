#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rand::SeedableRng;
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

use bayes::{
    Bernoulli, Categorical, DiscreteUniform, Exponential, Gamma, LogNormal, Moments, Normal,
    Poisson, Triangular, Uniform, Weibull,
};

const N: usize = 20_000;

fn moments<I: Iterator<Item = f64>>(xs: I) -> (f64, f64) {
    let v: Vec<f64> = xs.collect();
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0);
    (mean, var)
}

fn assert_close(label: &str, observed: f64, expected: f64, tol: f64) {
    let diff = (observed - expected).abs();
    assert!(
        diff <= tol,
        "{label}: observed={observed:.6}, expected={expected:.6}, |diff|={diff:.6} > tol={tol:.6}"
    );
}

#[test]
fn bernoulli_moments_match_theoretical() {
    for p in [0.1, 0.5, 0.9] {
        let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
        let d = Bernoulli::new(p).unwrap();
        let (mean, var) = moments((0..N).map(|_| if d.sample(&mut rng) { 1.0 } else { 0.0 }));
        let theo_var = p * (1.0 - p);
        let mean_tol = 6.0 * (theo_var / N as f64).sqrt();
        let var_tol = 6.0 * (2.0 * theo_var.powi(2) / (N as f64 - 1.0)).sqrt();
        assert_close(&format!("Bernoulli({p}) mean"), mean, p, mean_tol);
        assert_close(&format!("Bernoulli({p}) var"), var, theo_var, var_tol);
    }
}

#[test]
fn uniform_moments_match_theoretical() {
    for (a, b) in [(0.0, 1.0), (-3.0, 3.0), (10.0, 11.0)] {
        let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
        let d = Uniform::new(a, b).unwrap();
        let (mean, var) = moments((0..N).map(|_| d.sample(&mut rng)));
        let theo_mean = 0.5 * (a + b);
        let theo_var = (b - a).powi(2) / 12.0;
        let mean_tol = 6.0 * (theo_var / N as f64).sqrt();
        let var_tol = 6.0 * (2.0 * theo_var.powi(2) / (N as f64 - 1.0)).sqrt();
        assert_close(&format!("Uniform({a},{b}) mean"), mean, theo_mean, mean_tol);
        assert_close(&format!("Uniform({a},{b}) var"), var, theo_var, var_tol);
    }
}

#[test]
fn discrete_uniform_moments_match_theoretical() {
    for (a, b) in [(0i64, 10), (-5, 5), (100, 200)] {
        let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
        let d = DiscreteUniform::new(a, b).unwrap();
        let (mean, var) = moments((0..N).map(|_| d.sample(&mut rng) as f64));
        let n = (b - a) as f64;
        let theo_mean = (a as f64 + b as f64 - 1.0) / 2.0;
        let theo_var = (n * n - 1.0) / 12.0;
        let mean_tol = 6.0 * (theo_var / N as f64).sqrt();
        let var_tol = 6.0 * (2.0 * theo_var.powi(2) / (N as f64 - 1.0)).sqrt();
        assert_close(
            &format!("DiscreteUniform({a},{b}) mean"),
            mean,
            theo_mean,
            mean_tol,
        );
        assert_close(
            &format!("DiscreteUniform({a},{b}) var"),
            var,
            theo_var,
            var_tol,
        );
    }
}

#[test]
fn poisson_moments_match_theoretical() {
    for lambda in [0.5_f64, 5.0, 25.0, 50.0, 200.0] {
        let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
        let d = Poisson::new(lambda).unwrap();
        let (mean, var) = moments((0..N).map(|_| d.sample(&mut rng) as f64));
        let mean_tol = 6.0 * (lambda / N as f64).sqrt();
        let var_tol = 6.0 * (2.0 * lambda.powi(2) / (N as f64 - 1.0)).sqrt();
        assert_close(&format!("Poisson({lambda}) mean"), mean, lambda, mean_tol);
        assert_close(&format!("Poisson({lambda}) var"), var, lambda, var_tol);
    }
}

#[test]
fn normal_moments_match_theoretical() {
    for (mu, sigma) in [(0.0, 1.0), (5.0, 2.0), (-3.0, 0.5)] {
        let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
        let d = Normal::new(mu, sigma).unwrap();
        let (mean, var) = moments((0..N).map(|_| d.sample(&mut rng)));
        let theo_var = sigma.powi(2);
        let mean_tol = 6.0 * (theo_var / N as f64).sqrt();
        let var_tol = 6.0 * (2.0 * theo_var.powi(2) / (N as f64 - 1.0)).sqrt();
        assert_close(&format!("Normal({mu},{sigma}) mean"), mean, mu, mean_tol);
        assert_close(&format!("Normal({mu},{sigma}) var"), var, theo_var, var_tol);
    }
}

#[test]
fn moment2_matches_closed_form_second_moment() {
    // The `Moments::moment2` default is the computational-variance identity
    // `E[X²] = Var(X) + E[X]²` (riemann `Bayes/Moments.variance_eq_moment2_sub_sq`).
    // It must reproduce each family's analytic raw second moment, and — by
    // construction — equal `variance + mean²`.
    fn check<M: Moments>(d: &M, expected_e_x2: f64, label: &str) {
        let m2 = d.moment2().unwrap();
        assert_close(label, m2, expected_e_x2, 1e-12);
        assert_close(
            &format!("{label} (identity)"),
            m2,
            d.variance().unwrap() + d.mean().unwrap().powi(2),
            1e-12,
        );
    }

    check(&Normal::new(2.0, 3.0).unwrap(), 9.0 + 4.0, "Normal E[X²]");
    check(&Exponential::new(2.0).unwrap(), 0.5, "Exponential E[X²]"); // 2/λ²
    check(&Poisson::new(4.0).unwrap(), 20.0, "Poisson E[X²]"); // λ + λ²
    check(&Bernoulli::new(0.3).unwrap(), 0.3, "Bernoulli E[X²]"); // p (X²=X)
    check(&Gamma::new(2.0, 1.0).unwrap(), 6.0, "Gamma E[X²]"); // a(a+1)/b²
}

#[test]
fn closed_form_entropies_match_known_values() {
    use bayes::{Beta, Binomial, Discrete, Divergence, Geometric, KlDivergence};
    let ln2 = std::f64::consts::LN_2;

    // Bernoulli(½) = 1 bit = ln 2 nats; Beta(1,1) is uniform on (0,1) → 0;
    // Geometric(½) = 2 ln 2.
    assert_close(
        "Bernoulli(.5) H",
        Bernoulli::new(0.5).unwrap().entropy().unwrap(),
        ln2,
        1e-12,
    );
    assert_close(
        "Beta(1,1) H",
        Beta::new(1.0, 1.0).unwrap().entropy().unwrap(),
        0.0,
        1e-12,
    );
    assert_close(
        "Geometric(.5) H",
        Geometric::new(0.5).unwrap().entropy().unwrap(),
        2.0 * ln2,
        1e-12,
    );

    // Binomial(1, p) collapses to a Bernoulli, so its exact-sum entropy agrees.
    let b1 = Binomial::new(1, 0.3).unwrap();
    assert_close(
        "Binomial(1,.3) H",
        b1.entropy().unwrap(),
        Bernoulli::new(0.3).unwrap().entropy().unwrap(),
        1e-12,
    );
    // Binomial entropy is bounded by ln(n+1) (uniform over n+1 outcomes).
    let b = Binomial::new(10, 0.5).unwrap();
    assert!(b.entropy().unwrap() > 0.0 && b.entropy().unwrap() < 11.0_f64.ln());
    let _ = b.pmf(5_u64); // Discrete in scope

    // The entropy fill unlocks cross_entropy for these families.
    let p = Beta::new(2.0, 3.0).unwrap();
    let q = Beta::new(3.0, 2.0).unwrap();
    assert_close(
        "Beta cross-entropy",
        p.cross_entropy(&q).unwrap(),
        p.entropy().unwrap() + p.kl_divergence(&q),
        1e-12,
    );
}

#[test]
fn more_closed_form_entropies_match_known_values() {
    let ln2 = std::f64::consts::LN_2;

    // Weibull(scale λ, shape 1) is Exponential(rate 1/λ); the entropies agree.
    assert_close(
        "Weibull(2,1) H",
        Weibull::new(2.0, 1.0).unwrap().entropy().unwrap(),
        Exponential::new(0.5).unwrap().entropy().unwrap(),
        1e-12,
    );
    // Triangular(0,2,c): H = ½ + ln((b−a)/2) = ½, independent of the mode.
    assert_close(
        "Triangular H",
        Triangular::new(0.0, 2.0, 1.5).unwrap().entropy().unwrap(),
        0.5,
        1e-12,
    );
    // DiscreteUniform(0,10): 10 outcomes → ln 10.
    assert_close(
        "DiscreteUniform H",
        DiscreteUniform::new(0, 10).unwrap().entropy().unwrap(),
        10.0_f64.ln(),
        1e-12,
    );
    // Categorical: fair coin → ln 2; uniform over 4 → ln 4; point mass → 0.
    assert_close(
        "Categorical[.5,.5] H",
        Categorical::new(&[0.5, 0.5]).unwrap().entropy().unwrap(),
        ln2,
        1e-12,
    );
    assert_close(
        "Categorical 4-uniform H",
        Categorical::new(&[0.25; 4]).unwrap().entropy().unwrap(),
        4.0_f64.ln(),
        1e-12,
    );
    assert_close(
        "Categorical point-mass H",
        Categorical::new(&[1.0, 0.0]).unwrap().entropy().unwrap(),
        0.0,
        1e-12,
    );
}

#[test]
fn lognormal_moments_match_theoretical() {
    for (mu, sigma) in [(0.0_f64, 0.5), (1.0, 0.25), (-1.0, 0.1)] {
        let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
        let d = LogNormal::new(mu, sigma).unwrap();
        let (mean, var) = moments((0..N).map(|_| d.sample(&mut rng)));
        let theo_mean = (mu + sigma.powi(2) / 2.0).exp();
        let theo_var = ((sigma.powi(2)).exp() - 1.0) * (2.0 * mu + sigma.powi(2)).exp();
        let mean_tol = 6.0 * (theo_var / N as f64).sqrt();

        let var_tol = (8.0 * (theo_var.powi(2) / (N as f64 - 1.0)).sqrt()).max(0.01);
        assert_close(
            &format!("LogNormal({mu},{sigma}) mean"),
            mean,
            theo_mean,
            mean_tol,
        );
        assert_close(
            &format!("LogNormal({mu},{sigma}) var"),
            var,
            theo_var,
            var_tol,
        );
    }
}
