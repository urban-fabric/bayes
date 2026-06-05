#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rand::SeedableRng;
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

use bayes::{Bernoulli, DiscreteUniform, LogNormal, Normal, Poisson, Uniform, Uniform2D};

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
fn uniform2d_per_axis_moments_match() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
    let d = Uniform2D::new(-1.0, 1.0, 0.0, 4.0).unwrap();
    let pairs: Vec<(f64, f64)> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let xs = pairs.iter().map(|p| p.0);
    let ys = pairs.iter().map(|p| p.1);
    let (mx, vx) = moments(xs);
    let (my, vy) = moments(ys);

    assert_close(
        "Uniform2D x mean",
        mx,
        0.0,
        6.0 * (1.0 / 3.0 / N as f64).sqrt(),
    );
    assert_close("Uniform2D x var", vx, 1.0 / 3.0, 0.02);

    assert_close(
        "Uniform2D y mean",
        my,
        2.0,
        6.0 * (4.0 / 3.0 / N as f64).sqrt(),
    );
    assert_close("Uniform2D y var", vy, 4.0 / 3.0, 0.05);
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
