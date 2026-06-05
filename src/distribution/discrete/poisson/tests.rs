#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn invalid_lambda_errors() {
    assert!(Poisson::new(0.0).is_err());
    assert!(Poisson::new(-1.0).is_err());
    assert!(Poisson::new(f64::NAN).is_err());
    assert!(Poisson::new(f64::INFINITY).is_err());
    assert!(Poisson::new(Poisson::MAX_LAMBDA * 10.0).is_err());
}

#[test]
fn ln_factorial_loop_matches_stirling_at_boundary() {
    let exact_20 = ln_factorial(20);
    let nf = 20.0f64;
    let two_pi = std::f64::consts::TAU;
    let stirling = nf * nf.ln() - nf + 0.5 * (two_pi * nf).ln() + 1.0 / (12.0 * nf)
        - 1.0 / (360.0 * nf * nf * nf);
    let rel = (exact_20 - stirling).abs() / exact_20.abs();
    assert!(rel < 1e-7, "rel error {rel}");
}

#[test]
fn ln_factorial_matches_known_values() {
    let cases = [
        (0u64, 0.0f64),
        (1, 0.0),
        (5, 120.0_f64.ln()),
        (10, 3_628_800.0_f64.ln()),
    ];
    for (n, expected) in cases {
        let got = ln_factorial(n);
        assert!(
            (got - expected).abs() < 1e-12,
            "ln({n}!) got {got}, expected {expected}"
        );
    }
}

#[test]
fn knuth_sample_mean_close_to_lambda() {
    let p = Poisson::new(5.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
    let n = 50_000usize;
    let sum: u64 = (0..n).map(|_| p.sample(&mut rng)).sum();
    let mean = sum as f64 / n as f64;

    assert!((mean - 5.0).abs() < 0.05, "mean={mean}");
}

#[test]
fn rejection_branch_sample_mean_close_to_lambda() {
    let p = Poisson::new(50.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xDEAD);
    let n = 50_000usize;
    let sum: u64 = (0..n).map(|_| p.sample(&mut rng)).sum();
    let mean = sum as f64 / n as f64;

    assert!((mean - 50.0).abs() < 0.2, "mean={mean}");
}

#[test]
fn reproducible_by_seed() {
    let p = Poisson::new(7.5).unwrap();
    let mut a = ChaCha8Rng::seed_from_u64(42);
    let mut b = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..32 {
        assert_eq!(p.sample(&mut a), p.sample(&mut b));
    }
}
