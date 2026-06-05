#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn invalid_params_propagate_normal_error() {
    assert!(LogNormal::new(0.0, 0.0).is_err());
    assert!(LogNormal::new(0.0, -1.0).is_err());
    assert!(LogNormal::new(f64::NAN, 1.0).is_err());
}

#[test]
fn samples_are_strictly_positive() {
    let ln = LogNormal::new(0.0, 1.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
    for _ in 0..10_000 {
        let y = ln.sample(&mut rng);
        assert!(y > 0.0, "non-positive sample: {y}");
    }
}

#[test]
fn unit_lognormal_mean_close_to_exp_half() {
    let ln = LogNormal::new(0.0, 1.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xBEEF);
    let n = 100_000usize;
    let sum: f64 = (0..n).map(|_| ln.sample(&mut rng)).sum();
    let mean = sum / n as f64;
    let target = 0.5f64.exp();

    assert!((mean - target).abs() < 0.05, "mean={mean} target={target}");
}

#[test]
fn reproducible_by_seed() {
    let ln = LogNormal::new(0.0, 1.0).unwrap();
    let mut a = ChaCha8Rng::seed_from_u64(42);
    let mut b = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..32 {
        assert_eq!(ln.sample(&mut a), ln.sample(&mut b));
    }
}
