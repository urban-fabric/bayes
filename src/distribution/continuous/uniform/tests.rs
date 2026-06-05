#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn empty_or_inverted_range_errors() {
    assert!(Uniform::new(1.0, 1.0).is_err());
    assert!(Uniform::new(2.0, 1.0).is_err());
}

#[test]
fn nonfinite_bounds_error() {
    assert!(Uniform::new(f64::NAN, 1.0).is_err());
    assert!(Uniform::new(0.0, f64::INFINITY).is_err());
}

#[test]
fn symmetric_radius_builds_expected_interval() {
    let u = Uniform::symmetric_radius(0.25);
    assert_eq!(u.lo(), -0.25);
    assert_eq!(u.hi(), 0.25);
}

#[test]
fn samples_lie_within_half_open_interval() {
    let u = Uniform::new(-3.0, 5.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..10_000 {
        let x = u.sample(&mut rng);
        assert!((-3.0..5.0).contains(&x), "out of range: {x}");
    }
}

#[test]
fn empirical_mean_within_three_sigma() {
    let u = Uniform::new(0.0, 1.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let n = 10_000usize;
    let sum: f64 = (0..n).map(|_| u.sample(&mut rng)).sum();
    let mean = sum / n as f64;

    assert!((mean - 0.5).abs() < 0.01, "mean={mean}");
}

#[test]
fn reproducible_by_seed() {
    let u = Uniform::new(-1.0, 1.0).unwrap();
    let mut a = ChaCha8Rng::seed_from_u64(42);
    let mut b = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..32 {
        assert_eq!(u.sample(&mut a), u.sample(&mut b));
    }
}
