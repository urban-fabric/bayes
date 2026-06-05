#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn unit_square_samples_in_unit_square() {
    let u = Uniform2D::unit_square();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for _ in 0..10_000 {
        let (x, y) = u.sample(&mut rng);
        assert!((0.0..1.0).contains(&x), "x out of range: {x}");
        assert!((0.0..1.0).contains(&y), "y out of range: {y}");
    }
}

#[test]
fn rectangle_samples_in_rectangle() {
    let u = Uniform2D::new(-2.0, 2.0, 100.0, 200.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for _ in 0..10_000 {
        let (x, y) = u.sample(&mut rng);
        assert!((-2.0..2.0).contains(&x), "x out of range: {x}");
        assert!((100.0..200.0).contains(&y), "y out of range: {y}");
    }
}

#[test]
fn invalid_axis_propagates_error() {
    assert!(Uniform2D::new(1.0, 1.0, 0.0, 1.0).is_err());
    assert!(Uniform2D::new(0.0, 1.0, 5.0, 5.0).is_err());
    assert!(Uniform2D::new(f64::NAN, 1.0, 0.0, 1.0).is_err());
}

#[test]
fn axes_are_independent_at_two_sigma() {
    let u = Uniform2D::unit_square();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let n = 50_000usize;
    let mut sum_xy = 0.0f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for _ in 0..n {
        let (x, y) = u.sample(&mut rng);
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
    }
    let nf = n as f64;
    let cov = sum_xy / nf - (sum_x / nf) * (sum_y / nf);

    assert!(cov.abs() < 0.005, "covariance non-zero: {cov}");
}

#[test]
fn reproducible_by_seed() {
    let u = Uniform2D::unit_square();
    let mut a = ChaCha8Rng::seed_from_u64(42);
    let mut b = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..32 {
        assert_eq!(u.sample(&mut a), u.sample(&mut b));
    }
}
