#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn invalid_params_error() {
    assert!(Normal::new(0.0, 0.0).is_err());
    assert!(Normal::new(0.0, -1.0).is_err());
    assert!(Normal::new(f64::NAN, 1.0).is_err());
    assert!(Normal::new(0.0, f64::INFINITY).is_err());
}

#[test]
fn standard_normal_mean_close_to_zero() {
    let n = Normal::new(0.0, 1.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
    let m = 50_000usize;
    let sum: f64 = (0..m).map(|_| n.sample(&mut rng)).sum();
    let mean = sum / m as f64;

    assert!(mean.abs() < 0.025, "mean={mean}");
}

#[test]
fn standard_normal_variance_close_to_one() {
    let n = Normal::new(0.0, 1.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xBEEF);
    let m = 50_000usize;
    let xs: Vec<f64> = (0..m).map(|_| n.sample(&mut rng)).collect();
    let mean = xs.iter().sum::<f64>() / m as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0);

    assert!((var - 1.0).abs() < 0.04, "var={var}");
}

#[test]
fn shifted_normal_recovers_mean_and_std() {
    let n = Normal::new(7.0, 2.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let m = 50_000usize;
    let xs: Vec<f64> = (0..m).map(|_| n.sample(&mut rng)).collect();
    let mean = xs.iter().sum::<f64>() / m as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0);

    assert!((mean - 7.0).abs() < 0.05, "mean={mean}");
    assert!((var - 4.0).abs() < 0.15, "var={var}");
}

#[test]
fn reproducible_by_seed() {
    let n = Normal::new(0.0, 1.0).unwrap();
    let mut a = ChaCha8Rng::seed_from_u64(42);
    let mut b = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..32 {
        assert_eq!(n.sample(&mut a), n.sample(&mut b));
    }
}

#[test]
fn independent_normal_algebra_is_closed_form() {
    let a = Normal::new(1.0, 2.0).unwrap();
    let b = Normal::new(3.0, 4.0).unwrap();

    let sum = a.sum_independent(b);
    assert!((sum.mean().unwrap() - 4.0).abs() < 1e-12);
    assert!((sum.variance().unwrap() - 20.0).abs() < 1e-12);

    let diff = a.difference_independent(b);
    assert!((diff.mean().unwrap() + 2.0).abs() < 1e-12);
    assert!((diff.variance().unwrap() - 20.0).abs() < 1e-12);

    let affine = a.affine(-2.0, 5.0).unwrap();
    assert!((affine.mean().unwrap() - 3.0).abs() < 1e-12);
    assert!((affine.variance().unwrap() - 16.0).abs() < 1e-12);

    let combo = Normal::linear_combination_independent(&[(2.0, a), (-1.0, b)], 10.0).unwrap();
    assert!((combo.mean().unwrap() - 9.0).abs() < 1e-12);
    assert!((combo.variance().unwrap() - 32.0).abs() < 1e-12);
    assert!(Normal::linear_combination_independent(&[], 0.0).is_err());
}
