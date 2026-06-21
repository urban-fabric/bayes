#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn p_outside_unit_interval_errors() {
    assert!(Bernoulli::new(-0.1).is_err());
    assert!(Bernoulli::new(1.1).is_err());
    assert!(Bernoulli::new(f64::NAN).is_err());
}

#[test]
fn saturating_constructor_clamps_without_error() {
    assert_eq!(Bernoulli::from_probability_saturating(-0.1).p(), 0.0);
    assert_eq!(Bernoulli::from_probability_saturating(0.4).p(), 0.4);
    assert_eq!(Bernoulli::from_probability_saturating(1.1).p(), 1.0);
    assert_eq!(Bernoulli::from_probability_saturating(f64::NAN).p(), 0.0);
}

#[test]
fn boundary_values_short_circuit() {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let always_false = Bernoulli::new(0.0).unwrap();
    let always_true = Bernoulli::new(1.0).unwrap();
    for _ in 0..100 {
        assert!(!always_false.sample(&mut rng));
        assert!(always_true.sample(&mut rng));
    }
}

#[test]
fn fair_coin_is_within_three_sigma_at_n10000() {
    let coin = Bernoulli::new(0.5).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);
    let n = 10_000usize;
    let heads = (0..n).filter(|_| coin.sample(&mut rng)).count();

    let diff = (heads as i64 - 5000).unsigned_abs() as usize;
    assert!(diff < 150, "heads={heads}, |diff|={diff}, 3σ=150");
}

#[test]
fn reproducible_by_seed() {
    let coin = Bernoulli::new(0.3).unwrap();
    let mut a = ChaCha8Rng::seed_from_u64(42);
    let mut b = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..32 {
        assert_eq!(coin.sample(&mut a), coin.sample(&mut b));
    }
}
