#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn empty_or_inverted_range_errors() {
    assert!(DiscreteUniform::new(5, 5).is_err());
    assert!(DiscreteUniform::new(10, 5).is_err());
}

#[test]
fn samples_lie_within_half_open_interval() {
    let u = DiscreteUniform::new(-3, 5).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..10_000 {
        let x = u.sample(&mut rng);
        assert!((-3..5).contains(&x), "out of range: {x}");
    }
}

#[test]
fn singleton_range_always_returns_lo() {
    let u = DiscreteUniform::new(7, 8).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for _ in 0..100 {
        assert_eq!(u.sample(&mut rng), 7);
    }
}

#[test]
fn approximately_uniform_buckets() {
    let u = DiscreteUniform::new(0, 6).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0xBEEF);
    let n = 60_000usize;
    let mut counts = [0u32; 6];
    for _ in 0..n {
        counts[u.sample(&mut rng) as usize] += 1;
    }

    for (i, &c) in counts.iter().enumerate() {
        let diff = (c as i64 - 10_000).unsigned_abs();
        assert!(diff < 400, "bucket {i}: count={c}, diff={diff}");
    }
}

#[test]
fn reproducible_by_seed() {
    let u = DiscreteUniform::new(0, 100).unwrap();
    let mut a = ChaCha8Rng::seed_from_u64(42);
    let mut b = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..32 {
        assert_eq!(u.sample(&mut a), u.sample(&mut b));
    }
}
