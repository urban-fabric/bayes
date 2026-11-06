//! Halton against the radical inverse; Sobol against its net property.

use super::{PRIMES, radical_inverse};
use bayes_sampling::{Halton, Sobol};
use bayes_utils::Failures;

/// Halton's `d`-th coordinate is the radical inverse in the `d`-th prime,
/// **one-indexed**.
///
/// Bit for bit in **base 2**, where every value is a dyadic rational and
/// therefore exactly representable. In the other prime bases it cannot be
/// exact — `3/5` is not an `f64` — and two correct implementations that
/// accumulate the digits in different orders land one ulp apart, so those are
/// held to a few ulps instead. Demanding exactness everywhere was the natural
/// mistake and flagged 136 correct values.
///
/// The one-indexing is the part most worth pinning: the sequence starts at
/// `vdc(1) = 1/2`, not `vdc(0) = 0`, because a point at the origin is useless
/// for the inverse transform this sequence is for. Off by one here would shift
/// every coordinate of every point.
#[test]
fn halton_is_the_radical_inverse_in_the_prime_bases() {
    let mut f = Failures::default();
    for dimension in [1_usize, 2, 3, 5, 10, 32] {
        let mut h = Halton::new(dimension, 0).unwrap();
        for n in 0..24_u64 {
            let point = h.next_point();
            f.check(point.len() == dimension, || {
                format!("Halton({dimension}) returned {} coordinates", point.len())
            });
            for (d, &value) in point.iter().enumerate() {
                let expected = radical_inverse(n + 1, PRIMES[d]);
                // Base 2 is exact; the rest are representable only to a few
                // ulps, and the digit order is what is being pinned.
                let tolerance = if PRIMES[d] == 2 {
                    0.0
                } else {
                    8.0 * f64::EPSILON
                };
                f.check((value - expected).abs() <= tolerance, || {
                    format!(
                        "Halton({dimension}) point {n}, dimension {d} (base {}): {value:.17} \
                         but the radical inverse of {} is {expected:.17}",
                        PRIMES[d],
                        n + 1
                    )
                });
            }
        }
    }
    f.finish("Halton radical inverse");
}

/// Sobol's **net property**: among the first `2^m` points, each dimension hits
/// each of the `2^m` equal subintervals exactly once.
///
/// This is the defining property of a `(t,s)`-sequence in base 2, and the one
/// thing a marginal uniformity test cannot see: a sequence that clumps and one
/// that stratifies have the same coarse histogram. It is also exactly what a
/// single wrong direction number destroys — the points stay in `[0,1)` and stay
/// roughly uniform, and only the balance breaks.
///
/// Checked to `2^12` across every tabulated dimension, which is where a bad
/// entry in the direction table would show up.
#[test]
fn sobol_stratifies_every_dimension_at_every_power_of_two() {
    let mut f = Failures::default();
    let dimension = 6;
    for m in 1..=12_u32 {
        let count = 1_usize << m;
        let mut s = Sobol::new(dimension, 0).unwrap();
        let points = s.points(count);
        f.check(points.len() == count, || {
            format!(
                "Sobol returned {} of {count} requested points",
                points.len()
            )
        });
        for d in 0..dimension {
            let mut occupancy = vec![0_usize; count];
            for point in &points {
                let bucket = ((point[d] * count as f64).floor() as usize).min(count - 1);
                occupancy[bucket] += 1;
            }
            let wrong = occupancy.iter().filter(|&&c| c != 1).count();
            f.check(wrong == 0, || {
                format!(
                    "Sobol dimension {d}: of the first 2^{m} = {count} points, {wrong} of the \
                     {count} equal subintervals do not contain exactly one — the net property \
                     is broken, which is what a wrong direction number does"
                )
            });
        }
    }
    f.finish("Sobol net property");
}

/// Both sequences stay inside `[0, 1)` and never repeat a point.
///
/// Repetition is the failure mode of an index counter that wraps or a state
/// update that saturates: the points stay perfectly uniform and perfectly
/// stratified while silently contributing the same value twice.
#[test]
fn both_sequences_stay_in_the_unit_cube_and_never_repeat() {
    let mut f = Failures::default();
    for (name, points) in [
        ("Halton(4)", Halton::new(4, 0).unwrap().points(4096)),
        ("Sobol(4)", Sobol::new(4, 0).unwrap().points(4096)),
    ] {
        let mut seen = std::collections::HashSet::new();
        for (n, point) in points.iter().enumerate() {
            for (d, &value) in point.iter().enumerate() {
                f.check((0.0..1.0).contains(&value), || {
                    format!("{name} point {n} dimension {d} is {value}, outside [0, 1)")
                });
            }
            let key: Vec<u64> = point.iter().map(|v| v.to_bits()).collect();
            f.check(seen.insert(key), || {
                format!("{name} repeated a point at index {n}: {point:?}")
            });
        }
    }
    f.finish("unit cube and distinctness");
}

/// Quasi-random integration converges at the **rate** the module claims.
///
/// `O(n^{-1}(log n)^d)` against `O(n^{-1/2})` is a statement about rates, so
/// that is what is tested: quadrupling the point count must cut the
/// quasi-random error by well over the factor of two a random method gets.
/// Measured, the reduction is 3.3–5.2× for Sobol and Halton and 1.5–2.7× for
/// pseudo-random points, which is exactly the two rates.
///
/// A fixed "QMC beats MC by 10×" threshold was the first attempt, and it is the
/// wrong shape of claim — it depends on the integrand, the dimension and the
/// point count, and Halton legitimately failed it at 4096 points while being
/// four times better. A rate is scale-free and is the thing actually promised.
#[test]
fn quasi_random_integration_converges_faster_than_pseudo_random() {
    use rand::SeedableRng;
    use rand::distr::Distribution;
    use rand_chacha::ChaCha8Rng;

    // ∫₀¹∫₀¹ exp(x + y) dx dy = (e − 1)², known exactly.
    let truth = (std::f64::consts::E - 1.0).powi(2);
    let integrand = |p: &[f64]| (p[0] + p[1]).exp();
    let uniform = rand::distr::Uniform::new(0.0_f64, 1.0).unwrap();

    let error_at = |n: usize| {
        let sobol: f64 = Sobol::new(2, 0)
            .unwrap()
            .points(n)
            .iter()
            .map(|p| integrand(p))
            .sum::<f64>()
            / n as f64;
        let halton: f64 = Halton::new(2, 0)
            .unwrap()
            .points(n)
            .iter()
            .map(|p| integrand(p))
            .sum::<f64>()
            / n as f64;
        // The pseudo-random arm is a median over seeds, so the comparison is
        // against the method rather than against one unlucky draw.
        let mut errors: Vec<f64> = (0..9_u64)
            .map(|seed| {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let estimate: f64 = (0..n)
                    .map(|_| integrand(&[uniform.sample(&mut rng), uniform.sample(&mut rng)]))
                    .sum::<f64>()
                    / n as f64;
                (estimate - truth).abs()
            })
            .collect();
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (
            (sobol - truth).abs(),
            (halton - truth).abs(),
            errors[errors.len() / 2],
        )
    };

    let coarse = error_at(1 << 14);
    let fine = error_at(1 << 16);

    // Quadrupling the count: the quasi-random rate is near 4, random near 2.
    let sobol_rate = coarse.0 / fine.0;
    let halton_rate = coarse.1 / fine.1;
    let pseudo_rate = coarse.2 / fine.2;
    assert!(
        sobol_rate > 2.5,
        "Sobol's error should fall by well over 2x when n quadruples; got {sobol_rate:.2}"
    );
    assert!(
        halton_rate > 2.5,
        "Halton's error should fall by well over 2x when n quadruples; got {halton_rate:.2}"
    );
    assert!(
        pseudo_rate < 2.5,
        "pseudo-random error should fall by about 2x, not more; got {pseudo_rate:.2}"
    );
    assert!(
        sobol_rate > pseudo_rate && halton_rate > pseudo_rate,
        "both quasi-random rates ({sobol_rate:.2}, {halton_rate:.2}) should beat the \
         pseudo-random rate ({pseudo_rate:.2})"
    );

    // And at a fixed budget the absolute error is far smaller.
    assert!(
        fine.0 < fine.2 / 5.0 && fine.1 < fine.2 / 5.0,
        "at 65536 points Sobol {:.3e} and Halton {:.3e} should both be well under the \
         pseudo-random {:.3e}",
        fine.0,
        fine.1,
        fine.2
    );
}
