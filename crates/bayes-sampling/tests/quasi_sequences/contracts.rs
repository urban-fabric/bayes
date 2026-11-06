//! The module's own documented claims, and what happens at the edges.

use bayes_sampling::{Halton, Sobol};
use bayes_utils::Failures;

/// "The `n`-th point is a pure function of `n`, so a result reproduces without
/// storing a seed, and two processes can split the sequence by index with no
/// coordination."
///
/// That claim has three parts and each is checked: two constructions agree
/// element for element; `points(n)` agrees with `n` calls to `next_point`; and
/// a `burn` of `k` lands exactly on the `k`-th point, which is what "split by
/// index" requires.
#[test]
fn the_sequences_are_addressable_by_index_as_documented() {
    let mut f = Failures::default();

    // Two independent constructions agree.
    for dimension in [1_usize, 3, 6] {
        let left = Halton::new(dimension, 0).unwrap().points(64);
        let right = Halton::new(dimension, 0).unwrap().points(64);
        f.check(left == right, || {
            format!("two Halton({dimension}) sequences disagreed")
        });
        let left = Sobol::new(dimension, 0).unwrap().points(64);
        let right = Sobol::new(dimension, 0).unwrap().points(64);
        f.check(left == right, || {
            format!("two Sobol({dimension}) sequences disagreed")
        });
    }

    // Batched and streaming agree.
    let batched = Halton::new(3, 0).unwrap().points(32);
    let mut streaming = Halton::new(3, 0).unwrap();
    for (n, expected) in batched.iter().enumerate() {
        let actual = streaming.next_point();
        f.check(&actual == expected, || {
            format!("Halton point {n}: streaming {actual:?} vs batched {expected:?}")
        });
    }
    let batched = Sobol::new(3, 0).unwrap().points(32);
    let mut streaming = Sobol::new(3, 0).unwrap();
    for (n, expected) in batched.iter().enumerate() {
        let actual = streaming.next_point();
        f.check(&actual == expected, || {
            format!("Sobol point {n}: streaming {actual:?} vs batched {expected:?}")
        });
    }

    // A burn of `k` starts at the `k`-th point: this is the splitting contract.
    let whole = Halton::new(2, 0).unwrap().points(40);
    for burn in [0_u64, 1, 7, 13, 39] {
        let resumed = Halton::new(2, burn).unwrap().points(1);
        f.check(resumed[0] == whole[burn as usize], || {
            format!(
                "Halton burn {burn} gave {:?}, but the sequence's point {burn} is {:?}",
                resumed[0], whole[burn as usize]
            )
        });
    }
    let whole = Sobol::new(2, 0).unwrap().points(40);
    for burn in [0_u64, 1, 7, 13, 39] {
        let resumed = Sobol::new(2, burn).unwrap().points(1);
        f.check(resumed[0] == whole[burn as usize], || {
            format!(
                "Sobol burn {burn} gave {:?}, but the sequence's point {burn} is {:?}",
                resumed[0], whole[burn as usize]
            )
        });
    }

    // Halton also exposes direct addressing, which must agree with both.
    let addressed = Halton::new(3, 0).unwrap();
    let streamed = Halton::new(3, 0).unwrap().points(32);
    for (n, expected) in streamed.iter().enumerate() {
        let actual = addressed.point_at(n as u64);
        f.check(&actual == expected, || {
            format!("Halton point_at({n}) = {actual:?} but the sequence has {expected:?}")
        });
    }
    f.finish("index addressing");
}

/// **Sobol starts at the origin and Halton does not**, as `Sobol::next_point`
/// documents.
///
/// Pinned rather than discovered: the type's own docs explain that dropping the
/// origin would break the stratification property, since the first four points
/// would then put two in one quarter and none in another. This is the
/// regression test for that reasoning, and for the remedy the docs name.
///
/// It is worth a test because the consequence is easy to meet by accident.
/// Driving an inverse transform through these sequences maps the first Sobol
/// point to the distribution's lower support bound — `Q(0)` is `−inf` for a
/// normal — so a Sobol-driven sample begins with a degenerate draw where a
/// Halton-driven one does not.
#[test]
fn sobol_begins_at_the_origin_and_halton_does_not() {
    let first_sobol = Sobol::new(4, 0).unwrap().points(1);
    assert!(
        first_sobol[0].iter().all(|&v| v == 0.0),
        "Sobol's n = 0 point is the origin by construction, got {:?}",
        first_sobol[0]
    );

    let first_halton = Halton::new(4, 0).unwrap().points(1);
    assert!(
        first_halton[0].iter().all(|&v| v > 0.0),
        "Halton skips its origin and should start strictly inside, got {:?}",
        first_halton[0]
    );

    // A burn of 1 is the remedy, and it leaves the rest of the sequence intact.
    let burned = Sobol::new(4, 1).unwrap().points(8);
    assert!(
        burned.iter().all(|p| p.iter().all(|&v| v > 0.0)),
        "a burn of 1 should skip the origin: {burned:?}"
    );
    let unburned = Sobol::new(4, 0).unwrap().points(9);
    assert_eq!(
        burned,
        unburned[1..],
        "burning should skip the first point, not change the others"
    );
}

/// Dimensions outside the tabulated range are refused, not silently truncated.
///
/// Both sequences carry finite tables — 32 primes for Halton, a direction-number
/// table for Sobol — and a request beyond them has no answer. Returning fewer
/// coordinates than asked for, or recycling a base, would produce a sequence
/// with duplicated dimensions: perfectly uniform in each margin and perfectly
/// correlated between them, which is the single worst failure mode for
/// integration.
#[test]
fn an_untabulated_dimension_is_refused() {
    let mut f = Failures::default();
    for dimension in [0_usize, 33, 64, 1000, usize::MAX] {
        let halton = Halton::new(dimension, 0);
        f.check(halton.is_err() || dimension <= 32, || {
            format!("Halton accepted dimension {dimension}")
        });
        let sobol = Sobol::new(dimension, 0);
        f.check(sobol.is_err() || dimension <= 32, || {
            format!("Sobol accepted dimension {dimension}")
        });
    }
    // Zero dimensions is refused by both: there is no such point.
    f.check(Halton::new(0, 0).is_err(), || {
        "Halton should refuse dimension 0".into()
    });
    f.check(Sobol::new(0, 0).is_err(), || {
        "Sobol should refuse dimension 0".into()
    });

    // And wherever a dimension *is* accepted, it yields exactly that many
    // coordinates — no recycling.
    for dimension in 1..=32_usize {
        if let Ok(mut h) = Halton::new(dimension, 0) {
            let p = h.next_point();
            f.check(p.len() == dimension, || {
                format!("Halton({dimension}) gave {} coordinates", p.len())
            });
            // Distinct bases mean distinct coordinates for the first point.
            let mut sorted = p.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted.dedup();
            f.check(sorted.len() == dimension || dimension == 1, || {
                format!(
                    "Halton({dimension})'s first point has repeated coordinates, which means a \
                     recycled base: {p:?}"
                )
            });
        }
    }
    f.finish("dimension limits");
}

/// A large burn is answered in closed form, and a burn past the end of the
/// sequence is refused.
///
/// Reaching index `n` used to cost `n` steps for Sobol — `Sobol::new(3, 1 << 48)`
/// did not return — which made the documented "split by index with no
/// coordination" useless for exactly the sizes that need splitting. The
/// Gray-code recurrence telescopes, so the state at `n` is one XOR per set bit
/// of `n ^ (n >> 1)`.
///
/// Past `2^32` the sequence has no more addresses and the Gray code cycles, so
/// the closed form would hand back the origin. That is refused instead. Halton
/// has no such ceiling: its index feeds a radical inverse directly, so the
/// whole `u64` range is addressable, and this records that asymmetry as
/// deliberate.
#[test]
fn a_large_burn_is_closed_form_and_a_burn_past_the_end_is_refused() {
    let mut f = Failures::default();

    // Within range: instant, in the unit cube, and not the origin.
    for burn in [1_u64 << 10, 1 << 20, 1 << 31, (1 << 32) - 1] {
        let mut sobol = Sobol::new(3, burn).expect("inside the 32-bit address space");
        let point = sobol.points(1);
        for (d, &value) in point[0].iter().enumerate() {
            f.check(value.is_finite() && (0.0..1.0).contains(&value), || {
                format!("Sobol at burn {burn}, dimension {d} gave {value}")
            });
        }
    }
    // Past the end: refused rather than silently the origin.
    for burn in [1_u64 << 32, 1 << 48, u64::MAX] {
        f.check(Sobol::new(3, burn).is_err(), || {
            format!("Sobol accepted burn {burn}, past its 2^32 addresses")
        });
    }
    // Halton addresses the whole range, including the very top — where the
    // radical inverse is `1 − 2^-64` and must round *down* to stay inside
    // `[0, 1)` rather than up to exactly 1.
    for burn in [1_u64 << 32, 1 << 48, u64::MAX - 1] {
        let point = Halton::new(3, burn)
            .expect("Halton has no index ceiling")
            .points(1);
        for (d, &value) in point[0].iter().enumerate() {
            f.check(value.is_finite() && (0.0..1.0).contains(&value), || {
                format!("Halton at burn {burn}, dimension {d} gave {value}")
            });
        }
        let origin = Halton::new(3, 0).unwrap().points(1);
        f.check(point != origin, || {
            format!("Halton wrapped to its first point at burn {burn}")
        });
    }
    f.finish("large burns");
}
