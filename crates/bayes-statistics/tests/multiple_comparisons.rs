//! The multiple-comparison corrections, against their definitions and against
//! each other.
//!
//! Three kinds of check here, in increasing strength:
//!
//! 1. **Definitional** — Bonferroni is exactly `min(1, m·p)`, Šidák exactly
//!    `1 − (1−p)^m`, Benjamini–Yekutieli exactly Benjamini–Hochberg times the
//!    harmonic number. These are closed forms, so they are asserted at machine
//!    precision.
//! 2. **Ordering** — the corrections form a conservatism chain, and each link is
//!    a theorem: Holm dominates Bonferroni uniformly, BH dominates Holm. A
//!    violation means one of the sweeps runs the wrong way, which is the easiest
//!    thing to get wrong here and the hardest to notice.
//! 3. **Operational** — under the global null, Benjamini–Hochberg must actually
//!    hold the false discovery rate at the level requested. That is checked by
//!    simulation, because it is a statement about long-run behaviour rather than
//!    about any single call.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use bayes_distribution_ext::Uniform;
use bayes_statistics::multiple::{
    Correction, benjamini_hochberg, benjamini_yekutieli, bonferroni, holm, holm_sidak, reject_at,
    sidak,
};
use bayes_utils::Failures;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

/// A spread of p-values: two clearly significant, two borderline, two null.
const P: [f64; 6] = [0.001, 0.008, 0.021, 0.04, 0.3, 0.81];

const ALL: [Correction; 6] = [
    Correction::Bonferroni,
    Correction::Sidak,
    Correction::Holm,
    Correction::HolmSidak,
    Correction::BenjaminiHochberg,
    Correction::BenjaminiYekutieli,
];

#[test]
fn the_closed_form_corrections_match_their_definitions() {
    let mut f = Failures::default();
    let m = P.len() as f64;

    let b = bonferroni(&P);
    for (i, &p) in P.iter().enumerate() {
        f.close(
            || format!("bonferroni[{i}] = min(1, m·p)"),
            b[i],
            (p * m).min(1.0),
            1e-15,
        );
    }

    let s = sidak(&P);
    for (i, &p) in P.iter().enumerate() {
        f.close(
            || format!("sidak[{i}] = 1 - (1-p)^m"),
            s[i],
            1.0 - (1.0 - p).powf(m),
            1e-14,
        );
    }

    // Benjamini-Yekutieli is Benjamini-Hochberg scaled by the harmonic number,
    // except where either saturates at 1.
    let bh = benjamini_hochberg(&P);
    let by = benjamini_yekutieli(&P);
    let harmonic: f64 = (1..=P.len()).map(|k| 1.0 / k as f64).sum();
    for i in 0..P.len() {
        if by[i] < 1.0 && bh[i] * harmonic < 1.0 {
            f.close(
                || format!("by[{i}] = bh[{i}] · H(m)"),
                by[i],
                bh[i] * harmonic,
                1e-12,
            );
        }
    }

    f.finish("closed forms");
}

/// Benjamini–Hochberg worked by hand on `P`, to pin the step-up direction.
///
/// Raw `p · m / rank` is `0.006, 0.024, 0.042, 0.06, 0.36, 0.81`, which is
/// already increasing, so the running minimum from the top changes nothing and
/// the adjusted values are those numbers. A sweep in the wrong direction would
/// flatten them all to `0.006`.
#[test]
fn benjamini_hochberg_matches_a_hand_computation() {
    let mut f = Failures::default();
    let bh = benjamini_hochberg(&P);
    let expected = [0.006, 0.024, 0.042, 0.06, 0.36, 0.81];
    for (i, &want) in expected.iter().enumerate() {
        f.close(|| format!("bh[{i}]"), bh[i], want, 1e-12);
    }

    // And a case where the minimum *does* bite: a large third p-value drags its
    // own adjusted value up but must not drag the smaller ones with it.
    let q = [0.01, 0.02, 0.9];
    let bh2 = benjamini_hochberg(&q);
    // raw: 0.03, 0.03, 0.9 -> running min from the top leaves 0.03, 0.03, 0.9
    f.close(|| "bh2[0]".into(), bh2[0], 0.03, 1e-12);
    f.close(|| "bh2[1]".into(), bh2[1], 0.03, 1e-12);
    f.close(|| "bh2[2]".into(), bh2[2], 0.9, 1e-12);

    f.finish("BH hand computation");
}

/// The conservatism chain. Each of these is a theorem, not a coincidence.
#[test]
fn the_corrections_form_a_conservatism_chain() {
    let mut f = Failures::default();
    let mut rng = ChaCha8Rng::seed_from_u64(20_260_105);
    let u = Uniform::new(0.0, 1.0).unwrap();

    for m in [1_usize, 2, 5, 20, 200] {
        let p: Vec<f64> = (0..m).map(|_| u.sample(&mut rng)).collect();
        let (b, h, bh, by) = (
            bonferroni(&p),
            holm(&p),
            benjamini_hochberg(&p),
            benjamini_yekutieli(&p),
        );
        let hs = holm_sidak(&p);
        let sd = sidak(&p);

        for i in 0..m {
            // Holm is uniformly at least as powerful as Bonferroni.
            f.check(h[i] <= b[i] + 1e-12, || {
                format!("m={m}: holm[{i}] = {} exceeds bonferroni {}", h[i], b[i])
            });
            // Holm-Sidak likewise dominates Sidak.
            f.check(hs[i] <= sd[i] + 1e-12, || {
                format!("m={m}: holm_sidak[{i}] = {} exceeds sidak {}", hs[i], sd[i])
            });
            // Sidak is slightly less conservative than Bonferroni.
            f.check(sd[i] <= b[i] + 1e-12, || {
                format!("m={m}: sidak[{i}] = {} exceeds bonferroni {}", sd[i], b[i])
            });
            // FDR control is less conservative than FWER control.
            f.check(bh[i] <= h[i] + 1e-12, || {
                format!("m={m}: bh[{i}] = {} exceeds holm {}", bh[i], h[i])
            });
            // And BY sits between BH and Bonferroni.
            f.check(by[i] >= bh[i] - 1e-12, || {
                format!("m={m}: by[{i}] = {} is below bh {}", by[i], bh[i])
            });
            // Every adjusted value is a probability and never below the raw one.
            for (name, v) in [
                ("bonferroni", b[i]),
                ("sidak", sd[i]),
                ("holm", h[i]),
                ("holm_sidak", hs[i]),
                ("bh", bh[i]),
                ("by", by[i]),
            ] {
                f.check((0.0..=1.0).contains(&v), || {
                    format!("m={m}: {name}[{i}] = {v} is not a probability")
                });
                f.check(v >= p[i] - 1e-12, || {
                    format!("m={m}: {name}[{i}] = {v} is below the raw p-value {}", p[i])
                });
            }
        }
    }
    f.finish("conservatism chain");
}

/// Adjusted p-values must be monotone in the raw ones: if `p_i <= p_j` then the
/// adjusted values must be ordered the same way. Otherwise a stronger result
/// could be reported as weaker.
#[test]
fn adjustment_preserves_the_ordering_of_the_raw_p_values() {
    let mut f = Failures::default();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let u = Uniform::new(0.0, 1.0).unwrap();
    for m in [3_usize, 10, 50] {
        let p: Vec<f64> = (0..m).map(|_| u.sample(&mut rng)).collect();
        for c in ALL {
            let q = c.adjust(&p);
            for i in 0..m {
                for j in 0..m {
                    if p[i] <= p[j] {
                        f.check(q[i] <= q[j] + 1e-12, || {
                            format!(
                                "{c:?} at m={m}: p[{i}]={} <= p[{j}]={} but adjusted {} > {}",
                                p[i], p[j], q[i], q[j]
                            )
                        });
                    }
                }
            }
        }
    }
    f.finish("monotonicity");
}

#[test]
fn a_single_test_is_left_alone() {
    let mut f = Failures::default();
    for c in ALL {
        let q = c.adjust(&[0.037]);
        f.close(
            || format!("{c:?} on one test is the identity"),
            q[0],
            0.037,
            1e-12,
        );
    }
    f.finish("m = 1");
}

#[test]
fn degenerate_input_yields_an_empty_result_rather_than_a_panic() {
    let mut f = Failures::default();
    for c in ALL {
        for bad in [vec![], vec![f64::NAN], vec![0.5, -0.1], vec![0.5, 1.5]] {
            let q = c.adjust(&bad);
            f.check(q.is_empty(), || {
                format!("{c:?} accepted {bad:?} and returned {q:?}")
            });
        }
        // A valid input still works after the invalid ones.
        f.check(c.adjust(&[0.1, 0.2]).len() == 2, || {
            format!("{c:?} failed on valid input")
        });
    }
    // An out-of-range alpha rejects nothing rather than panicking.
    f.check(
        !reject_at(&P, f64::NAN, Correction::Holm).iter().any(|&b| b),
        || "a NaN alpha rejected something".into(),
    );
    f.finish("degenerate input");
}

/// The operational guarantee: under the **global null** — every hypothesis true,
/// so every p-value uniform — Benjamini–Hochberg must reject at most `alpha` of
/// the time, and in that regime the FDR equals the family-wise rate.
#[test]
fn benjamini_hochberg_holds_the_false_discovery_rate_under_the_null() {
    let mut rng = ChaCha8Rng::seed_from_u64(20_260_105);
    let u = Uniform::new(0.0, 1.0).unwrap();
    let alpha = 0.1;
    let (m, trials) = (50_usize, 4_000);

    let mut any_rejection = 0;
    for _ in 0..trials {
        let p: Vec<f64> = (0..m).map(|_| u.sample(&mut rng)).collect();
        if reject_at(&p, alpha, Correction::BenjaminiHochberg)
            .iter()
            .any(|&r| r)
        {
            any_rejection += 1;
        }
    }
    let rate = any_rejection as f64 / trials as f64;
    // Under the complete null the FDR and the FWER coincide, so the rejection
    // rate must sit at or below alpha. The standard error at 4000 trials is
    // about 0.005, so allow three of those above the nominal level.
    assert!(
        rate <= alpha + 0.015,
        "BH rejected something in {rate:.4} of null trials, above the nominal {alpha}"
    );

    // Bonferroni, being stricter, must reject strictly less often.
    let mut bonf_rejections = 0;
    let mut rng2 = ChaCha8Rng::seed_from_u64(20_260_105);
    for _ in 0..trials {
        let p: Vec<f64> = (0..m).map(|_| u.sample(&mut rng2)).collect();
        if reject_at(&p, alpha, Correction::Bonferroni)
            .iter()
            .any(|&r| r)
        {
            bonf_rejections += 1;
        }
    }
    assert!(
        bonf_rejections <= any_rejection,
        "Bonferroni rejected more often ({bonf_rejections}) than BH ({any_rejection})"
    );
}

/// And the other half: when there *are* real effects, the correction must still
/// find them. A test that rejects nothing controls every error rate trivially.
#[test]
fn real_effects_survive_correction() {
    let mut f = Failures::default();
    // Ten genuine effects at p = 1e-6, ninety nulls spread over [0,1].
    let mut p: Vec<f64> = vec![1e-6; 10];
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let u = Uniform::new(0.0, 1.0).unwrap();
    p.extend((0..90).map(|_| u.sample(&mut rng)));

    for c in ALL {
        let rejected = reject_at(&p, 0.05, c);
        let found = rejected[..10].iter().filter(|&&r| r).count();
        f.check(found == 10, || {
            format!("{c:?} found only {found} of 10 genuine effects at p = 1e-6")
        });
        // And BH should find at least as many total discoveries as Bonferroni.
        if c == Correction::BenjaminiHochberg {
            let bh_total = rejected.iter().filter(|&&r| r).count();
            let bonf_total = reject_at(&p, 0.05, Correction::Bonferroni)
                .iter()
                .filter(|&&r| r)
                .count();
            f.check(bh_total >= bonf_total, || {
                format!("BH made {bh_total} discoveries, Bonferroni {bonf_total}")
            });
        }
    }
    f.finish("power retained");
}

#[test]
fn the_dispatch_enum_agrees_with_the_free_functions() {
    let mut f = Failures::default();
    let pairs: [(Correction, Vec<f64>); 6] = [
        (Correction::Bonferroni, bonferroni(&P)),
        (Correction::Sidak, sidak(&P)),
        (Correction::Holm, holm(&P)),
        (Correction::HolmSidak, holm_sidak(&P)),
        (Correction::BenjaminiHochberg, benjamini_hochberg(&P)),
        (Correction::BenjaminiYekutieli, benjamini_yekutieli(&P)),
    ];
    for (c, direct) in pairs {
        let via_enum = c.adjust(&P);
        f.check(via_enum == direct, || {
            format!("{c:?}: enum gave {via_enum:?}, function gave {direct:?}")
        });
    }
    // The FWER/FDR classification is part of the API contract.
    f.check(Correction::Bonferroni.controls_fwer(), || {
        "Bonferroni is FWER".into()
    });
    f.check(Correction::Holm.controls_fwer(), || "Holm is FWER".into());
    f.check(!Correction::BenjaminiHochberg.controls_fwer(), || {
        "BH is FDR".into()
    });
    f.check(!Correction::BenjaminiYekutieli.controls_fwer(), || {
        "BY is FDR".into()
    });
    f.finish("dispatch");
}
