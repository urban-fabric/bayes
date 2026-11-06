//! Adversarial sweep over every eager discrete family.
//!
//! The load-bearing check is the closed-form CDF against **direct summation of
//! the PMF**. Those are independent code paths (a regularized incomplete beta or
//! gamma on one side, a plain loop on the other), so agreement is evidence both
//! are right. The analytic moments and entropy get the same treatment, summed
//! term by term over the support.
//!
//! Parameters include the degenerate ends: `p = 0` and `p = 1`, `n = 0`,
//! `Binomial(10⁶, 10⁻⁹)` where the mass sits almost entirely on zero,
//! `Poisson(10⁵)` where it sits far from the origin and the log-factorials are
//! large, and a `Categorical` carrying an exact zero weight.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

// The glob brings every family constructor into scope for the validation sweep.
use bayes_distribution_ext::{Moments, Support, *};
use bayes_utils::families::{DiscCase, discrete_cases, summation_limit};
use bayes_utils::{Failures, rel_error as rel};

// ---------------------------------------------------------------------------

#[test]
fn pmf_is_a_probability_and_sums_to_one() {
    let mut f = Failures::default();
    for DiscCase { name, dist } in discrete_cases() {
        let top = summation_limit(&*dist);
        let mut total = 0.0;
        for k in dist.lo()..=top {
            let m = dist.pmf(k);
            f.check(!m.is_nan() && (0.0..=1.0).contains(&m), || {
                format!("{name}: pmf({k}) = {m} outside [0,1]")
            });
            total += m;
        }
        f.check((total - 1.0).abs() < 1e-9, || {
            format!(
                "{name}: Σpmf over [{}, {top}] = {total:.15}, expected 1",
                dist.lo()
            )
        });
    }
    f.finish("pmf normalization");
}

#[test]
fn ln_pmf_agrees_with_pmf() {
    let mut f = Failures::default();
    for DiscCase { name, dist } in discrete_cases() {
        let top = summation_limit(&*dist).min(dist.lo() + 5_000);
        for k in dist.lo()..=top {
            let m = dist.pmf(k);
            if m > 1e-300 {
                let lm = dist.ln_pmf(k);
                f.check(rel(lm, m.ln()) < 1e-9, || {
                    format!("{name}: ln_pmf({k}) = {lm}, pmf.ln() = {}", m.ln())
                });
            }
        }
    }
    f.finish("ln_pmf consistency");
}

/// The load-bearing check: closed-form CDF against a running sum of the PMF.
#[test]
fn cdf_matches_the_running_sum_of_the_pmf() {
    let mut f = Failures::default();
    for DiscCase { name, dist } in discrete_cases() {
        let top = summation_limit(&*dist);
        let mut running = 0.0;
        for k in dist.lo()..=top {
            running += dist.pmf(k);
            let c = dist.cdf(k);
            f.check((c - running).abs() < 1e-9, || {
                format!(
                    "{name}: cdf({k}) = {c:.15} but Σpmf(≤{k}) = {running:.15} \
                     (diff {:.3e})",
                    (c - running).abs()
                )
            });
        }
    }
    f.finish("cdf vs pmf summation");
}

#[test]
fn cdf_is_monotone_in_the_unit_interval_and_sf_is_its_complement() {
    let mut f = Failures::default();
    for DiscCase { name, dist } in discrete_cases() {
        let top = summation_limit(&*dist);
        let mut prev = 0.0;
        for k in dist.lo()..=top {
            let c = dist.cdf(k);
            f.check(!c.is_nan() && (0.0..=1.0).contains(&c), || {
                format!("{name}: cdf({k}) = {c} outside [0,1]")
            });
            f.check(c >= prev - 1e-12, || {
                format!("{name}: cdf decreased at k={k}: {prev} -> {c}")
            });
            prev = c;
            let s = dist.sf(k);
            f.check((c + s - 1.0).abs() < 1e-9, || {
                format!("{name}: cdf({k}) + sf({k}) = {}, expected 1", c + s)
            });
        }
        // The CDF saturates at the top of a bounded support.
        if dist.hi() <= 400_000 {
            let c = dist.cdf(dist.hi());
            f.check((c - 1.0).abs() < 1e-9, || {
                format!("{name}: cdf(max={}) = {c}, expected 1", dist.hi())
            });
        }
    }
    f.finish("discrete cdf shape");
}

/// `inverse_cdf(p)` is by contract the smallest `k` with `cdf(k) ≥ p`.
#[test]
fn inverse_cdf_is_the_smallest_k_whose_cdf_reaches_p() {
    let mut f = Failures::default();
    for DiscCase { name, dist } in discrete_cases() {
        for &p in &[1e-9, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0 - 1e-12] {
            let k = dist.inverse_cdf(p);
            f.check(dist.cdf(k) >= p - 1e-12, || {
                format!(
                    "{name}: inverse_cdf({p}) = {k} but cdf({k}) = {} < {p}",
                    dist.cdf(k)
                )
            });
            if k > dist.lo() {
                let below = dist.cdf(k - 1);
                f.check(below < p + 1e-12, || {
                    format!(
                        "{name}: inverse_cdf({p}) = {k} is not minimal: \
                         cdf({}) = {below} already reaches {p}",
                        k - 1
                    )
                });
            }
        }
    }
    f.finish("discrete quantile");
}

/// Analytic mean and variance against term-by-term summation.
#[test]
fn analytic_moments_match_summation() {
    let mut f = Failures::default();
    for DiscCase { name, dist } in discrete_cases() {
        let (mean, var) = (dist.mean(), dist.variance());
        if !mean.is_finite() || !var.is_finite() {
            continue;
        }
        let top = summation_limit(&*dist);
        let mut m1 = 0.0;
        let mut m2 = 0.0;
        for k in dist.lo()..=top {
            let w = dist.pmf(k);
            m1 += k as f64 * w;
            m2 += (k as f64 - mean).powi(2) * w;
        }
        f.check(rel(m1, mean) < 1e-7, || {
            format!("{name}: Σk·pmf = {m1:.9}, analytic mean = {mean:.9}")
        });
        f.check(rel(m2, var) < 1e-6, || {
            format!("{name}: Σ(k-μ)²·pmf = {m2:.9}, analytic variance = {var:.9}")
        });
    }
    f.finish("discrete moments");
}

/// Shannon entropy against `-Σ p ln p`.
#[test]
fn analytic_entropy_matches_summation() {
    let mut f = Failures::default();
    for DiscCase { name, dist } in discrete_cases() {
        let h = dist.entropy();
        if !h.is_finite() {
            continue;
        }
        let top = summation_limit(&*dist);
        let mut num = 0.0;
        for k in dist.lo()..=top {
            let w = dist.pmf(k);
            if w > 0.0 {
                num -= w * w.ln();
            }
        }
        f.check(rel(num, h) < 1e-6, || {
            format!("{name}: -Σp·ln p = {num:.9}, analytic entropy = {h:.9}")
        });
    }
    f.finish("discrete entropy");
}

#[test]
fn mass_outside_a_bounded_support_is_zero() {
    let mut f = Failures::default();
    for DiscCase { name, dist } in discrete_cases() {
        let hi = dist.hi();
        if hi < u64::MAX - 4 {
            for k in [hi + 1, hi + 2] {
                let m = dist.pmf(k);
                f.check(m == 0.0, || {
                    format!("{name}: pmf({k}) = {m} above support max {hi}")
                });
                let c = dist.cdf(k);
                f.check((c - 1.0).abs() < 1e-9, || {
                    format!("{name}: cdf({k}) = {c} above support max {hi}, expected 1")
                });
            }
        }
    }
    f.finish("mass outside support");
}

#[test]
fn discrete_constructors_reject_invalid_parameters() {
    let mut f = Failures::default();
    let mut bad = |label: &str, rejected: bool| {
        f.check(rejected, || format!("{label} was accepted"));
    };
    bad("Bernoulli(-0.1)", Bernoulli::new(-0.1).is_err());
    bad("Bernoulli(1.1)", Bernoulli::new(1.1).is_err());
    bad("Bernoulli(NaN)", Bernoulli::new(f64::NAN).is_err());
    bad("Binomial(10,-0.1)", Binomial::new(10, -0.1).is_err());
    bad("Binomial(10,1.1)", Binomial::new(10, 1.1).is_err());
    bad("Binomial(10,NaN)", Binomial::new(10, f64::NAN).is_err());
    bad("Poisson(-1)", Poisson::new(-1.0).is_err());
    bad("Poisson(0)", Poisson::new(0.0).is_err());
    bad("Poisson(NaN)", Poisson::new(f64::NAN).is_err());
    bad("Poisson(inf)", Poisson::new(f64::INFINITY).is_err());
    bad("Geometric(0)", Geometric::new(0.0).is_err());
    bad("Geometric(-0.1)", Geometric::new(-0.1).is_err());
    bad("Geometric(1.1)", Geometric::new(1.1).is_err());
    bad(
        "NegativeBinomial(0,0.5)",
        NegativeBinomial::new(0.0, 0.5).is_err(),
    );
    bad(
        "NegativeBinomial(-1,0.5)",
        NegativeBinomial::new(-1.0, 0.5).is_err(),
    );
    bad(
        "NegativeBinomial(5,0)",
        NegativeBinomial::new(5.0, 0.0).is_err(),
    );
    bad(
        "NegativeBinomial(5,1.1)",
        NegativeBinomial::new(5.0, 1.1).is_err(),
    );
    bad(
        "Hypergeometric(10,11,5)",
        Hypergeometric::new(10, 11, 5).is_err(),
    );
    bad(
        "Hypergeometric(10,5,11)",
        Hypergeometric::new(10, 5, 11).is_err(),
    );
    bad(
        "BetaBinomial(10,0,1)",
        BetaBinomial::new(10, 0.0, 1.0).is_err(),
    );
    bad(
        "BetaBinomial(10,-1,1)",
        BetaBinomial::new(10, -1.0, 1.0).is_err(),
    );
    bad(
        "BetaBinomial(10,NaN,1)",
        BetaBinomial::new(10, f64::NAN, 1.0).is_err(),
    );
    bad("Categorical[]", Categorical::new(&[]).is_err());
    bad("Categorical[0,0]", Categorical::new(&[0.0, 0.0]).is_err());
    bad("Categorical[-1,2]", Categorical::new(&[-1.0, 2.0]).is_err());
    bad(
        "Categorical[NaN,1]",
        Categorical::new(&[f64::NAN, 1.0]).is_err(),
    );
    bad(
        "Categorical[inf,1]",
        Categorical::new(&[f64::INFINITY, 1.0]).is_err(),
    );
    f.finish("discrete parameter validation");
}

/// `DiscreteUniform` is indexed by `i64` rather than `u64`, so it sits outside the
/// sweep above and gets the same identities checked directly.
///
/// Note the convention: this family's range is the **half-open** `[lo, hi)`, so it
/// carries `hi − lo` values and `Support::max()` is `hi − 1`. That differs from
/// every other bounded family here (`Binomial`, `Hypergeometric`, `BetaBinomial`
/// and the continuous `Uniform` are all inclusive), which makes it an easy thing
/// to get wrong at a call site — but it is documented on the type and internally
/// consistent, so it is asserted as-is rather than treated as a defect.
#[test]
fn discrete_uniform_over_signed_counts() {
    use bayes_distribution_ext::{IntegerDiscrete, IntegerDiscreteCdf};
    let mut f = Failures::default();
    for (lo, hi) in [(0_i64, 5_i64), (-7, -3), (-4, 4), (0, 1), (-1, 1)] {
        let d = DiscreteUniform::new(lo, hi).unwrap();
        let n = (hi - lo) as f64;
        let top = hi - 1; // inclusive top of the half-open range
        f.check(<DiscreteUniform as Support<i64>>::max(&d) == top, || {
            format!(
                "DiscreteUniform({lo},{hi}): Support::max() = {}, expected {top}",
                <DiscreteUniform as Support<i64>>::max(&d)
            )
        });
        let mut total = 0.0;
        for k in lo..=top {
            let m = IntegerDiscrete::pmf(&d, k);
            f.check((m - 1.0 / n).abs() < 1e-15, || {
                format!(
                    "DiscreteUniform({lo},{hi}): pmf({k}) = {m}, expected {}",
                    1.0 / n
                )
            });
            total += m;
            let c = IntegerDiscreteCdf::cdf(&d, k);
            f.check((c - total).abs() < 1e-12, || {
                format!("DiscreteUniform({lo},{hi}): cdf({k}) = {c}, Σpmf = {total}")
            });
        }
        f.check((total - 1.0).abs() < 1e-12, || {
            format!("DiscreteUniform({lo},{hi}): Σpmf = {total}, expected 1")
        });
        for outside in [lo - 1, hi] {
            f.check(IntegerDiscrete::pmf(&d, outside) == 0.0, || {
                format!("DiscreteUniform({lo},{hi}): pmf({outside}) is nonzero, outside support")
            });
        }
        // Closed forms for the half-open range: mean (lo + hi − 1)/2, variance (n²−1)/12.
        let mean = (lo as f64 + top as f64) / 2.0;
        f.check((Moments::mean(&d) - mean).abs() < 1e-12, || {
            format!(
                "DiscreteUniform({lo},{hi}): mean = {}, expected {mean}",
                Moments::mean(&d)
            )
        });
        let var = (n * n - 1.0) / 12.0;
        f.check((Moments::variance(&d) - var).abs() < 1e-12, || {
            format!(
                "DiscreteUniform({lo},{hi}): variance = {}, expected {var}",
                Moments::variance(&d)
            )
        });
    }
    f.check(DiscreteUniform::new(5, 4).is_err(), || {
        "DiscreteUniform(5,4) inverted range was accepted".into()
    });
    f.check(DiscreteUniform::new(3, 3).is_err(), || {
        "DiscreteUniform(3,3) carries no values under [lo,hi) and must be rejected".into()
    });
    f.finish("DiscreteUniform");
}
