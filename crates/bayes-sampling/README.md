# bayes-sampling

**Exact samplers for distributions that do not sample easily.**

## The problem

The families in `bayes-distribution-ext` each know how to draw from themselves,
because each has a closed-form inverse CDF or a known algorithm. Everything else
does not: an unnormalized posterior, a density you wrote down five minutes ago, a
categorical over fifty thousand outcomes, a truncated or reweighted version of
something standard.

The usual next step is MCMC — and then burn-in, thinning, autocorrelation, trace
plots, and a convergence diagnostic you have to trust. For a *univariate* density
that is almost always more machinery than the problem needs.

Everything in this crate is **exact**. A draw that survives is distributed
precisely as the target, and consecutive draws are independent. There is nothing
to converge and nothing to diagnose.

## Choosing

| you have | use | cost per draw |
|----------|-----|---------------|
| finite outcomes with known weights | `AliasTable` | O(1), one uniform |
| a density and a known envelope | `RejectionSampler` | O(1/acceptance) |
| a **log-concave** density and its derivative | `AdaptiveRejection` | tends to 1 proposal |
| an invertible CDF | nothing here — call `inverse_cdf(u)` | O(1) |

The ordering is deliberate. If the CDF inverts, invert it. If it does not but the
density is log-concave, `AdaptiveRejection` needs no tuning at all. Only when
neither holds do you construct an envelope by hand.

## Why adaptive rejection is the interesting one

Plain rejection sampling makes you supply a good envelope, and for an irregular
density you usually cannot: you do not know where the mode is, how heavy the
tails are, or what constant makes the bound valid. Choose badly and you either
bias the result or throw away 99% of proposals.

If `ln f` is **concave**, two facts come for free: a tangent lies above the
density everywhere, so tangents form an upper hull that `exp` turns into a
piecewise-exponential density you can sample exactly; and a chord lies below it,
giving a free "squeeze" test that accepts most proposals without evaluating the
density at all. Every rejection then *adds* a point to the hull, tightening it
exactly where the fit was worst.

The worked example sampling a non-conjugate Poisson posterior reaches **99.9%
acceptance after 129 density evaluations across 40,000 draws** — the squeeze test
avoids 99.7% of the work. Log-concavity covers the normal, exponential, gamma and
Weibull with shape ≥ 1, beta with both shapes ≥ 1, logistic, Gumbel, and most
posteriors built from those likelihoods.

## The one mistake that does not announce itself

An envelope that dips below the target does not panic, does not error, and does
not make the output look wrong. It silently under-represents the region where the
envelope failed. Both samplers track it — `RejectionSampler::envelope_was_breached`
and `AdaptiveRejection::hull_was_breached` — and both should be **checked**, not
assumed. The second doubles as a log-concavity test, since a density above its own
upper hull was never concave.

## Verifying it

`tests/correctness.rs` points each sampler at a target whose exact distribution
bayes already computes independently, and judges it by Kolmogorov–Smirnov or
Pearson chi-squared with thresholds derived rather than tuned. Adaptive rejection
is checked against the exact CDFs of `Normal`, `Gamma` and `Beta`; the alias table
against `Categorical`; and a deliberately bimodal density is confirmed to trip the
breach flag.

```sh
cargo run -p bayes-sampling --example irregular_posterior
```

## The boundary

Sampling *from a density you can evaluate* is this crate's job. Exploring a
posterior you cannot evaluate pointwise — or one in enough dimensions that
rejection acceptance decays exponentially — is an inference layer's, and stays
out of bayes along with MCMC and variational methods.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
