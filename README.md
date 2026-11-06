# bayes

Closed-form probability and statistics for Rust.

**Write a density once; evaluate it eagerly, or stage it into a compiled model.**

```rust
use bayes_distribution_ext::{Continuous, ContinuousCdf, Moments, Normal};

let n = Normal::new(2.0, 3.0)?;
n.pdf(2.0);              // density
n.inverse_cdf(0.975);    // quantile
n.entropy();             // closed form, not estimated from draws

// The log-CDF stays finite 40 sigma out, where the naive form is already gone.
let z = Normal::new(0.0, 1.0)?;
let ln_tail: f64 = z.ln_cdf(-40.0);     // ≈ -804.608
let naive: f64 = z.cdf(-40.0);
assert!(ln_tail.is_finite());
assert_eq!(naive.ln(), f64::NEG_INFINITY);
```

(That example is compiled and asserted as a test, in
`crates/bayes-distribution-ext/tests/readme_example.rs`.)

## The Problem

Probability code tends to be written twice, and the two copies disagree.

The same log-density is needed in two incompatible places. A statistical summary
wants it **eagerly**, on host `f64`, right now. A model's log-likelihood wants it
**staged** — pushed into a tensor runtime or an expression graph, to be compiled
and differentiated later. Those are different enough that the usual answer is two
implementations: one in your stats library, one hand-written inside your model
code. Then they drift. A `ln_gamma` normalizer gets dropped from one because it is
constant in the parameters, a variance is corrected in the other, and the model's
reported likelihood stops matching the library's density by an amount nobody
notices.

The second problem is reaching for simulation where a closed form exists. It is
easy to find a distribution type that can *sample* but cannot tell you its CDF,
quantile, entropy, or its divergence from another distribution — so those get
estimated from draws, introducing Monte Carlo error into a quantity that has an
exact answer, and burning a thousand samples to get three digits of a number that
is one `ln_gamma` call away.

The third is tails. The textbook formula is usually the wrong way to evaluate the
formula. `1.0 - cdf(x)` has no significant digits left in the upper tail,
`pdf(x)` underflows to zero while `ln_pdf(x)` is perfectly finite, and
`cdf(-40.0).ln()` is `-inf` when the answer is about `-804.6`. Code that is
correct across the range you tested and silently wrong in the tail is the
characteristic failure of hand-rolled statistics.

## The Approach

**Closed form first.** If a quantity has an exact form, bayes ships the exact
form: CDFs and quantiles, moments and entropies, divergences between families,
conjugate posteriors, confidence intervals, and hypothesis tests. Sampling is
available, but it is not how the library answers a question that algebra can
answer.

**One formula, any carrier.** Every density is written once against
[euclid](https://github.com/urban-fabric/euclid)'s numeric contract rather than
against `f64`. A formula whose only bound is `Real` compiles against an eager
scalar *and* against a deferred carrier — a graph node, a tensor element, an
expression handle — with no second implementation to keep in sync. The eager
families and the staged kernels are then not two copies of the Normal density;
they are the same closed form reached through two doors, and the test suite
asserts they agree.

**Log domain by default.** `ln_pdf` is the primitive and `pdf` is derived from
it. Survival functions (`sf`, `ln_sf`) exist so nobody writes `1 - cdf`. The far
tails are explicitly tested, because they are where the naive form dies.

## Why the Crates Split in Two

Several crates come in pairs — `bayes-core` / `bayes-core-ext`,
`bayes-distribution` / `bayes-distribution-ext`. The split is not layering for
its own sake; it tracks exactly one question:

> **Does this code need a host `bool`?**

A deferred carrier cannot answer `if x < 0.0`. A comparison on a graph node
produces another node, not a branch — you get a mask, which you `select` on. You
also cannot `Copy` a symbolic handle, write `2.0` into an expression tree, or
return a `Result` from a kernel that will be evaluated a million times inside a
compiled graph. Those three restrictions are what make staged execution possible,
and they are inherited from euclid.

So the plain crate holds everything that survives those restrictions, and the
`-ext` crate holds everything that needs to break them:

|  | plain (`bayes-core`, `bayes-distribution`) | `-ext` (`bayes-core-ext`, `bayes-distribution-ext`) |
|---|---|---|
| **runs on** | any euclid carrier — `f64`, `f32`, tensors, graph nodes | host scalars only |
| **constructors** | infallible: `Normal::new(mu, sigma)` | validating: `Normal::new(mu, sigma) -> Result<_, BayesError>` |
| **comparisons** | masks, via `select` | host `bool`, so real branching |
| **support checks** | none — the caller guarantees the point is on support | checked, and out-of-support returns the limit value |
| **has** | parameter structs, `ln_pdf`, algebraic CDFs | sampling, fitting, moments, combinators, conjugate updates |

Concretely, `bayes_distribution::Normal::new(0.0, 1.0)` cannot fail and cannot
validate, because on a deferred carrier there is nothing to validate *yet* — the
parameters may not have been computed. `bayes_distribution_ext::Normal::new(0.0,
1.0)` returns a `Result` and rejects a negative standard deviation, because on a
host scalar it can.

The two are not independent implementations. `-ext` depends on the plain crate
and on `bayes-kernels`, so the eager `Normal`'s density *is* the carrier-generic
one with validation and support handling wrapped around it.

**Which do you want?** If you are writing ordinary statistics on `&[f64]`, you
want the `-ext` crates — the validating, sampling, fitting surface. The plain
crates matter when you are writing a formula that has to run somewhere `f64` does
not reach.

## The Crates

| crate | holds |
|-------|-------|
| `bayes-kernels` | the closed forms themselves — ~200 carrier-generic log-density, CDF, and divergence kernels, taking pre-validated parameters and returning algebra. No validation, no branching, no sampling. |
| `bayes-core` | the trait spine, carrier-generic: `Continuous`, `Discrete`, `ContinuousCdf`, `Support`, `Moments`, the constraint vocabulary, and the probability-vector/scoring-rule surface. |
| `bayes-core-ext` | the eager additions to the spine: `Estimate` (fitting from a sample) and the host-side probability utilities. |
| `bayes-distribution` | 31 lazy families (22 continuous, 9 discrete) — parameter structs with carrier-generic density and CDF impls, plus euclid's explicit-key sampler contract. |
| `bayes-distribution-ext` | those families eagerly, plus `Triangular` (32 in all): validating constructors, `rand` sampling, moments, entropy, quantiles, fitting, combinators (truncation, mixtures, order statistics), bijective transforms, exponential-family sufficient statistics, and 6 conjugate models. |
| `bayes-divergence` | KL, reverse KL, Hellinger, Rényi, χ², total variation, Jensen–Shannon, and the general *f*-divergence, as closed forms between families — plus Shannon/joint/conditional entropy and mutual information. |
| `bayes-statistics` | statistics over data: summaries and order statistics, rank correlation, effect sizes, weighted and log-weighted reductions with ESS, OLS, bootstrap/jackknife, 19 hypothesis tests, closed-form and distribution-free confidence intervals, Cramér–Rao diagnostics, KDE, ECDF, and streaming accumulators (`RunningStats`, `TDigest`). |
| `bayes-sampling` | exact samplers for the distributions that do not sample easily: O(1) alias tables, rejection sampling from an envelope, and adaptive rejection for log-concave densities. No burn-in, no autocorrelation. |
| `bayes-process` | stochastic processes: Gaussian random walks, Brownian paths, homogeneous and inhomogeneous Poisson processes, and the GP covariance kernels (RBF, Matérn, periodic, linear). |
| `bayes-utils` | **dev-only** (`publish = false`): the shared test probes — failure accumulation, quadrature, numerical derivatives, goodness-of-fit, and the canonical family sweep every suite iterates. Never a runtime dependency. |

There is no umbrella `bayes` crate. Depend on the crates you use.

## Examples

Worked problems, each asserting every number it prints:

```sh
cargo run -p bayes-statistics --example ab_test          # frequentist and Bayesian, side by side
cargo run -p bayes-statistics --example sample_size      # power analysis, checked by simulation
cargo run -p bayes-statistics --example reliability      # Weibull lifetimes, hazard rates
cargo run -p bayes-statistics --example quality_control  # control limits and capability
cargo run -p bayes-process    --example arrivals         # a Poisson process, three ways
cargo run -p bayes-sampling   --example irregular_posterior  # a non-conjugate posterior, sampled exactly
```

`ab_test` is the one to read first: it answers the same question two ways and
prints both, so the one-sided frequentist p-value landing on the Bayesian
posterior tail to three decimals is visible rather than asserted out of sight.

## The Boundary

bayes is **scalar and closed-form**. Two things are deliberately elsewhere:

- **Anything matrix-coupled** — multivariate normal, Dirichlet, Wishart, GP
  regression, Markov chains, multiple regression — lives in the sibling crate
  `ito`. bayes depends on euclid's numeric tier only, never the container tier,
  which is what keeps it a dependency-light scalar library. The *kernels* for
  those densities are here; the matrix algebra that completes them is not.
- **Inference** — MCMC, variational inference, optimization. bayes gives you
  exact posteriors where conjugacy provides them, and the log-densities an
  inference layer needs. Running a sampler over them is a higher layer's job.

Also not here: special functions. `ln_gamma`, `digamma`, `beta_reg`, `gamma_lr`
and friends belong to euclid; bayes composes closed forms out of them.

## Status

**Beta.** The trait spine is still moving. Version the workspace, read the PR
titles, and expect to track rather than pin. euclid is tracked from git `main`
with a semver assertion, so `Cargo.lock` is the record of which euclid a given
bayes commit was tested against.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development loop — the repository
is managed with [Jujutsu](https://github.com/jj-vcs/jj) and the checks run
locally rather than in CI.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE).
