# bayes

**The closed-form probability & statistics foundation for Rust.**

`bayes` is a standalone, domain-free numerics library for direct formulas,
finite-sample algorithms, exact/direct sampling, and bounded streaming summaries.

## What's in it

- **Probability distributions** on a decomposed trait spine — continuous (Normal,
  LogNormal, Exponential, Gamma, Beta, Cauchy, Laplace, Student-t, χ², Weibull,
  Pareto, F, Triangular, Rayleigh, Gumbel, Half-Normal, Inverse-Gamma, Logistic,
  Lomax, Frechet, Wald, VonMises, Uniform, Uniform2D), discrete (Bernoulli,
  Binomial, Poisson, Geometric, DiscreteUniform, NegativeBinomial, Hypergeometric,
  Categorical, BetaBinomial), multivariate (MVN, multivariate Student-t,
  Dirichlet, Multinomial, Dirichlet-multinomial counts, Wishart,
  Inverse-Wishart), plus combinators (Mixture, OrderStatistic, Truncated,
  Transformed). MVNs include closed-form marginal, conditional,
  affine-transform, batch-likelihood, and Gaussian-product / precision-fusion
  operations. Multivariate Student-t distributions include closed-form
  marginals, conditionals, affine transforms, Mahalanobis geometry, entropy,
  and batch likelihoods. Wishart families include matrix densities, entropy/KL,
  expected log-determinants, entry covariances, and inverse-family transforms.
  Dirichlet distributions include component/sum Beta marginals,
  category lumping, conditional sub-simplex laws, and full Dirichlet-multinomial
  predictive count distributions. Directional models include normalized von
  Mises/von Mises-Fisher, spherical-uniform, and Watson-on-S² distributions with
  closed-form moments, entropy/KL, sampling, and same-family product algebra
  where it stays normalized. Kent/Bingham expose closed-form potential algebra
  (ratios, gradients, spectral/canonical forms, identity-shift invariance, rank-one
  Watson conversion) plus normalized special cases where the partition function,
  moments, and entropy are analytic; the generic Kent/Bingham normalizers remain
  outside the closed-form surface.
  Normalized scalar distributions provide PDF/PMF, CDF/quantile/survival
  (incl. stable `ln_cdf`/`ln_sf`/`isf`) plus reliability helpers
  (`hazard`, `ln_hazard`, `cumulative_hazard`),
  analytic moments (`Option`-typed — Cauchy honestly has no mean), and `rand` sampling.
  Same-family pairs also expose a closed-form `kl_divergence` (and Jeffreys, JS,
  Hellinger, total-variation, χ², Rényi, mutual information). The **f-divergence
  umbrella** (`f_divergence`) unifies these into one generic `D_f(p ‖ q)` over
  probability vectors with pluggable convex generators, plus a stochastic
  `Channel` carrying the data-processing inequality. Probability helpers validate
  simplex vectors, row-stochastic kernels/matrices, and expose proper scoring
  rules (`log_score`, `brier_score`, `spherical_score`, ranked probability score)
  for categorical forecasts.
  Functional adapters (`ContinuousFn`, `DiscreteFn`, `ContinuousDensity`,
  `DiscreteMass`) lift closures into the same trait spine for custom likelihoods.
  Mixtures also expose stable component responsibilities in log or probability
  form.
- **Special functions** (`function`) — gamma/beta/erf/Bessel families,
  multivariate gamma/beta, combinatorics, and stable log-domain helpers
  (`ln_sum_exp`, `ln_mean_exp`, `softmax`, `log_softmax`, `ln_1p_exp`,
  `ln_1m_exp`, `xlogy`, `erfcx`).
- **Sample statistics** (`statistics`) — summary/order/robust/weighted statistics,
  raw/log-domain weight normalization, ESS diagnostics, rank correlation & Cramér's V,
  effect sizes, multivariate sample covariance/correlation with matrix
  admissibility checks, Mahalanobis/Hotelling statistics, OLS regression
  diagnostics, resampling, hypothesis tests
  (z/t/F/χ²/KS/ANOVA/Wilcoxon/Fisher/binomial/McNemar), grouped variance
  decomposition, confidence intervals, KDE, ECDF/histogram, and mergeable
  streaming accumulators (`RunningStats`, `RunningCovariance`, `TDigest`).
- **Stochastic processes** (`process`) — Gaussian processes (with closed-form
  posterior), finite Markov chains with direct stationary distributions,
  communicating/recurrent/transient classes, hitting/absorption quantities, and
  finite-step distributions, fundamental matrices/visit counts, recurrence
  times, reversibility, and time reversal, plus Poisson processes and random
  walks.
- **Closed-form Bayesian layer** — exponential-family fitting via mergeable
  sufficient statistics, plus conjugate posteriors, predictives, and exact log
  marginal likelihoods, including Normal-Inverse-Gamma / Normal-Gamma scalar
  Gaussian conjugacy and Normal-Inverse-Wishart / Normal-Wishart multivariate
  Gaussian conjugacy. Scalar Gaussian conjugate models expose finite-batch
  posterior predictives for future observations, sums, and means. Gamma-Poisson
  and Gamma-Exponential conjugacy also support closed-form exposure/right-censoring
  sufficient statistics.

```rust
use bayes::{Normal, Continuous, ContinuousCdf, Moments, Distribution};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

let mut rng = ChaCha8Rng::seed_from_u64(42);
let n = Normal::new(0.0, 1.0).unwrap();      // mean, std-dev (validated)
let x: f64 = n.sample(&mut rng);             // sample
let d = n.pdf(x);                            // density
let p = n.cdf(1.96);                         // P(X ≤ 1.96) ≈ 0.975
let (mean, var) = (n.mean(), n.variance());  // Option<f64>
```

Functional composition:

```rust
use bayes::{ContinuousCdfExt, ContinuousDistributionExt, ContinuousFn, Continuous};

let unit = ContinuousFn::new(
    0.0,
    1.0,
    |x| if (0.0..=1.0).contains(&x) { 1.0 } else { 0.0 },
    |x: f64| x.clamp(0.0, 1.0),
    |p: f64| p.clamp(0.0, 1.0),
).unwrap();

let log_likelihood = unit.ln_likelihood(&[0.2, 0.4, 0.8]);
let mass = unit.probability_between(0.25, 0.75);
let shifted = unit.affine(2.0, -1.0).unwrap();
let density_at_zero = shifted.pdf(0.0);
```

Closed-form and finite-sample statistics:

```rust
use bayes::{cohen_d, proportion_wilson, weighted_mean};
use bayes::statistics::hypothesis::binomial_test;

let effect = cohen_d(&[3.0, 4.0, 5.0], &[1.0, 2.0, 3.0]);
let ci = proportion_wilson(8, 10, 0.95);
let weighted = weighted_mean(&[1.0, 3.0], &[1.0, 3.0]);
let test = binomial_test(9, 10, 0.5);
```

Closed-form Bayesian updating (no MCMC):

```rust
use bayes::{Beta, BetaBernoulli, Conjugate, Moments};

let model = BetaBernoulli::new(Beta::new(1.0, 1.0).unwrap());   // uniform prior
let posterior = model.posterior(&[true, true, false, true]);   // Beta(4, 2)
let p_hat = posterior.mean().unwrap();                         // 2/3
```

## Build

Standard cargo; runtime dependencies are `rand`, `rand_chacha`, and `thiserror`.

```
cargo build
cargo test
cargo fmt --check
cargo clippy --all-targets
cargo doc --no-deps
```

## Scope

The governing rule is **direct/finite-sample vs. iterative inference**: `bayes`
owns distributions, special functions, descriptive/frequentist statistics,
generative processes, closed-form Bayesian updates, exact tests, direct resampling
utilities, and bounded streaming summaries. Iterative inference (MCMC, variational
inference, optimisation-based fitting) is a **higher layer** that depends on
`bayes`. It holds no graph/routing/domain types. See `ARCHITECTURE.md` for the
current module boundaries and scope rules.
