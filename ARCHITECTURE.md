# bayes architecture

`bayes` is the closed-form probability and statistics foundation for the
urban-fabric stack. It defines random objects, how to evaluate them, how to sample
them, and the statistics that can be computed directly from data.

The governing boundary is:

- In scope: direct formulas, finite-sample algorithms, exact/direct sampling,
  closed-form posterior updates, sufficient statistics, descriptive or
  frequentist statistics, direct resampling utilities, and bounded streaming
  summaries.
- Out of scope: MCMC, variational inference, iterative optimization-based fitting,
  probabilistic-programming DSLs, graph/routing/domain types, and iterative or
  approximate inference engines.

If a capability is "run a loop until convergence", it belongs in a higher layer
that depends on `bayes`.

## Module Map

```text
src/
  lib.rs
  error.rs
  linalg.rs
  function/
  distribution/
  statistics/
  process/
```

`lib.rs` exposes the crate-level ergonomic surface. Distribution types and common
traits are re-exported at the root, as are the main statistics and process types.
Most statistics helpers are also re-exported flat from `bayes::statistics`; the
`statistics::hypothesis` subgroup remains public because tests are naturally
organized by family. Function helpers remain grouped under `bayes::function`.

`error.rs` contains one shared `BayesError`. Constructors return
`Result<Self, BayesError>` when invalid parameters are ordinary user input.
Statistics over malformed samples generally return `NaN` or `None`, matching the
existing "do not panic for degenerate data" convention.

`linalg.rs` is a small internal dense linear algebra layer: Cholesky, triangular
solves, and Gaussian elimination. It is deliberately private so the public API is
not committed to a hand-rolled matrix representation.

## Distribution Layer

`distribution/` is the trait spine plus concrete distributions.

Core scalar traits:

- `Continuous`: `pdf` / stable `ln_pdf`.
- `Discrete`: `pmf` / stable `ln_pmf`.
- `ContinuousCdf` and `DiscreteCdf`: CDF, survival, inverse CDF, log CDF, log
  survival, and inverse survival.
- `Support`: support bounds.
- `Constraint` / `Constrained`: support-domain metadata and smooth maps between
  constrained and unconstrained scalar coordinates for inference layers.
- `Moments`: optional analytic moments and summary points.
- `Estimate`: batch closed-form parameter fitting when available.

The concrete distributions are grouped by kind:

- `continuous/`: Normal, Gamma, Beta, Student-t, Cauchy, Exponential, Weibull,
  Pareto, Logistic, VonMises, and related scalar continuous families.
- `discrete/`: Bernoulli, Binomial, Poisson, Geometric, Categorical,
  Hypergeometric, NegativeBinomial, BetaBinomial, and related count families.
- `multivariate/`: MultivariateNormal, Dirichlet, Multinomial, with a small
  multivariate trait spine.
- `constraint.rs`: Scalar support constraints and unconstraining transforms.
- `combinator/`: Mixture and Truncated.
- `transformed/`: Bijectors and `Transformed<D, B>`.
- `functional/`: closure-backed density/mass/CDF adapters plus extension traits
  for log-likelihoods, interval probabilities, truncation, and bijective
  composition.
- `exponential_family/`: mergeable sufficient statistics and closed-form fits.
- `conjugate/`: closed-form conjugate prior/posterior updates and predictives.
- `divergence.rs`: closed-form KL divergence between same-family distributions
  (Normal, Gamma, Beta, Exponential, Poisson, Bernoulli).

Every distribution should validate constructor parameters, implement the relevant
spine traits, and implement `rand::distr::Distribution` for sampling.

The functional surface follows the lessons from `statrs`, Haskell `statistics`,
OCaml `pareto`, `kixi.stats`, and `statistex`: keep probability evaluation as
small traits (`pdf`/`pmf`, CDF/quantile, support), make log-probability a first
class operation, expose sample/statistical functions as pure total functions, and
represent optional analytic quantities with `Option` instead of sentinel values.
Named distributions and closure-backed distributions must compose through the
same traits.

## Function Layer

`function/` owns numerical special functions needed by the distribution layer:

- gamma and beta functions, including regularized incomplete forms;
- error functions and inverse error functions;
- factorial/binomial helpers;
- Bessel helpers for circular distributions;
- multivariate gamma and beta functions;
- stable log-domain helpers such as log-sum-exp and `xlogy`.

These functions are dependency-light and `f64`-first. More general numeric
abstractions should wait for a concrete consumer.

## Statistics Layer

`statistics/` is organized by statistical task. Most module files are private
implementation groups with their public functions re-exported from
`bayes::statistics`; `hypothesis` is public as a grouped namespace.

- `summary`: mean, variance, covariance, correlation, shape statistics, robust
  summaries, z-scores, and Gini mean difference;
- `order`: quantiles, percentiles, five-number summary, MAD, Tukey fences;
- `correlation`: Spearman, Kendall, Cramer's V;
- `effect`: standardized mean effects, ANOVA effect sizes, and 2x2 risk/odds
  effects;
- `weighted`: weighted means, variance, covariance, correlation, quantiles, and
  effective sample size;
- `multivariate`: mean/covariance/correlation matrices, pooled covariance,
  Mahalanobis distance, and Hotelling's T-squared;
- `regression`: simple OLS diagnostics and explicit-design least squares;
- `resampling`: bootstrap, jackknife, percentile intervals;
- `hypothesis`: z/t/Welch/F, Pearson correlation, chi-squared, Fisher exact, exact
  binomial, McNemar, KS, Jarque-Bera, ANOVA, Mann-Whitney, Wilcoxon, sign test;
- `confidence`: closed-form confidence intervals for means, proportions, rates,
  variance/standard deviation, and mean/proportion differences;
- `empirical`: ECDF and histograms;
- `kde`: Gaussian kernel density estimates;
- `online`: mergeable running statistics and covariance;
- `tdigest`: mergeable streaming quantile sketch.

The statistics API favors total functions: malformed or degenerate data returns
`NaN`, `None`, an empty vector, or a zero coefficient vector according to the local
module convention. It should not panic for user-provided sample shape issues.

## Process Layer

`process/` contains generative stochastic processes:

- Gaussian processes and posterior predictive conditioning;
- covariance kernels (`Rbf`, `Matern`, `Linear`, `Periodic`);
- Poisson processes;
- Gaussian random walks and Wiener paths;
- finite-state Markov chains.

These are direct/generative APIs. Markov chains here are random processes, not
MCMC inference machinery.

## Current Invariants

- The crate is domain-free: no graph, routing, geospatial, or application-specific
  types.
- Runtime dependencies are intentionally small: `rand`, `rand_chacha`, and
  `thiserror`.
- Public constructors reject invalid distribution parameters with `BayesError`.
- Matrix-like public inputs must validate rectangular shape before indexing.
- Kernel and Gaussian-process inputs must not silently ignore mismatched point
  dimensions.
- `cargo fmt --check`, `cargo test`, `cargo clippy --all-targets`,
  `cargo doc --no-deps`, and `cargo package --list --allow-dirty` should pass
  before merge.

## Backlog

Near-term improvements that fit the current structure:

- Consider a small public matrix/input validation helper if matrix-shaped APIs
  keep expanding.
- Decide whether kernel constructors should gain fallible `try_new` variants.
- Expand root-level examples for process and conjugate-update surfaces.
- Add package/readme examples as doctests where practical.
- Revisit optional features only when a consumer needs them, especially larger
  linear algebra backends or scalar-generic numerics.
