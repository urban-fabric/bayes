# bayes architecture

`bayes` is the closed-form probability and statistics foundation for the
urban-fabric stack. It defines random objects, how to evaluate them, how to sample
them, and the statistics that can be computed directly from data.

The governing boundary is:

- In scope: direct formulas, exact sampling, closed-form posterior updates,
  sufficient statistics, and descriptive or frequentist statistics.
- Out of scope: MCMC, variational inference, iterative optimization-based fitting,
  probabilistic-programming DSLs, graph/routing/domain types, and approximate
  inference engines.

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
Specialized helpers remain available under their modules.

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
- `Moments`: optional analytic moments and summary points.
- `Estimate`: batch closed-form parameter fitting when available.

The concrete distributions are grouped by kind:

- `continuous/`: Normal, Gamma, Beta, Student-t, Cauchy, Exponential, Weibull,
  Pareto, Logistic, VonMises, and related scalar continuous families.
- `discrete/`: Bernoulli, Binomial, Poisson, Geometric, Categorical,
  Hypergeometric, NegativeBinomial, BetaBinomial, and related count families.
- `multivariate/`: MultivariateNormal, Dirichlet, Multinomial, with a small
  multivariate trait spine.
- `combinator/`: Mixture and Truncated.
- `transformed/`: Bijectors and `Transformed<D, B>`.
- `exponential_family/`: mergeable sufficient statistics and closed-form fits.
- `conjugate/`: closed-form conjugate prior/posterior updates and predictives.

Every distribution should validate constructor parameters, implement the relevant
spine traits, and implement `rand::distr::Distribution` for sampling.

## Function Layer

`function/` owns numerical special functions needed by the distribution layer:

- gamma and beta functions, including regularized incomplete forms;
- error functions and inverse error functions;
- factorial/binomial helpers;
- Bessel helpers for circular distributions;
- stable log-domain helpers such as log-sum-exp and `xlogy`.

These functions are dependency-light and `f64`-first. More general numeric
abstractions should wait for a concrete consumer.

## Statistics Layer

`statistics/` is organized by statistical task:

- `summary`: mean, variance, covariance, correlation, shape statistics;
- `order`: quantiles, percentiles, five-number summary, MAD;
- `correlation`: Spearman, Kendall, Cramer's V;
- `regression`: simple OLS and explicit-design least squares;
- `resampling`: bootstrap, jackknife, percentile intervals;
- `hypothesis`: z/t/Welch, chi-squared, Fisher exact, KS, Jarque-Bera, ANOVA,
  Mann-Whitney, Wilcoxon;
- `confidence`: closed-form confidence intervals;
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

These are still closed-form/generative. Markov chains here are random processes,
not MCMC inference machinery.

## Current Invariants

- The crate is domain-free: no graph, routing, geospatial, or application-specific
  types.
- Runtime dependencies are intentionally small: `rand`, `rand_chacha`, and
  `thiserror`.
- Public constructors reject invalid distribution parameters with `BayesError`.
- Matrix-like public inputs must validate rectangular shape before indexing.
- Kernel and Gaussian-process inputs must not silently ignore mismatched point
  dimensions.
- `cargo fmt --check`, `cargo test`, `cargo clippy --all-targets`, and
  `cargo doc --no-deps` should pass before merge.

## Backlog

Near-term improvements that fit the current structure:

- Add a small public matrix/input validation helper if more multivariate APIs land.
- Decide whether kernel constructors should gain fallible `try_new` variants.
- Expand root-level examples for the process and conjugate-update surfaces.
- Add package/readme examples as doctests where practical.
- Revisit optional features only when a consumer needs them, especially larger
  linear algebra backends or scalar-generic numerics.
