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
- `multivariate/`: MultivariateNormal, MultivariateStudentT, Dirichlet,
  Multinomial, Wishart, InverseWishart, normalized spherical-uniform,
  von Mises-Fisher, and Watson-on-S² laws, Dirichlet-multinomial count
  distributions, Kent/Bingham unnormalized potentials, with a small
  multivariate trait spine.
  MultivariateNormal owns direct Gaussian algebra: coordinate marginals,
  conditioning, affine transforms, batch likelihoods, Gaussian product integrals,
  and precision fusion. MultivariateStudentT owns the matching elliptical
  algebra that remains closed form: coordinate marginals, affine images,
  conditional Student-t laws, Mahalanobis geometry, entropy, and batch
  likelihoods. Wishart families own matrix densities, entropy/KL, expected
  log-determinants, entry variances/covariances where their moments exist,
  congruence transforms, and inverse-family transforms. Dirichlet owns the
  simplex algebra that stays
  closed under projection: component and subset-sum Beta marginals, category
  lumping into a smaller Dirichlet, conditional sub-simplex laws, count updates,
  expected categorical/count log likelihoods, cross-entropy, and full
  Dirichlet-multinomial predictive count distributions with predictive log
  probabilities. Directional models keep the same closed-form boundary:
  spherical-uniform owns the sphere base
  measure, von Mises-Fisher owns normalized entropy, KL, moments, and product
  fusion, and WatsonS2 owns the antipodally symmetric S² law with closed-form
  normalizer, moments, entropy/KL, gradients, sampling, Bingham-potential
  embedding, and coaxial product fusion. Kent/Bingham own potential algebra,
  gradients, spectral/canonical forms, identity-shift invariance, rank-one
  Watson conversion, and normalized special cases (Kent beta=0, isotropic
  Bingham, two-dimensional Bingham, and rank-one S² Bingham via Watson).
  Generic Kent/Bingham normalizers remain outside the current closed-form surface.
- `constraint.rs`: Scalar support constraints and unconstraining transforms.
  Constraint transforms expose both forward and inverse log-Jacobian corrections,
  using stable log-domain sigmoid identities in the logistic tails so downstream
  inference layers do not rederive transform math.
- `probability.rs`: Probability-vector and row-stochastic kernel/matrix
  validation, probability weight normalization, and closed-form proper scoring
  rules for categorical forecasts.
- `combinator/`: Mixture, OrderStatistic, and Truncated. Mixtures expose stable
  component responsibilities via the same log-softmax/log-sum-exp substrate used
  by log-densities. Order statistics lift any continuous PDF/CDF/quantile into
  the finite-sample minimum, maximum, or kth sample statistic through the direct
  beta-CDF identity.
- `transformed/`: Bijectors and `Transformed<D, B>`.
- `functional/`: closure-backed density/mass/CDF adapters plus extension traits
  for log-likelihoods, interval probabilities, reliability/survival quantities,
  order statistics, truncation, and bijective composition.
- `exponential_family/`: mergeable sufficient statistics and closed-form fits.
- `conjugate/`: closed-form conjugate prior/posterior updates and predictives,
  including Normal-Inverse-Gamma and Normal-Gamma for scalar Gaussian
  mean/variance or mean/precision conjugacy, plus Normal-Inverse-Wishart and
  Normal-Wishart for multivariate Gaussian mean/covariance or mean/precision
  conjugacy. Scalar Gaussian conjugate wrappers also expose finite-batch
  posterior predictives for future observations, sums, and means. Multivariate
  Gaussian conjugate wrappers expose predictive log densities, future-batch
  predictive likelihoods, and sequential predictive factorizations of the
  marginal likelihood. Beta, Dirichlet, Gamma, and multivariate Gaussian
  conjugate wrappers expose the same predictive likelihood and
  sequential-evidence surface. Gamma-Poisson and Gamma-Exponential carry that
  surface through exposure/right-censoring updates where the sufficient
  statistic is `(events, exposure)` rather than raw observations.
  Every family is one instance of the `Conjugate` trait's generic
  `posterior = η + Σ T(xᵢ)` accumulation in a `NaturalParams` monoid — supplying
  only its statistic `T`, prior coordinates `η`, and the map back to a
  distribution; the streaming `posterior_natural` accumulator and the
  online-equals-batch / order-independence guarantees are proved generically in
  `riemann` (`Bayes/ExponentialFamily`, `Bayes/ConjugateFamilies`).
  Family wrappers also expose exact log marginal likelihoods where the likelihood
  parameters are integrated out analytically.
- `divergence/`: the closed-form divergence toolkit. `mod.rs` holds the
  per-family closed forms — KL (17 families), Jeffreys/reverse-KL, Jensen–Shannon,
  squared Hellinger, total variation, Pearson χ², Rényi, and mutual information.
  `fdiv.rs` is the **f-divergence umbrella**: one generic `f_divergence(p, q, gen)`
  over probability vectors with pluggable convex generators (`Kl`, `ReverseKl`,
  `ChiSquared`, `SquaredHellinger`, `TotalVariation`, `Jeffreys`, `Alpha`,
  `Custom`), plus the row-stochastic `Channel` (Markov kernel) carrying the
  data-processing inequality. The three universal guarantees (nonnegativity,
  data-processing monotonicity, joint convexity) are proved generically in
  `riemann` (`Bayes/Divergence`: `fDiv_nonneg`, `fDiv_dataProcessing`,
  `fDiv_convex`).

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
- stable log-domain helpers such as log-sum-exp, log-mean-exp, softmax/log-softmax,
  and `xlogy`.

These functions are dependency-light and `f64`-first. More general numeric
abstractions should wait for a concrete consumer.

## Lazy Numeric Layer

`lazy` owns closed-form probability kernels over `euclid::numeric::LazyReal`
and `LazyComplex`. These are unchecked/on-support formulas for downstream lazy
expression systems: no support branching, parameter validation, sampling, CDFs,
quantiles, or eager special-function work is hidden inside the entry points.
Scalar constants are lifted through Euclid's lazy scalar substrate; callers
provide precomputed special-function normalizers. The existing distribution
spine remains `f64`-first and source-compatible.

The kernel surface covers the scalar continuous and discrete log-densities
(Normal through Gumbel, Bernoulli through hypergeometric), the simplex/diagonal
vector forms (Dirichlet, diagonal Normal, multinomial, Dirichlet-multinomial),
and — taking the caller's linear-algebra/geometry term as input — the
multivariate Gaussian and Student-t log-densities from a precomputed Mahalanobis
quadratic form, and the von Mises-Fisher and Watson directional log-densities
from a precomputed sphere cosine. It also includes the circular von Mises
density (over the lazy `cos`); the closed-form **CDFs** — Normal/half-normal (erf),
Gamma/chi-squared/Poisson (regularized incomplete gamma), Beta/Student-t/F/
binomial/negative-binomial (regularized incomplete beta), and the pure-on-support
exponential/logistic/Weibull/Gumbel/Frechet/Rayleigh/Pareto/Lomax; and the
differential/Shannon **entropies** for the same families (pure algebra for most,
the lazy `ln_gamma`/`digamma` for the gamma/beta-shaped ones); the same-family
**divergences** — KL, squared Hellinger, Pearson χ², Rényi, and total variation —
over the families whose closed form needs only the lazy surface (now including the
circular **von Mises** divergences, written with `cos` and the scaled Bessel
`I₀ᵉ`/`I₁ᵉ`, and the **Wald**/**binomial**/**negative-binomial** divergences across
KL/Hellinger/χ²/Rényi — the binomial as `n ×` / `(·)^n` of its Bernoulli kernel);
and the stable log-CDF tails via `ln_1m_exp`/`ln_1p_exp`. The
Student-t CDF is written branch-free with `sign`, and the `+∞`-outside-containment
divergences carry an `_on_support` interior form, so every entry point stays
traceable.

The vector layer is the simplex/log-domain reductions (tensor `Reduce` shaped):
the log-sum-exp family (`log_sum_exp`/`log_mean_exp`/`softmax`/`log_softmax`),
the probability-vector information theory (Shannon/cross entropy, Bhattacharyya,
Jensen–Shannon), the categorical divergences (KL, χ², Rényi, total variation,
Hellinger), and Dirichlet KL/entropy over the lazy `ln_gamma`/`digamma`. These take
equal-length, on-support vectors and fold branch-free.

The reparameterization layer carries the constrained↔unconstrained bijections an
inference layer differentiates through — positive (`exp`/`ln`), unit-interval
(`sigmoid`/`logit`), and bounded-interval maps with their log-Jacobian corrections
— plus the categorical proper scoring rules (Brier, spherical). The boundary `±∞`
is the caller's branch, so each transform traces unchanged to a backend.

Closed-form Bayesian evidence rounds it out: the conjugate posterior-predictive
log-densities (Beta-Bernoulli, Gamma-Poisson, Gamma-Exponential) and log marginal
likelihoods (Beta-Bernoulli over sufficient statistics, Normal known-variance
Gaussian integral), all over the lazy `ln_gamma` with no caller-supplied
normalizer. Only the matrix-conjugate evidences (NIW/NW) wait on the matrix
contract.

The analytic higher moments close the loop on the scalar surface: the
parameter-dependent skewness, excess kurtosis, and non-trivial mean/variance
(log-normal, gamma, beta, chi-squared, Wald, Poisson, binomial, geometric,
negative-binomial) — pure algebra in the shape parameters.

Euclid's `LazyReal` carries the whole elementwise analysis surface —
transcendentals, trig, `min`/`max`/`clamp` selectors, and the special functions
(`erf`/`erfc`/`erf_inv`, `ln_gamma`/`digamma`/`trigamma`, regularized incomplete
gamma/beta) whose concrete `f64` reference lives in `euclid::special` (re-exported
through `bayes::function` so the distribution spine and the lazy kernels share one
implementation). What still must be caller-supplied is therefore only the
genuinely non-scalar work: a matrix solve/log-determinant, or a general-order
Bessel normalizer. Each entry point stays pure analysis over the lazy traits, so
it lowers unchanged to a tensor backend.

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
- `weighted`: raw/log-domain weight normalization, weighted means, variance,
  covariance, correlation, quantiles, and effective sample size/admissibility. The
  log-weight helpers are the closed-form numeric part of the riemann
  `Markov/LogSumExp` and `Markov/ImportanceSampling` proofs: softmax/normalized
  weights form a probability vector, and normalized nonnegative weights have
  `1 <= ESS <= n`;
- `multivariate`: mean/covariance/correlation matrices, covariance/correlation
  admissibility predicates, pooled covariance, Mahalanobis distance, and
  Hotelling's T-squared;
- `regression`: simple OLS diagnostics and explicit-design least squares;
- `resampling`: bootstrap, jackknife, percentile intervals;
- `estimation`: Fisher-information diagnostics, Cramér–Rao/Bhattacharyya variance
  floors, Rao–Blackwellization, grouped variance decomposition, and UMVU
  certification. The grouped decomposition is the direct finite-sample form of
  riemann `Markov/Diagnostics`: within-group plus between-group sums of squares
  recover total spread about the grand mean;
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
- finite-state Markov chains, including direct stationary distributions,
  communicating/recurrent/transient classes, hitting and absorption quantities,
  first-return probabilities, exact finite-step transition/distribution
  evolution, fundamental matrices/visit counts, recurrence times, reversibility,
  and time reversal.

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

- Extend the existing validation helpers only when a second caller needs the same
  matrix-shaped invariant.
- Decide whether kernel constructors should gain fallible `try_new` variants.
- Expand root-level examples for process and conjugate-update surfaces.
- Add package/readme examples as doctests where practical.
- Revisit optional features only when a consumer needs them, especially larger
  linear algebra backends or scalar-generic numerics.
