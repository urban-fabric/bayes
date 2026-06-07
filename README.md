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
  Categorical, BetaBinomial), multivariate (MVN, Dirichlet, Multinomial), plus combinators
  (Mixture, Truncated, Transformed).
  Each provides PDF/PMF, CDF/quantile/survival (incl. stable `ln_cdf`/`ln_sf`/`isf`),
  analytic moments (`Option`-typed — Cauchy honestly has no mean), and `rand` sampling.
  Same-family pairs also expose a closed-form `kl_divergence`.
  Functional adapters (`ContinuousFn`, `DiscreteFn`, `ContinuousDensity`,
  `DiscreteMass`) lift closures into the same trait spine for custom likelihoods.
- **Special functions** (`function`) — gamma/beta/erf/Bessel families,
  multivariate gamma/beta, combinatorics, and stable log-domain helpers
  (`ln_sum_exp`, `ln_1p_exp`, `ln_1m_exp`, `xlogy`, `erfcx`).
- **Sample statistics** (`statistics`) — summary/order/robust/weighted statistics,
  rank correlation & Cramér's V, effect sizes, multivariate sample covariance and
  Mahalanobis/Hotelling statistics, OLS regression diagnostics, resampling,
  hypothesis tests (z/t/F/χ²/KS/ANOVA/Wilcoxon/Fisher/binomial/McNemar),
  confidence intervals, KDE, ECDF/histogram, and mergeable streaming accumulators
  (`RunningStats`, `RunningCovariance`, `TDigest`).
- **Stochastic processes** (`process`) — Gaussian processes (with closed-form
  posterior), Markov chains, Poisson processes, random walks.
- **Closed-form Bayesian layer** — exponential-family fitting via mergeable
  sufficient statistics, plus conjugate posteriors and their predictives.

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
