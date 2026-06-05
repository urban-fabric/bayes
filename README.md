# bayes

**The closed-form probability & statistics foundation for Rust.**

`bayes` is a standalone, domain-free numerics library: everything computable by a
direct formula or an exact sample. It is the home of the distribution code formerly
in `abel-stats` (the abel kernel now consumes `bayes` through a thin shim).

## What's in it

- **~36 distributions** on a decomposed trait spine — continuous (Normal,
  LogNormal, Exponential, Gamma, Beta, Cauchy, Laplace, Student-t, χ², Weibull,
  Pareto, F, Triangular, Rayleigh, Gumbel, Half-Normal, Inverse-Gamma, Logistic,
  Lomax, Frechet, Wald, VonMises, Uniform, Uniform2D), discrete (Bernoulli, Binomial,
  Poisson,
  Geometric, DiscreteUniform, NegativeBinomial, Hypergeometric, Categorical,
  BetaBinomial), multivariate (MVN, Dirichlet, Multinomial), plus combinators
  (Mixture, Truncated, Transformed).
  Each provides PDF/PMF, CDF/quantile/survival (incl. stable `ln_cdf`/`ln_sf`/`isf`),
  analytic moments (`Option`-typed — Cauchy honestly has no mean), and `rand` sampling.
- **Special functions** (`function`) — gamma/beta/erf/Bessel families, combinatorics,
  and stable log-domain helpers (`ln_sum_exp`, `ln_1p_exp`, `ln_1m_exp`, `xlogy`,
  `erfcx`).
- **Sample statistics** (`statistics`) — summary, order, rank correlation & Cramér's
  V, OLS regression, resampling, hypothesis tests (z/t/χ²/KS/ANOVA/Wilcoxon/Fisher),
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
```

## Scope

The governing rule is **closed-form / exact vs. iterative / approximate**: `bayes`
owns everything computable by a direct formula or an exact sample — distributions,
special functions, descriptive/frequentist statistics, generative processes, and
closed-form Bayesian updates. Iterative inference (MCMC, variational inference,
optimisation-based fitting) is a **higher layer** that depends on `bayes`. It holds
no graph/routing/domain types. See `ARCHITECTURE.md` for the current module
boundaries and scope rules.
