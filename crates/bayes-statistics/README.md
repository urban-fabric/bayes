# bayes-statistics

**Statistics over data, computed the way that survives the data you actually have.**

## The problem

The textbook formula is usually the wrong way to evaluate the formula. A variance
written as `Σx²/n − x̄²` is algebraically correct and numerically useless: on
`[10⁹+1, 10⁹+2, 10⁹+3]` the two terms agree to eighteen digits and their
difference is rounding noise, where the true answer is exactly 1. A standardized
moment that cubes before dividing overflows at data scale `10¹⁵⁰` and underflows
at `10⁻¹⁵⁰`, reporting `NaN` where the answer is an ordinary small number.

Both of those are the single most common defect in hand-rolled statistics code,
and both are fixed here — the first by accumulating about a running mean, the
second by standardizing before taking the power.

## What it holds

- **summary and order** — means, variance, skewness, kurtosis, covariance,
  correlation, robust spreads, quantiles, median, IQR, MAD, Tukey fences
- **correlation** — Spearman ρ, Kendall τ-b, Cramér's V
- **effect sizes** — Cohen's *d*, Hedges' *g*, Glass's Δ, η², ω², odds and risk
- **weighted** — normalization in the linear *and* log domains, ESS diagnostics,
  weighted moments and quantiles
- **regression** — simple OLS with full diagnostics
- **resampling** — bootstrap and jackknife with percentile intervals
- **19 hypothesis tests** — z, t, Welch, paired, F, χ², Mann–Whitney, Wilcoxon,
  sign, KS, Jarque–Bera, ANOVA, Fisher's exact, binomial, McNemar
- **confidence intervals** — closed-form for means, proportions, rates,
  variances; plus a distribution-free concentration tier (Chebyshev, Hoeffding,
  McDiarmid, Chernoff, Bernstein, Paley–Zygmund, Popoviciu)
- **estimation diagnostics** — Cramér–Rao and Bhattacharyya efficiency, observed
  Fisher information, Rao–Blackwellization, a Lehmann–Scheffé UMVU certifier
- **streaming** — `RunningStats` (Welford, mergeable), `RunningCovariance`,
  `TDigest`
- **density** — ECDF, histogram, Gaussian KDE

## One contract to rely on

Degenerate input yields `NaN`, **never a panic**. An empty slice, a one-element
slice, mismatched lengths, a constant sample, `±inf`, subnormals — every
reduction and every test returns a number or `NaN` and does not abort. A p-value
is always either a probability or undefined; a variance is never negative.

## Examples

```sh
cargo run -p bayes-statistics --example ab_test          # frequentist and Bayesian, side by side
cargo run -p bayes-statistics --example sample_size      # power analysis, checked by simulation
cargo run -p bayes-statistics --example reliability      # Weibull lifetimes and hazard rates
cargo run -p bayes-statistics --example quality_control  # control limits and capability
```

Each asserts every number it prints. `ab_test` is the one to read first.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
