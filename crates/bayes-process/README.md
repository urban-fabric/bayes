# bayes-process

**Random processes, not just random numbers.**

## The problem

Drawing i.i.d. samples from a distribution is the easy case. Many questions are
about a whole *trajectory* — how many tickets arrive before lunch, where does a
price path end up, how correlated are two points on a surface — and those need
the joint structure, not a marginal.

## What it holds

- **`PoissonProcess`** — arrivals in continuous time, homogeneous and (via
  `inhomogeneous_arrivals`) with a time-varying rate
- **`GaussianRandomWalk`** and **`wiener_path`** — discrete walks and Brownian
  motion on a given time grid
- **covariance `Kernel`s** — RBF, Matérn at half-integer smoothness, periodic,
  linear — with the Gaussian-process marginal-likelihood objective

## Generative, not inferential

These define and sample processes. Inference *over* them — MCMC, variational
methods, GP posterior fitting — is a higher layer's job, and the matrix-coupled
processes (`GaussianProcess` regression, `MarkovChain` trajectories) live in the
sibling crate `ito`, built on these kernels. What stays here is everything that
needs no matrix algebra.

## The example is the documentation

```sh
cargo run -p bayes-process --example arrivals
```

A Poisson process is pinned down by one rate, and everything else follows: the
count in a window is Poisson, the gaps are exponential, and conditional on the
count the arrival times are uniform. The example simulates all three and checks
each against the closed form — which is also a cross-check between this crate and
`bayes-distribution-ext`.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
