# bayes-distribution-ext

**The distribution families you actually call.**

## The problem

The carrier-generic families in `bayes-distribution` cannot validate their
parameters, check support membership, sample, or fit — not because those are
unimportant but because none of them can be expressed without a host `bool`.
Somebody still has to do them.

This crate is that somebody. It wraps each lazy family with everything the
restrictions excluded, so the eager `Normal` is the carrier-generic density with
validation, support handling and host branching around it — not a second
implementation of the same formula.

## What it holds

32 families (23 continuous, 9 discrete) with:

- **validating constructors** — `Normal::new(0.0, -1.0)` is an `Err`, not a
  distribution that quietly answers nonsense
- **sampling** through `rand::distr::Distribution`
- moments, entropy, quantiles, survival functions
- **combinators** — truncation, mixtures, order statistics
- bijective **transformed** distributions
- **exponential-family** fitting via sufficient statistics
- 6 **conjugate** models with closed-form posteriors and predictives
- characteristic, moment-generating and probability-generating functions;
  L-moments; Fisher information

## Where the test suite lives

All four of the workspace's distribution suites are here, because this is the
first crate that can see everything they compare:

| suite | asks |
|-------|------|
| `tests/conformance/` | is one family internally consistent, under hostile input? |
| `tests/relationships/` | do families agree with the families they relate to? |
| `tests/kernels/` | do the carrier-generic kernels agree with the eager families? |
| `tests/sampling/` | do the samplers agree with their own densities? |

The sampling suite exists because it is the only thing an identity cannot check:
the density, CDF, quantile and moments all constrain each other, but a sampler is
a separate algorithm, and one drawing from the wrong law passes every
closed-form test.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
