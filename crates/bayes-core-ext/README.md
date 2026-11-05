# bayes-core-ext

**The eager additions to the trait spine.**

## The problem

Some statistical operations cannot be carrier-generic, and pretending otherwise
costs more than it buys. Fitting parameters from a sample needs the sample to
*exist* — you cannot estimate a mean from data that has not been computed yet.
So `Estimate` cannot live in `bayes-core` alongside the contracts that a graph
node can satisfy.

This crate is where that line is drawn. It layers the host-only surface on top of
`bayes-core` and re-exports the whole spine, so downstream code gets both the
contract and the eager extensions from one import.

## What it holds

`Estimate` — fitting a family's parameters from `&[f64]` — and the host-side
probability utilities, over the re-exported `bayes-core` spine.

## Why it is a separate crate rather than a feature flag

The split is the one axis the whole workspace is organized along: *does this code
need a host `bool`?* A crate boundary makes that answer checkable by the compiler
rather than by review. A deferred carrier can depend on `bayes-core` and be
statically unable to reach anything that would not lower.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
