# bayes-kernels

**The closed forms themselves, written once for every carrier.**

## The problem

A log-density gets written twice. Once eagerly, on host `f64`, for a statistical
summary; once again inside model code, staged into a tensor runtime or an
expression graph to be compiled and differentiated later. The two copies drift —
a `ln_gamma` normalizer gets dropped from one because it is constant in the
parameters, a variance is corrected in the other — and the model's reported
likelihood stops matching the library's density by an amount nobody notices.

This crate is the single artifact that removes the duplication. Every closed form
is written against `euclid_numeric::Real` rather than `f64`, so the same function
compiles against an eager scalar *and* against a deferred carrier.

## What it holds

Roughly 200 carrier-generic kernels: log-densities and log-masses for 30-odd
families, CDFs, quantiles, divergences, the GP covariance functions, GLM
link/score/curvature, reparameterization bijections, and the normal and
complex-normal log-probability primitives.

## What it refuses to do

A kernel does **not** validate parameters, check support membership, sample, or
branch. It takes already-checked parameters and returns algebra. That is not
minimalism for its own sake: a kernel that reaches for a host `bool` cannot run
on a graph node, which is the entire point of the crate. Support checks and
validation belong to the eager wrapper in `bayes-distribution-ext`.

Naming follows the restriction: `_on_support` and `_on_unit` kernels assume the
caller has already checked membership. Full-domain wrappers exist only where the
support behaviour can be expressed with euclid's mask surface.

## Verifying it

The kernels sit *below* the families in the dependency graph, so nothing here can
see a `Normal` to compare against. The cross-checks therefore live one level up,
in `bayes-distribution-ext/tests/kernels/`, where both are visible. If they ever
disagree, the claim that the eager family and the staged kernel are "the same
closed form reached through two doors" is false.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
