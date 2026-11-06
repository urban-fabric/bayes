# bayes-utils

**Shared test probes for the bayes workspace. Dev-only.**

`publish = false`. Nothing here reaches a consumer; this crate exists so the test
suites can share machinery instead of each growing its own copy.

## The problem

Two of them.

**Duplication.** Before this crate the suites had accumulated five copies of the
same failure accumulator and three of the same Simpson rule, which drifted — one
copy skipped non-finite samples, another propagated them.

**Uneven coverage.** When each suite keeps its own list of families to sweep, a
new family gets added to the list the author was looking at and silently missed
by the others. That is how a sweep ends up covering `Normal` thoroughly and
`Lomax` not at all.

## What it holds

**`probe`** — the failure accumulator that collects every failure in a sweep
rather than aborting on the first (the difference between "one family is wrong"
and "thirteen families are wrong in the same way" is the difference between a
typo and a defect in a shared kernel); quadrature, including the
endpoint-clustering substitution that singular quantile functions need; and
Richardson-extrapolated numerical derivatives.

**`families`** — one object-safe view over every family, and **the** canonical
parameter sweep. Adding a family to `continuous_cases` adds it to the
conformance, sampling and relationship suites at once. The parameters are chosen
to break things: scales from `1e-9` to `1e10`, shapes from `1e-3` to `1e4`,
degrees of freedom where moments cease to exist, and a location so far from the
origin that the family's own quantiles stop being representable.

**`gof`** — goodness-of-fit probes for judging a *sampler*, the one part of a
distribution no identity can pin down. Every threshold here is **derived** rather
than tuned, because a bound tightened until the suite passes is a bound that
asserts the sampler's current bias. The Kolmogorov p-value is a self-contained
series sharing no code with what it judges; a sample-mean bound is the
central-limit standard error; a variance bound uses the fourth moment and
declines to exist when the kurtosis does not.

## The dependency cycle is deliberate

This crate depends on `bayes-distribution-ext` while that crate dev-depends on
this one. The cycle runs only through dev edges, which cargo permits, and it is
what lets one canonical sweep serve every suite.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
