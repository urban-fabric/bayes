# bayes-core

**The trait spine every distribution implements, carrier-generic.**

## The problem

Two backends reading the same method signature differently is the one failure
mode that defeats the point of a shared contract: the same formula would produce
different answers on different interfaces. So the contract has to be small,
explicit, and documented down to its conventions — what `inverse_cdf(0)` returns,
whether a support bound is inclusive, what an undefined moment reports.

## What it holds

The decomposed spine — `Continuous`, `Discrete`, `IntegerDiscrete`,
`ContinuousCdf`, `DiscreteCdf`, `IntegerDiscreteCdf`, `Support`, `Moments` — plus
the kernel companions that drop the support check, the constraint vocabulary
(`Boundary`, `Constrained`, `RealConstraint`, …), the error types, and the
probability-vector and scoring-rule surface (Brier, log, spherical, ranked
probability scores; row-stochastic validation).

Decomposed rather than one fat trait on purpose: a family with no closed-form CDF
should be able to implement the density without lying about the rest. `VonMises`
implements `Continuous` and `Moments` and no `ContinuousCdf`, because the
circular CDF has no closed form.

## Two conventions worth knowing

**Undefined moments are `NaN`, not `Option`.** `Cauchy::mean()` returns `NaN`
rather than `None` so the formula stays value-level: a deferred carrier can
express parameter-dependent existence with a `select`, which it could not do
through a host `Option`.

**Constraints return masks, not `bool`.** Membership predicates are value-level
over euclid's comparison surface for the same reason.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
