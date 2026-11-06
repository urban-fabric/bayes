# bayes-distribution

**31 distribution families that run on any carrier.**

## The problem

A distribution type that can only hold `f64` cannot appear in a compiled model.
The moment you want a log-likelihood staged into a tensor runtime or an
expression graph, an `f64`-only `Normal` is useless and you write the density
again by hand.

So these families are generic in their parameters: `Normal<T>` for any
`T: Real`. The same struct holds host doubles, graph nodes, or tensor elements,
and `ln_pdf` is the same code in every case.

## What it holds

22 continuous and 9 discrete families as parameter structs with carrier-generic
density, mass and CDF implementations; closure-backed functional adapters; the
conjugate natural-parameter monoids; and euclid's explicit-key sampler contract,
where a draw is a pure function of an immutable key rather than of a mutable RNG
handle.

## The constructors cannot fail, deliberately

`Normal::new(mu, sigma)` returns `Self`, not `Result`. That is not an oversight:
on a deferred carrier there is nothing to validate *yet* — the parameters may be
graph nodes whose values are not known until the graph runs. Validation is the
eager wrapper's job, in `bayes-distribution-ext`.

The same reasoning removes support checks from the density. `ln_pdf` here is the
closed-form algebra for a point already known to be on the support; the eager
crate adds the mask.

## Which crate do you want?

If you are writing ordinary statistics on `&[f64]`, you want
`bayes-distribution-ext` — the validating, sampling, fitting surface. This crate
matters when your formula has to run somewhere `f64` does not reach.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
