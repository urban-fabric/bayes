# Contributing to bayes

bayes is the probability and statistics layer for Urban Fabric. It is a **beta**
repository: the trait spine is still moving, and downstream projects are expected
to track it rather than pin to a stable contract.

It also sits downstream of [euclid](https://github.com/urban-fabric/euclid), which
is itself beta. A euclid release can change a numeric trait under you; see
[Tracking euclid](#tracking-euclid) below.

## Version Control: Jujutsu

**This repository uses [Jujutsu](https://github.com/jj-vcs/jj) (`jj`), not git
directly.** It is a *colocated* repo — `.jj` and `.git` sit side by side — so git
tooling can still read it, but `jj` is the interface you should work through.

One consequence to know up front: in a colocated repo git's `HEAD` is normally
**detached**, because jj owns the working copy. That is expected, not a problem to
fix. Running `git commit` here will create a commit git can see but that sits
outside jj's view of history until the next `jj` invocation imports it, so it is
worth avoiding.

The everyday loop:

```sh
jj status                   # what has changed (jj auto-tracks new files)
jj diff                     # review it
jj commit -m "message"      # commit the working copy and start a fresh change
jj log                      # history
```

Useful beyond that:

```sh
jj describe -m "message"    # set/replace the current change's message
jj new                      # start a new empty change on top
jj edit <rev>               # move the working copy to an existing change
jj squash                   # fold the working copy into its parent
jj undo                     # undo the last jj operation
jj op log                   # every operation, so nothing is truly lost
```

## Publishing: A Branch and a Squash Merge, Never a Direct Push

**Do not push to `main`.** Push the work to a new branch, open a pull request, and
squash-merge that into `main`.

The reason is the section below: there is no CI here, so the pull request is the
only review gate the project has. A direct push to `main` does not skip a
formality — it skips the only point at which a change is read by someone other
than its author. And because the merge is a *squash*, the PR title and body are
not commentary on the commit, they **become** it: whatever you write there is what
lands on `main` and is all anyone reading `jj log` later will have.

Bookmarks (jj's branches) **do not advance on their own**, which works in your
favour here — after `jj commit`, nothing has moved and nothing has left your
machine. Publishing is a separate, deliberate step:

```sh
jj git push --named my-feature=@-   # create the bookmark and push it in one go
```

`--named` creates the remote branch and starts tracking it. Push follow-up commits
on the same branch with `jj bookmark set my-feature -r @-` and then
`jj git push --bookmark my-feature`. If you do not care what the branch is called,
`jj git push -c @-` will name one after the change id.

Then open the PR and squash-merge it on GitHub.

**After the squash merge, reconcile deliberately.** A squash merge puts a *new,
different* commit on `main` — it does not fast-forward to yours, so your local
commits are now redundant copies of work that has already landed under another
hash. Do not merge or rebase them forward:

```sh
jj git fetch                 # pick up the squashed commit on main
jj new main@origin           # start the next change on top of it
jj abandon <your-local-commits>   # drop the now-redundant originals
```

Skipping that is how the same change ends up in the history twice.

Pushing and merging both reach outside your machine, so treat them as publish
actions rather than part of the commit loop — and run the checks below first.

## Checks Run Locally, Not in CI

**There is no CI, by choice.** We do not want to pay for external compute to run
a test suite that takes well under a second on a laptop. The consequence is that
**the checks are your responsibility before you push** — nothing downstream will
catch a regression for you.

Run all six, from the workspace root:

```sh
cargo fmt --all --check
cargo test --all-targets --workspace
cargo clippy --all-targets --workspace -- -D warnings
cargo test --doc --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items
for e in ab_test sample_size reliability quality_control; do
  cargo run -q -p bayes-statistics --example "$e" >/dev/null
done
cargo run -q -p bayes-process --example arrivals >/dev/null
```

All six must be clean.

**Step six** is there because `--all-targets` *compiles* examples but never runs
them. Every example asserts its own output, so running them is a real check — and
an example that compiles while printing a wrong number is worse than no example.
See [Examples](#examples) below.

**Step five** is not redundant with `cargo test --doc`: that runs doctests, while
this resolves **intra-doc links**. Nothing else catches a `[`foo`](path)` whose
path has gone stale, and a broken link is worse than a missing one — it silently
points somewhere plausible. bayes is a ten-crate workspace whose crates
re-export each other heavily, so stale links are the common failure here.

**A git `pre-push` hook will not help you here.** `jj` does not run git hooks, so
a `.git/hooks/pre-push` script is silently skipped on `jj git push` — exactly when
you would want it. Use a `jj` alias instead, in your own
`~/.config/jj/config.toml`:

```toml
[aliases]
check = ["util", "exec", "--", "sh", "-c", """
  set -e
  cargo fmt --all --check
  cargo test --all-targets --workspace
  cargo clippy --all-targets --workspace -- -D warnings
  cargo test --doc --workspace
  RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items
"""]
```

Then `jj check` before `jj git push`. That lives in your user config rather than
the repo, so it stays a personal habit rather than a project mandate — and it
costs nothing but your own CPU.

Lints are declared once in `[workspace.lints]` at the root and inherited by every
crate, so `clippy -- -D warnings` is the same gate everywhere. `unsafe_code`,
`unwrap_used`, `expect_used`, and `panic` are all denied; reference code in tests
is held to the same bar, which is why test modules open with an explicit
`#![cfg_attr(test, allow(...))]` rather than leaving the lint unsatisfied.

## No CHANGELOG

**There is no CHANGELOG, by choice.** While bayes is beta, the version number and
the `jj log` are the record. Maintaining a hand-written changelog for a surface
that is still being reshaped costs more than it informs, and a stale changelog is
worse than none.

Instead:

- Put the explanation in the **pull request body**, which the squash merge turns
  into the commit message on `main`. That is where a moved distribution family or
  a changed parameterization gets described. Write it as the permanent record it
  becomes, not as a note to the reviewer.
- Bump the workspace version in the root `Cargo.toml` when the public surface
  moves. All crates share `[workspace.package]`, so they version together.
- Say so in the PR title when imports or semantics change, since downstream
  projects read titles to decide whether they need to react — and since under a
  squash merge the title is the one line that survives into `jj log`.

This should be revisited if bayes ever reaches 1.0 and outside consumers depend
on it, but that is not the situation today.

## Where a Change Goes

bayes is split along one axis: **does this formula need a host `bool`?** That
question decides the crate, and getting it wrong is the most common structural
mistake here. The `-ext` crates are the eager half; the plain ones are the
carrier-generic half. See the README's [Why the crates split in
two](README.md#why-the-crates-split-in-two) for the reasoning.

| Adding | Goes in |
|--------|---------|
| a closed-form log-density, CDF, or divergence formula | `bayes-kernels` |
| a trait every distribution should implement | `bayes-core` (carrier-generic) or `bayes-core-ext` (eager-only) |
| a lazy parameter struct + density impl | `bayes-distribution` |
| an eager family: constructor validation, sampling, moments, fitting | `bayes-distribution-ext` |
| a divergence or information measure between families | `bayes-divergence` |
| a statistic computed over `&[f64]` data | `bayes-statistics` |
| a stochastic process or covariance kernel | `bayes-process` |
| a sampler for a density with no closed-form inverse | `bayes-sampling` |
| a shared test probe, or a family to sweep | `bayes-utils` (dev-only) |

Two rules that fall out of that split:

- **A kernel does not validate, branch, or sample.** `bayes-kernels` functions
  take already-checked parameters and return the algebra. Support checks and
  parameter validation belong to the eager wrapper that calls them. A kernel that
  reaches for a host `bool` has to be rewritten on euclid's mask surface or it
  does not belong in that crate.
- **A new eager family should be cross-checked against its kernel.** The point of
  having both is that they are written independently and must agree; that
  agreement is the test, and it lives in `bayes-distribution-ext/tests/`.

## What a Change Should Come With

- **New distribution family** — the lazy struct, the eager wrapper, and the
  kernel cross-check asserting the two agree. State the parameterization in the
  struct's doc comment: `Gamma(shape, rate)` and `Gamma(shape, scale)` are both
  defensible and silently differ by an inversion, so the doc comment is the
  contract.
- **New or changed trait method** — update every implementor. The spine traits
  are small on purpose and widely implemented, so a method added to `Moments` or
  `ContinuousCdf` is a change to every family.
- **New statistic** — say which convention it follows where one exists (`n` vs
  `n-1` denominators, inclusive vs exclusive quantile interpolation, which of the
  several "Kendall's tau" definitions). Two libraries disagreeing on a quantile
  definition is not a bug in either, which is exactly why it has to be written
  down.
- **Docs are part of the deliverable.** Document the support, the
  parameterization, the panics, and the precision limits. Skip the
  design-rationale essay unless it stops someone from "fixing" something that is
  deliberate.

## Tests

Unit tests live in a **sibling `tests.rs`** beside the `mod.rs` they cover, not
in a `#[cfg(test)] mod tests` block inside it, and not in a self-named
`tests/tests.rs`. `self_named_module_files` is denied at the workspace root, so
clippy enforces the shape.

Integration tests in a crate's `tests/` directory are the right home for anything
that **spans crates** — the kernel-vs-family cross-checks are the main example,
since the kernels sit below the families in the dependency graph and nothing below
can see both. Where that needs a crate the implementation does not use, add it
under `[dev-dependencies]`: `bayes-distribution-ext` dev-depends on
`bayes-divergence` for exactly this reason.

### How the suites are laid out

`bayes-distribution-ext/tests/` is four directory targets, each answering one
question. The split is by *question*, not by family, so a new family is swept by
all of them at once rather than needing a new file in each:

| target | asks |
|--------|------|
| `conformance/` | is one family internally consistent, under hostile input? |
| `relationships/` | do families agree with the other families they relate to? |
| `kernels/` | do the carrier-generic kernels agree with the eager families? |
| `sampling/` | do the samplers agree with their own densities? |

Each directory's `main.rs` is its index and explains what belongs in it. Read
that before adding a file.

**`bayes-utils` holds everything shared.** The failure accumulator, the
quadrature (including the endpoint-clustering substitution that singular
quantile functions need), the numerical derivatives, the goodness-of-fit probes,
and — most importantly — [`continuous_cases`] and [`discrete_cases`], the
canonical parameter sweep. Add a family there and every suite picks it up.
Before writing a helper in a test file, check whether it belongs in `bayes-utils`
instead; the suites had accumulated five copies of the same `Failures` struct
before it existed.

`bayes-utils` depends on `bayes-distribution-ext` while that crate dev-depends
on it. That cycle is only through dev edges, which cargo permits, and it
is what lets one sweep serve every suite.

### Sampling is the one thing identities cannot check

Every other suite rests on the density, CDF, quantile and moments constraining
each other. A **sampler** has no such protection: it is a separate algorithm, and
a sampler drawing from the wrong law passes every closed-form test. It can only
be checked statistically, which makes the tolerance the delicate part.

**Derive the threshold; never tune it.** A bound tightened until the suite passes
is a bound that asserts the sampler's current bias. `bayes_utils::gof` exists
so the derivations are in one place: the Kolmogorov p-value is a self-contained
series sharing no code with what it judges, a sample-mean bound is
`sigmas·√(Var/n)`, and a variance bound uses the fourth moment and declines to
exist when the kurtosis does not. Seeds are fixed so a failure reproduces.

### The relationship network

The most valuable tests here are not the ones that check a function against a
number — they are the ones that check a function against **another function**.
`crates/bayes-distribution-ext/tests/relationships/` is organized around that
idea, and its module header is the index. The principle:

> Two implementations that share no code cannot agree while both are wrong.

A `Poisson` CDF is a sum of log-factorials; a `ChiSquared` CDF is a regularized
incomplete gamma. They have nothing in common, yet `P(Poisson(λ) ≤ k)` and
`P(χ²_{2(k+1)} > 2λ)` are the same number. An identity across that much
implementation distance is the strongest self-check the library can produce, and
it is why the suite found real defects that per-function tests had not.

**A new function should gain at least one edge in that network.** A limit, a
reparameterization, a derivative, a discrete↔continuous bridge, or a test
against the distribution it looks its p-value up in. One edge there is worth
more than a dozen sampled reference values, because a reference value can be
transcribed wrongly and an identity cannot.

When an identity fails, work out **which side is wrong** before changing
anything. Several apparent defects found while building this suite were the
probe's fault, not the library's — a Simpson rule defeated by a `p^(-1/2)`
singularity, an MGF summed outside its radius of convergence, a tolerance that
ignored the distribution's own scale. Compute the exact answer independently
first.

### Identities over reference values

Prefer **identity tests** over hand-copied reference digits. A closed-form
relation validates itself and cannot be transcribed wrongly:

- `pdf` integrates to the `cdf`; `cdf(inverse_cdf(p)) == p`
- analytic `Moments` match the sample moments of a large seeded draw
- a conjugate posterior update matches Bayes' rule applied directly
- a divergence is zero between identical distributions and positive otherwise
- a limiting case collapses to another family (`StudentT(∞) == Normal`,
  `NegativeBinomial → Poisson`)

Where a reference value is genuinely needed, derive it from an exact closed form
or an independent implementation, and say in a comment where it came from.

Seed every sampler. `rand_chacha::ChaCha8Rng::seed_from_u64` is what the existing
tests use; an unseeded statistical assertion is a test that fails for someone else
at some percentage rate, which is worse than no test.

Never "fix" a test by loosening a tolerance without understanding why the error
grew. An unexplained loosening hides the regression the tolerance was there to
catch.

## Examples

`examples/` carries worked problems rather than snippets: an A/B test answered
frequentist *and* Bayesian, a Weibull reliability fit, a Poisson arrival process.
They are the answer to "how do I actually use this", so they are prose-commented
and print a narrative.

Two rules:

- **An example asserts everything it prints.** `cargo test --all-targets`
  compiles examples but does not run them, so an un-asserted example rots
  silently into a program that prints something wrong. Every number in an output
  line should have an `assert!` behind it.
- **Prefer examples that cross-check.** `ab_test` is the model: it computes the
  same answer two ways and prints both, so the reader learns the API *and* sees
  the one-sided frequentist p-value land on the Bayesian posterior tail to three
  decimals. That is documentation and verification in the same lines.

## Numerical Work

Numeric changes carry risk that ordinary review does not catch, so they get their
own bar. The usual failure is not a wrong formula but a right formula evaluated
the wrong way.

- **Work in the log domain.** Densities underflow. `ln_pdf` is the primitive and
  `pdf` is `ln_pdf().exp()`, not the other way round.
- **Tails are where the naive form dies.** `cdf(-40.0).ln()` is `-inf`;
  `ln_cdf(-40.0)` must not be. Test the far tail explicitly — there is a suite of
  these already, and they exist because the naive versions passed every test in
  the normal range.
- **Catastrophic cancellation.** Prefer `ln_1p`, `exp_m1`, `ln_1p_exp`, and the
  `sf`/`ln_sf` survival forms over `1.0 - cdf(x)`, which loses every significant
  digit in the upper tail.
- **Lift constants, do not write them.** `Real::scalar(2.0)`, not `2.0` — a
  literal does not exist on a deferred carrier. euclid's `[workspace.lints]`
  reasoning applies here too: the rule looks arbitrary until you try to run a
  formula on an expression handle.

If you need a special function — `ln_gamma`, `digamma`, `beta_reg`, `gamma_lr` —
it belongs in euclid, not here. bayes composes closed forms out of euclid's
numeric surface; adding a hand-rolled series approximation to bayes is a sign the
change belongs upstream.

## Tracking euclid

bayes consumes euclid from git `main` with a semver assertion:

```toml
euclid-numeric = { version = "0.3.0", git = "...", branch = "main" }
```

The `version` is not redundant with the `git` source. euclid is unpublished, so
the branch is what gets fetched — but the version makes a euclid that moves to
`0.4` fail resolution here rather than silently changing a formula's meaning.
When euclid's minor version moves, bump it here in the same change that adapts to
it, and say so in the PR title.

Because euclid is fetched from a branch rather than a tag, `Cargo.lock` is the
only record of *which* euclid a given bayes commit was tested against. It is
committed, and it should stay committed.

## License

By contributing you agree your contributions are licensed under the
[Apache License, Version 2.0](LICENSE), per section 5 of that license.
