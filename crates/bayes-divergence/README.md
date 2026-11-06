# bayes-divergence

**How far apart are two distributions? In closed form.**

## The problem

The usual answer is to sample: draw from both, estimate the expectation, accept
the Monte Carlo error. That is wasteful when the answer is exact — the KL
divergence between two Gaussians is four arithmetic operations and a logarithm —
and it is misleading when the estimate's error is comparable to the quantity
being measured, which is exactly the regime where a divergence matters.

Every divergence here is a closed form between parameterized families. No
sampling, no quadrature, no tolerance.

## What it holds

KL and its reverse, the Jeffreys symmetrization, squared Hellinger, Rényi of any
order, χ², total variation, Jensen–Shannon, and the general *f*-divergence with
pluggable generators. Plus the discrete information-theory surface: Shannon,
joint and conditional entropy, mutual information and its normalizations,
variation of information, Bhattacharyya coefficient, and the maximum-entropy
bounds.

## Coverage is uneven, and says so

KL has the widest closed-form coverage. Hellinger, Rényi and χ² close across the
Gaussian, gamma, directional, inverse-Gaussian, uniform and finite-support
discrete families. Jensen–Shannon and total variation stay on the finite-support
discretes, where the mixture stays in-family. Several heavy-tailed families
expose KL only, because their other divergences have no general closed form.

That is a property of the mathematics, not a gap in the implementation, and the
docs name which is which rather than silently falling back to an approximation.

## The *f*-divergence umbrella

Each per-family closed form instantiates one general object,
`D_f(p ‖ q) = Σ qᵢ f(pᵢ/qᵢ)` for convex `f` with `f(1) = 0`. The generator
surface exposes that directly, with a `Channel` carrying the data-processing
inequality. The three universal guarantees — non-negativity, data-processing
monotonicity, joint convexity — follow from convexity alone, and are
machine-checked in the sibling `riemann` Lean development.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
