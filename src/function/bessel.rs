//! Modified Bessel functions of the first kind — now owned by [`euclid::special`]
//! and re-exported here. `I₀`/`I₁` (plus the scaled `I₀ᵉ`/`I₁ᵉ`, the stable ratio
//! `I₁/I₀`, `ln I₀`, and the general-order `ln Iᵥ`) are the normalising constants
//! and moments of the directional distributions, and the concrete reference behind
//! the [`euclid::numeric::LazyReal`] `bessel_i0e`/`bessel_i1e` methods.

pub use euclid::special::bessel::{i0, i0e, i1, i1_over_i0, i1e, ln_i0, ln_iv};
