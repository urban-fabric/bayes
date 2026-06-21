//! Lazy-real kernel trait implementations for the concrete families.
//!
//! These impls keep the algebraic probability formulas on the lazy numeric
//! surfaces. Support checks are expressed with lazy masks/selects so the same
//! traits work for host scalars, expression backends, and tensor backends.

mod continuous_density;
mod continuous_traits;
mod discrete;
mod moments;

use euclid::LazyReal;

fn neg_inf<T: LazyReal>() -> T {
    T::neg_infinity()
}
