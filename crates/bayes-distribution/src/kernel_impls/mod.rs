//! Lazy-real kernel trait implementations for the concrete families.
//!
//! These impls keep the algebraic probability formulas on the lazy numeric
//! surfaces. Support checks are expressed with lazy masks/selects so the same
//! traits work for host scalars, expression backends, and tensor backends.

mod continuous_density;
mod continuous_traits;
mod discrete;

use euclid::LazyReal;

fn scalar<T: LazyReal>(x: f64) -> T {
    T::scalar(x)
}

fn neg_inf<T: LazyReal>() -> T {
    T::scalar(f64::NEG_INFINITY)
}
