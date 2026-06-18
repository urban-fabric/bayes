//! The beta function family (`ln_beta`/`beta`, regularized incomplete beta
//! `beta_reg`) — now owned by [`euclid::special`] and re-exported here. `beta_reg`
//! `Iₓ(a, b)` is the CDF kernel for the Beta, Student's-t, and F distributions, and
//! the concrete reference behind the [`euclid::numeric::LazyReal::beta_reg`] method.

pub use euclid::special::beta::{beta, beta_reg, ln_beta};
