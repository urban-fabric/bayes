//! **Composition combinators** — build new distributions from existing ones, in
//! closed form. A [`Mixture`] is a weighted blend of components with stable
//! log-densities and component responsibilities; a [`Truncated`] restricts a
//! continuous distribution to an interval and renormalizes. (The other combinator,
//! [`Transformed`](crate::distribution::Transformed), lives in the `transformed`
//! module.)
//!
//! All are *closed-form / direct-sampling* constructors — in scope per the
//! library's dividing principle.

mod mixture;
mod truncated;

pub use mixture::Mixture;
pub use truncated::Truncated;
