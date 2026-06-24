//! Closure-backed distribution adapters.
//!
//! These lift user-supplied PDF/PMF/CDF/quantile closures into the same
//! distribution trait spine as the named families. They are generic over the
//! same Euclid scalar surfaces as the lazy named families.

mod continuous;
mod discrete;
mod ext;

pub use continuous::{ContinuousDensity, ContinuousFn};
pub use discrete::{DiscreteFn, DiscreteMass};
pub use ext::{ContinuousCdfExt, ContinuousDistributionExt, DiscreteDistributionExt};
