//! One uniform view over every family, plus **the** canonical parameter sweep.
//!
//! **Adding a family here adds it to every suite at once.** That is the point:
//! the conformance, sampling and relationship suites all iterate these lists, so
//! a new family is swept by all of them without a new file in each.
//!
//! The parameter sets are chosen to break things — scales from `1e-9` to `1e10`,
//! shapes from `1e-3` to `1e4`, a location so far from the origin that the
//! family's own quantiles stop being representable, and degrees of freedom where
//! moments cease to exist.

mod continuous;
mod discrete;

pub use continuous::{
    ContCase, ContView, continuous_cases, quantile_unrepresentable_at,
    quantiles_are_unrepresentable,
};
pub use discrete::{
    DiscCase, DiscSampleCase, DiscSampleView, DiscView, discrete_cases, discrete_sampler_cases,
    summation_limit,
};
