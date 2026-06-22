//! Distribution-family implementation modules.
//!
//! The implementation home for distribution-specific kernels is here. Use
//! `families::normal::*`, `families::poisson::*`, and similar modules when you
//! want to understand one distribution end-to-end. Operation modules such as
//! [`crate::density`] and [`crate::cdf`] re-export these same functions as
//! purpose-oriented views.

mod common;

pub mod continuous;
pub mod discrete;

pub use continuous::*;
pub use discrete::*;
