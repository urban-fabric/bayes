//! Backend-generic support, CDF, and quantile impls for the lazy families.
//!
//! Split by what the CDF needs to evaluate, which is also what decides whether
//! the quantile is elementary or iterative:
//!
//! | module | families |
//! |--------|----------|
//! | [`symmetric`] | real-line and elementary-CDF families; exact inverses |
//! | [`positive`] | half-line and unit-interval families; incomplete gamma/beta |
//! | [`discrete`] | the count families; integer bisection quantiles |

mod discrete;
mod positive;
mod symmetric;
