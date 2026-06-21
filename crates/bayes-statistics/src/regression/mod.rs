//! Ordinary least squares: **simple linear regression** ([`LinearRegression`] —
//! fit `y ≈ slope·x + intercept`, with `R²`). Multiple regression over an explicit
//! design matrix (`fit_least_squares`) is matrix-coupled and lives in the sibling
//! crate `ito`.

mod linear;

pub use linear::LinearRegression;
