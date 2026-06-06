//! Ordinary least squares: **simple linear regression** ([`LinearRegression`] —
//! fit `y ≈ slope·x + intercept`, with `R²`) and **multiple regression over an
//! explicit design matrix** ([`fit_least_squares`]).

mod least_squares;
mod linear;

pub use least_squares::fit_least_squares;
pub use linear::LinearRegression;
