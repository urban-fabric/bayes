//! Conformance of the **continuous** families, split by the kind of invariant.
//!
//! | module | asks |
//! |--------|------|
//! | [`shape`] | is the CDF a CDF and the density a density? |
//! | [`quantile`] | does the quantile invert the CDF, including at the bounds? |
//! | [`cross_checks`] | do two independent code paths agree on the same number? |
//! | [`hostile`] | do infinities, `NaN`s and bad parameters behave? |

mod cross_checks;
mod hostile;
mod quantile;
mod shape;
