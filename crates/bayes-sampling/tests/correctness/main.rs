//! Do the samplers draw from the distributions they claim to?
//!
//! The only question that matters here, and the one no identity can answer: a
//! sampler is a separate algorithm from the density it targets, so nothing in
//! the closed-form surface notices if it draws from the wrong law. Each sampler
//! is therefore pointed at a target whose exact distribution bayes already knows
//! independently, and judged statistically.
//!
//! Every threshold is **derived**, not tuned — a Kolmogorov series for the
//! continuous cases, Pearson's chi-squared for the discrete ones, a
//! central-limit standard error for the moments. They live in `bayes_utils::gof`,
//! which shares no code with what it judges.
//!
//! | module | covers |
//! |--------|--------|
//! | [`alias`] | the O(1) alias table, and its agreement with `Categorical` |
//! | [`continuous`] | rejection and adaptive rejection, plus the log-concavity check |

mod alias;
mod continuous;
