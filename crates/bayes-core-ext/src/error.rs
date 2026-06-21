use thiserror::Error;

/// Why a distribution constructor (or fit) rejected its parameters.
///
/// A single shared error for the eager probability/statistics layer;
/// constructors return `Result<Self, BayesError>`. Per-distribution error enums
/// are deferred until a distribution needs context this cannot carry.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum BayesError {
    #[error("probability must be in [0, 1] (got {value})")]
    ProbabilityOutOfRange { value: f64 },

    #[error("{name} must be strictly positive (got {value})")]
    NotPositive { name: &'static str, value: f64 },

    #[error("range must satisfy lo < hi (got lo={lo}, hi={hi})")]
    EmptyRange { lo: f64, hi: f64 },

    #[error("range must satisfy lo < hi (got lo={lo}, hi={hi})")]
    EmptyIntRange { lo: i64, hi: i64 },

    #[error("{name} must be a finite number (got {value})")]
    NonFinite { name: &'static str, value: f64 },

    #[error("{name}={value} exceeds the supported upper bound {max}")]
    AboveMaximum {
        name: &'static str,
        value: f64,
        max: f64,
    },

    #[error("malformed input: {what}")]
    Malformed { what: &'static str },
}

/// Former name of [`BayesError`], kept as a compatibility alias.
pub type StatsError = BayesError;
