use core::fmt;

/// Why a distribution constructor (or fit) rejected its parameters.
///
/// A single shared error for host-side probability validation. Per-distribution
/// error enums are deferred until a distribution needs context this cannot carry.
#[derive(Clone, Debug, PartialEq)]
pub enum BayesError {
    /// A probability was outside `[0, 1]`.
    ProbabilityOutOfRange { value: f64 },
    /// A parameter that must be strictly positive was not.
    NotPositive { name: &'static str, value: f64 },
    /// A real-valued range was empty.
    EmptyRange { lo: f64, hi: f64 },
    /// An integer-valued range was empty.
    EmptyIntRange { lo: i64, hi: i64 },
    /// A parameter was not finite.
    NonFinite { name: &'static str, value: f64 },
    /// A parameter exceeded an implementation-defined upper bound.
    AboveMaximum {
        name: &'static str,
        value: f64,
        max: f64,
    },
    /// A structured input was malformed.
    Malformed { what: &'static str },
}

impl fmt::Display for BayesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProbabilityOutOfRange { value } => {
                write!(f, "probability must be in [0, 1] (got {value})")
            }
            Self::NotPositive { name, value } => {
                write!(f, "{name} must be strictly positive (got {value})")
            }
            Self::EmptyRange { lo, hi } => {
                write!(f, "range must satisfy lo < hi (got lo={lo}, hi={hi})")
            }
            Self::EmptyIntRange { lo, hi } => {
                write!(f, "range must satisfy lo < hi (got lo={lo}, hi={hi})")
            }
            Self::NonFinite { name, value } => {
                write!(f, "{name} must be a finite number (got {value})")
            }
            Self::AboveMaximum { name, value, max } => {
                write!(f, "{name}={value} exceeds the supported upper bound {max}")
            }
            Self::Malformed { what } => write!(f, "malformed input: {what}"),
        }
    }
}

impl std::error::Error for BayesError {}

/// Former name of [`BayesError`], kept as a compatibility alias.
pub type StatsError = BayesError;
