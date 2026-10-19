mod chi_squared;
mod exponential;
mod fisher_snedecor;
// This directory is the gamma *family*; `gamma` within it is the Gamma
// distribution itself, alongside its four relatives. Renaming either would
// describe the code less well than the repetition does.
#[allow(clippy::module_inception)]
mod gamma;
mod inverse_gamma;

pub use chi_squared::ChiSquared;
pub use exponential::Exponential;
pub use fisher_snedecor::FisherSnedecor;
pub use gamma::Gamma;
pub use inverse_gamma::InverseGamma;
