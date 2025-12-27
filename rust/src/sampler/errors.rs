/// Sampler termination reason enum
#[derive(Clone, Debug)]
pub enum SamplerTermination {
    EvidenceConverged,
    InformationConverged,
    MaxIterationReached,
    InsufficientLivePoints,
    EvaluationFailed(String),
}
