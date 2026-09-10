"""FR8: normalise, validate, score and store the trajectories FR7 forwards.

Two invariants hold everywhere below. A trajectory that fails the frozen schema
is quarantined and surfaced, never dropped (UC4 E2); and which evaluators run is
decided by the trajectory's archetype, never by the caller.
"""

from packages.eval_ingest.pipeline import IngestPipeline, IngestResult, IngestStats
from packages.eval_ingest.quarantine import Quarantine, QuarantinedTrajectory
from packages.eval_ingest.registry import EvaluatorRegistry, UnknownArchetypeError, default_registry
from packages.eval_ingest.service import IngestService
from packages.eval_ingest.store import InMemoryStore, JsonlFileStore, TrajectoryStore
from packages.eval_ingest.validation import is_valid, validation_errors

__all__ = [
    "EvaluatorRegistry",
    "InMemoryStore",
    "IngestPipeline",
    "IngestResult",
    "IngestService",
    "IngestStats",
    "JsonlFileStore",
    "Quarantine",
    "QuarantinedTrajectory",
    "TrajectoryStore",
    "UnknownArchetypeError",
    "default_registry",
    "is_valid",
    "validation_errors",
]
