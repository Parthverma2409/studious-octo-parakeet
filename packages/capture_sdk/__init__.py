"""FR7: forward agent trajectories to the evaluation platform, asynchronously.

The contract with the rest of the system is one sentence: nothing in here may
appear in the latency of a user-facing response, including when the ingest
service is down (UC4 E1).
"""

from packages.capture_sdk.buffer import BoundedBuffer
from packages.capture_sdk.client import CaptureClient, CaptureMetrics
from packages.capture_sdk.instrument import CAPTURED_METHODS, InstrumentedAgent, instrument
from packages.capture_sdk.transport import (
    HttpTransport,
    MemoryTransport,
    Transport,
    TransportError,
)

__all__ = [
    "BoundedBuffer",
    "CAPTURED_METHODS",
    "CaptureClient",
    "CaptureMetrics",
    "HttpTransport",
    "InstrumentedAgent",
    "MemoryTransport",
    "Transport",
    "TransportError",
    "instrument",
]
