"""
Metric depth backends.

``SimulatedDepthSource`` is deterministic and dependency-light (tests, demos,
benchmarks). ``ZoeDepthBackend`` is a lazy, optional metric monocular-depth
backend (needs ``torch`` + ``transformers`` + a one-time model download).
"""
from .metric_depth import ZoeDepthBackend
from .synthetic import SimulatedDepthSource

__all__ = ["SimulatedDepthSource", "ZoeDepthBackend"]