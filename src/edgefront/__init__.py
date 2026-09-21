"""edgefront - measure whether you need a hosted decision model."""

from .bench import BenchConfig, run_backend
from .frontier import decide, pareto
from .types import Backend, BackendResult, Example, Prediction, TaskSpec

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "BackendResult",
    "BenchConfig",
    "Example",
    "Prediction",
    "TaskSpec",
    "decide",
    "pareto",
    "run_backend",
]
