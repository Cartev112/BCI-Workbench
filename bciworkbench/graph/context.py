from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bciworkbench.ontology.schemas import ExperimentSpec


@dataclass
class RunContext:
    """Shared execution state for a BCI Workbench run."""

    run_id: str
    run_dir: Path
    spec: ExperimentSpec
    started_at: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

