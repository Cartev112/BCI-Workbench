from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bciworkbench.ontology.packets import FeaturePacket, IntentPacket


@dataclass(frozen=True)
class DecoderResult:
    predictions: list[IntentPacket]
    train_size: int
    test_size: int
    decoder_name: str
    calibration_time_s: float
    model_card: dict[str, Any] = field(default_factory=dict)
    model_path: Path | None = None


class Decoder(Protocol):
    """Minimal supervised decoder contract for Phase 4."""

    decoder_name: str

    def fit_predict(self, features: list[FeaturePacket]) -> DecoderResult:
        ...

    def save(self, path: Path) -> None:
        ...

    def model_card(self) -> dict[str, Any]:
        ...

