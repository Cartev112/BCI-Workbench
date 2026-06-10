from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from bciworkbench.ontology.packets import AdaptationPacket, IntentPacket


@dataclass(frozen=True)
class AdaptationResult:
    predictions: list[IntentPacket]
    packets: list[AdaptationPacket] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class AdaptationAdapter(Protocol):
    adapter_id: str

    def adapt(self, predictions: list[IntentPacket]) -> AdaptationResult:
        ...
