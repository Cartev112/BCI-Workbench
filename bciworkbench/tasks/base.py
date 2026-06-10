from __future__ import annotations

from typing import Protocol

from bciworkbench.graph.context import RunContext
from bciworkbench.ontology.packets import FeedbackPacket, IntentPacket, TaskStatePacket


class TaskEnvironment(Protocol):
    """Closed-loop task contract."""

    def reset(self, context: RunContext | None = None) -> TaskStatePacket:
        ...

    def step(
        self,
        intent: IntentPacket,
        context: RunContext | None = None,
    ) -> tuple[TaskStatePacket, FeedbackPacket]:
        ...
