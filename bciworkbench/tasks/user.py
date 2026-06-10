from __future__ import annotations

from dataclasses import dataclass, field
from bciworkbench.ontology.packets import IntentPacket


@dataclass
class DelayedIntentUserModel:
    """Apply decoder intents through a feedback/action delay queue."""

    feedback_delay_ms: float = 0.0
    control_interval_s: float = 0.5
    confidence_threshold: float = 0.0
    _queue: list[tuple[int, float, str]] = field(default_factory=list)

    @property
    def delay_steps(self) -> int:
        if self.feedback_delay_ms <= 0:
            return 0
        return int((self.feedback_delay_ms / 1000.0) // self.control_interval_s)

    def reset(self) -> None:
        self._queue.clear()

    def choose_action(self, intent: IntentPacket) -> tuple[int, float, str]:
        direction = intent_direction(intent.intent)
        confidence = float(intent.confidence)
        if confidence < self.confidence_threshold:
            direction = 0
        desired = (direction, confidence, intent.intent)
        if self.delay_steps <= 0:
            return desired
        self._queue.append(desired)
        if len(self._queue) <= self.delay_steps:
            return (0, 0.0, "delayed_noop")
        return self._queue.pop(0)


def intent_direction(label: str | None) -> int:
    if label is None:
        return 0
    lowered = str(label).lower()
    if "left" in lowered or lowered in {"non_target", "nontarget", "0"}:
        return -1
    if "right" in lowered or lowered in {"target", "1"}:
        return 1
    return 0
