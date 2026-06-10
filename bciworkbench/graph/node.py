from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bciworkbench.graph.context import RunContext


class Node(ABC):
    """Base class for deterministic milestone graph nodes."""

    def __init__(self, node_id: str, node_type: str, params: dict[str, Any] | None = None) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.params = params or {}

    def setup(self, context: RunContext) -> None:
        return None

    @abstractmethod
    def process(self, payload: Any, context: RunContext) -> Any:
        raise NotImplementedError

    def teardown(self, context: RunContext) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "params": self.params,
        }

