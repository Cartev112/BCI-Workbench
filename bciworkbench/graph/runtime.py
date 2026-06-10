from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from bciworkbench.graph.context import RunContext
from bciworkbench.graph.node import Node


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TelemetryRecord:
    node_id: str
    node_type: str
    started_at: str
    finished_at: str
    duration_ms: float
    status: str
    input_type: str | None
    output_type: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LinearRuntime:
    """Deterministic linear graph executor."""

    def __init__(self, nodes: list[Node]) -> None:
        if not nodes:
            raise ValueError("LinearRuntime requires at least one node")
        self.nodes = nodes
        self.telemetry: list[TelemetryRecord] = []

    def describe_graph(self) -> dict[str, Any]:
        return {
            "runtime": "linear",
            "nodes": [node.describe() for node in self.nodes],
            "edges": [
                {"from": self.nodes[index].node_id, "to": self.nodes[index + 1].node_id}
                for index in range(len(self.nodes) - 1)
            ],
        }

    def run(self, context: RunContext) -> Any:
        payload: Any = None
        for node in self.nodes:
            node.setup(context)
            started_at = utc_now_iso()
            start = perf_counter()
            input_type = type(payload).__name__ if payload is not None else None
            try:
                payload = node.process(payload, context)
            except Exception as exc:
                finished_at = utc_now_iso()
                self.telemetry.append(
                    TelemetryRecord(
                        node_id=node.node_id,
                        node_type=node.node_type,
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_ms=(perf_counter() - start) * 1000.0,
                        status="error",
                        input_type=input_type,
                        output_type=None,
                        error=str(exc),
                    )
                )
                raise
            finally:
                node.teardown(context)

            finished_at = utc_now_iso()
            self.telemetry.append(
                TelemetryRecord(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=(perf_counter() - start) * 1000.0,
                    status="ok",
                    input_type=input_type,
                    output_type=type(payload).__name__ if payload is not None else None,
                )
            )
        return payload

