from __future__ import annotations

from typing import Any

from bciworkbench.adaptation.recalibration import (
    ConfidenceGatedRecalibrationAdapter,
    DriftTriggeredRecalibrationAdapter,
    NoOpAdapter,
    SupervisedBatchRecalibrationAdapter,
)


def build_adaptation_adapter(params: dict[str, Any] | None):
    params = dict(params or {})
    adapter_type = str(params.pop("type", "none"))
    if adapter_type in {"none", "noop"}:
        return NoOpAdapter()
    if adapter_type == "supervised_batch":
        return SupervisedBatchRecalibrationAdapter(
            batch_size=int(params.get("batch_size", 8)),
            min_samples=int(params.get("min_samples", 2)),
        )
    if adapter_type == "confidence_gated":
        return ConfidenceGatedRecalibrationAdapter(
            confidence_gate=float(params.get("confidence_gate", 0.8)),
            batch_size=int(params.get("batch_size", 8)),
            min_samples=int(params.get("min_samples", 2)),
        )
    if adapter_type == "drift_triggered":
        return DriftTriggeredRecalibrationAdapter(
            accuracy_floor=float(params.get("accuracy_floor", 0.7)),
            confidence_floor=float(params.get("confidence_floor", 0.55)),
            batch_size=int(params.get("batch_size", 8)),
            min_samples=int(params.get("min_samples", 2)),
        )
    raise ValueError(f"unsupported adaptation type: {adapter_type}")
