from __future__ import annotations

from typing import Any

from bciworkbench.eval.metrics import decoder_metrics
from bciworkbench.ontology.packets import IntentPacket


def prediction_metrics(predictions: list[IntentPacket]) -> dict[str, Any]:
    metrics = decoder_metrics(predictions)
    return {
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "n_predictions": metrics.get("n_predictions"),
        "mean_confidence": metrics.get("mean_confidence"),
    }


def adaptation_stability_metrics(
    original: list[IntentPacket],
    adapted: list[IntentPacket],
    packets: list,
    catastrophic_drop_threshold: float = 0.1,
) -> dict[str, Any]:
    before = prediction_metrics(original)
    after = prediction_metrics(adapted)
    changed = sum(1 for old, new in zip(original, adapted, strict=True) if old.intent != new.intent)
    confidence_delta = [
        abs(float(old.confidence) - float(new.confidence))
        for old, new in zip(original, adapted, strict=True)
    ]
    before_balanced = before.get("balanced_accuracy")
    after_balanced = after.get("balanced_accuracy")
    drop = None
    catastrophic = False
    if before_balanced is not None and after_balanced is not None:
        drop = float(before_balanced) - float(after_balanced)
        catastrophic = drop > catastrophic_drop_threshold
    return {
        "adaptation_update_count": len(packets),
        "adaptation_update_rate": len(packets) / max(len(original), 1),
        "adaptation_changed_prediction_count": changed,
        "adaptation_changed_prediction_rate": changed / max(len(original), 1),
        "adaptation_mean_abs_confidence_change": sum(confidence_delta) / len(confidence_delta) if confidence_delta else 0.0,
        "adaptation_accuracy_before": before.get("accuracy"),
        "adaptation_accuracy_after": after.get("accuracy"),
        "adaptation_balanced_accuracy_before": before_balanced,
        "adaptation_balanced_accuracy_after": after_balanced,
        "adaptation_balanced_accuracy_drop": drop,
        "adaptation_catastrophic_update_warning": catastrophic,
    }
