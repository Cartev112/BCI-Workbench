from __future__ import annotations

from collections import Counter
from typing import Any

from bciworkbench.ontology.packets import IntentPacket


def decoder_metrics(predictions: list[IntentPacket]) -> dict[str, Any]:
    if not predictions:
        return {"accuracy": None, "balanced_accuracy": None, "n_predictions": 0}

    labels = sorted({item.label for item in predictions if item.label is not None})
    correct = sum(1 for item in predictions if item.label == item.intent)
    per_class: dict[str, float] = {}
    for label in labels:
        class_items = [item for item in predictions if item.label == label]
        if class_items:
            per_class[label] = sum(1 for item in class_items if item.intent == label) / len(class_items)
    confusion: dict[str, dict[str, int]] = {}
    for item in predictions:
        true_label = item.label or "unknown"
        confusion.setdefault(true_label, {})
        confusion[true_label][item.intent] = confusion[true_label].get(item.intent, 0) + 1

    pred_counts = Counter(item.intent for item in predictions)
    return {
        "accuracy": correct / len(predictions),
        "balanced_accuracy": sum(per_class.values()) / len(per_class) if per_class else None,
        "n_predictions": len(predictions),
        "class_accuracy": per_class,
        "confusion_matrix": confusion,
        "prediction_counts": dict(pred_counts),
        "mean_confidence": sum(item.confidence for item in predictions) / len(predictions),
        "mean_decoder_latency_ms": sum(item.latency_ms for item in predictions) / len(predictions),
    }

