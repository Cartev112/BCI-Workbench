from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from bciworkbench.adaptation.base import AdaptationResult
from bciworkbench.adaptation.metrics import adaptation_stability_metrics, prediction_metrics
from bciworkbench.ontology.packets import AdaptationPacket, IntentPacket


@dataclass(frozen=True)
class NoOpAdapter:
    adapter_id: str = "adaptation.noop"

    def adapt(self, predictions: list[IntentPacket]) -> AdaptationResult:
        return AdaptationResult(
            predictions=predictions,
            packets=[
                AdaptationPacket(
                    adapter_id=self.adapter_id,
                    update_type="adaptation.noop",
                    input_window_ids=tuple(prediction.window_id for prediction in predictions),
                    labels=tuple(prediction.label for prediction in predictions if prediction.label is not None),
                    metrics_before=prediction_metrics(predictions),
                    metrics_after=prediction_metrics(predictions),
                )
            ],
            metrics=adaptation_stability_metrics(predictions, predictions, []),
        )


@dataclass(frozen=True)
class SupervisedBatchRecalibrationAdapter:
    batch_size: int = 8
    min_samples: int = 2
    adapter_id: str = "adaptation.supervised_batch"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")

    def adapt(self, predictions: list[IntentPacket]) -> AdaptationResult:
        return _recalibrate_predictions(
            predictions=predictions,
            adapter_id=self.adapter_id,
            batch_size=self.batch_size,
            min_samples=self.min_samples,
            confidence_gate=None,
            update_type="adaptation.supervised_batch_recalibration",
            trigger=None,
        )


@dataclass(frozen=True)
class ConfidenceGatedRecalibrationAdapter:
    confidence_gate: float = 0.8
    batch_size: int = 8
    min_samples: int = 2
    adapter_id: str = "adaptation.confidence_gated"

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_gate <= 1:
            raise ValueError("confidence_gate must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")

    def adapt(self, predictions: list[IntentPacket]) -> AdaptationResult:
        return _recalibrate_predictions(
            predictions=predictions,
            adapter_id=self.adapter_id,
            batch_size=self.batch_size,
            min_samples=self.min_samples,
            confidence_gate=self.confidence_gate,
            update_type="adaptation.confidence_gated_recalibration",
            trigger=None,
        )


@dataclass(frozen=True)
class DriftTriggeredRecalibrationAdapter:
    accuracy_floor: float = 0.7
    confidence_floor: float = 0.55
    batch_size: int = 8
    min_samples: int = 2
    adapter_id: str = "adaptation.drift_triggered"

    def __post_init__(self) -> None:
        if not 0 <= self.accuracy_floor <= 1:
            raise ValueError("accuracy_floor must be between 0 and 1")
        if not 0 <= self.confidence_floor <= 1:
            raise ValueError("confidence_floor must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")

    def adapt(self, predictions: list[IntentPacket]) -> AdaptationResult:
        def trigger(chunk: list[IntentPacket]) -> bool:
            metrics = prediction_metrics(chunk)
            balanced = metrics.get("balanced_accuracy")
            confidence = metrics.get("mean_confidence")
            return (
                balanced is not None
                and float(balanced) < self.accuracy_floor
                or confidence is not None
                and float(confidence) < self.confidence_floor
            )

        return _recalibrate_predictions(
            predictions=predictions,
            adapter_id=self.adapter_id,
            batch_size=self.batch_size,
            min_samples=self.min_samples,
            confidence_gate=None,
            update_type="adaptation.drift_triggered_recalibration",
            trigger=trigger,
        )


def _recalibrate_predictions(
    predictions: list[IntentPacket],
    adapter_id: str,
    batch_size: int,
    min_samples: int,
    confidence_gate: float | None,
    update_type: str,
    trigger,
) -> AdaptationResult:
    correction_map: dict[str, str] = {}
    adapted: list[IntentPacket] = []
    packets: list[AdaptationPacket] = []

    for start in range(0, len(predictions), batch_size):
        chunk = predictions[start : start + batch_size]
        adapted_chunk = [_apply_correction(prediction, correction_map, adapter_id) for prediction in chunk]
        adapted.extend(adapted_chunk)

        if trigger is not None and not trigger(adapted_chunk):
            continue
        eligible = [
            prediction
            for prediction in adapted_chunk
            if prediction.label is not None and (confidence_gate is None or prediction.confidence >= confidence_gate)
        ]
        new_map = _learn_correction_map(eligible, min_samples=min_samples)
        if not new_map:
            continue
        before = prediction_metrics(adapted_chunk)
        after_chunk = [_apply_correction(prediction, new_map, adapter_id) for prediction in chunk]
        after = prediction_metrics(after_chunk)
        previous_map = dict(correction_map)
        correction_map.update(new_map)
        packets.append(
            AdaptationPacket(
                adapter_id=adapter_id,
                update_type=update_type,
                input_window_ids=tuple(prediction.window_id for prediction in eligible),
                labels=tuple(prediction.label for prediction in eligible if prediction.label is not None),
                confidence_gate=confidence_gate,
                parameters_changed={
                    "previous_correction_map": previous_map,
                    "new_correction_map": dict(correction_map),
                    "batch_start": start,
                    "batch_size": len(chunk),
                },
                metrics_before=before,
                metrics_after=after,
            )
        )

    return AdaptationResult(
        predictions=adapted,
        packets=packets,
        metrics=adaptation_stability_metrics(predictions, adapted, packets),
    )


def _learn_correction_map(predictions: list[IntentPacket], min_samples: int) -> dict[str, str]:
    label_counts: dict[str, Counter] = defaultdict(Counter)
    for prediction in predictions:
        if prediction.label is None:
            continue
        label_counts[prediction.intent][prediction.label] += 1
    learned: dict[str, str] = {}
    for intent, counts in label_counts.items():
        label, count = counts.most_common(1)[0]
        if count >= min_samples:
            learned[intent] = str(label)
    return learned


def _apply_correction(prediction: IntentPacket, correction_map: dict[str, str], adapter_id: str) -> IntentPacket:
    corrected = correction_map.get(prediction.intent)
    if corrected is None or corrected == prediction.intent:
        return prediction
    posterior = dict(prediction.posterior)
    if corrected not in posterior:
        posterior[corrected] = prediction.confidence
    return IntentPacket(
        intent_id=prediction.intent_id,
        intent=corrected,
        confidence=prediction.confidence,
        posterior=posterior,
        latency_ms=prediction.latency_ms,
        window_id=prediction.window_id,
        decoder_id=prediction.decoder_id,
        label=prediction.label,
        metadata={
            **prediction.metadata,
            "adapted_by": adapter_id,
            "original_intent": prediction.intent,
        },
    )
