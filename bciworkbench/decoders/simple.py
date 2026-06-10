from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from bciworkbench.ontology.packets import FeaturePacket, IntentPacket


@dataclass(frozen=True)
class DecoderResult:
    predictions: list[IntentPacket]
    train_size: int
    test_size: int
    decoder_name: str


class _NearestCentroid:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "_NearestCentroid":
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([x[y == label].mean(axis=0) for label in self.classes_])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        distances = ((x[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(distances, axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        distances = ((x[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        scores = -distances
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)


class SupervisedDecoder:
    def __init__(self, estimator: str = "lda", calibration_fraction: float = 0.6) -> None:
        if not 0 < calibration_fraction < 1:
            raise ValueError("calibration_fraction must be between 0 and 1")
        self.estimator = estimator
        self.calibration_fraction = calibration_fraction

    @classmethod
    def from_params(cls, params: dict) -> "SupervisedDecoder":
        return cls(
            estimator=str(params.get("estimator", "lda")),
            calibration_fraction=float(params.get("calibration_fraction", 0.6)),
        )

    def fit_predict(self, features: list[FeaturePacket]) -> DecoderResult:
        labeled = [item for item in features if item.label is not None]
        if len(labeled) < 4:
            raise ValueError("at least 4 labeled feature packets are required")
        x = np.vstack([item.features for item in labeled])
        y = np.asarray([item.label for item in labeled])
        split = max(2, min(len(labeled) - 1, int(round(len(labeled) * self.calibration_fraction))))
        train_x, test_x = x[:split], x[split:]
        train_y, test_y = y[:split], y[split:]
        model, decoder_name = self._build_model()
        model.fit(train_x, train_y)

        start = perf_counter()
        pred_y = model.predict(test_x)
        elapsed_ms = (perf_counter() - start) * 1000.0
        per_prediction_ms = elapsed_ms / max(len(test_y), 1)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(test_x)
            classes = [str(item) for item in model.classes_]
        else:
            probabilities = np.ones((len(test_y), 1), dtype=float)
            classes = [str(pred_y[0])]

        predictions: list[IntentPacket] = []
        for index, (feature, pred, label) in enumerate(zip(labeled[split:], pred_y, test_y, strict=True)):
            posterior = {classes[col]: float(probabilities[index, col]) for col in range(len(classes))}
            confidence = max(posterior.values()) if posterior else 1.0
            predictions.append(
                IntentPacket(
                    intent_id=f"prediction-{index:04d}",
                    intent=str(pred),
                    confidence=confidence,
                    posterior=posterior,
                    latency_ms=per_prediction_ms,
                    window_id=feature.window_id,
                    decoder_id=decoder_name,
                    label=str(label),
                )
            )
        return DecoderResult(
            predictions=predictions,
            train_size=len(train_y),
            test_size=len(test_y),
            decoder_name=decoder_name,
        )

    def _build_model(self):
        if self.estimator == "lda":
            try:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

                return LinearDiscriminantAnalysis(), "sklearn_lda"
            except Exception:
                return _NearestCentroid(), "nearest_centroid_fallback"
        if self.estimator in {"nearest_centroid", "centroid"}:
            return _NearestCentroid(), "nearest_centroid"
        raise ValueError(f"unsupported estimator: {self.estimator}")

