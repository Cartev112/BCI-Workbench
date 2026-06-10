from __future__ import annotations

import pickle
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from bciworkbench.decoders.base import DecoderResult
from bciworkbench.ontology.packets import FeaturePacket, IntentPacket


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


class SklearnDecoder:
    """Supervised decoder adapter for sklearn-like estimators.

    Supports sklearn LDA and logistic regression when sklearn is installed,
    plus a deterministic nearest-centroid fallback that keeps core examples
    runnable without optional dependencies.
    """

    def __init__(self, estimator: str = "lda", calibration_fraction: float = 0.6) -> None:
        if not 0 < calibration_fraction < 1:
            raise ValueError("calibration_fraction must be between 0 and 1")
        self.estimator = estimator
        self.calibration_fraction = calibration_fraction
        self.decoder_name = "unfit"
        self.model: Any | None = None
        self.classes_: list[str] = []
        self.feature_names_: tuple[str, ...] = ()
        self.train_size_: int = 0
        self.test_size_: int = 0
        self.calibration_time_s_: float = 0.0

    @classmethod
    def from_params(cls, params: dict) -> "SklearnDecoder":
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

        calibration_start = perf_counter()
        model.fit(train_x, train_y)
        self.calibration_time_s_ = perf_counter() - calibration_start
        self.model = model
        self.decoder_name = decoder_name
        self.classes_ = [str(item) for item in model.classes_]
        self.feature_names_ = labeled[0].feature_names
        self.train_size_ = len(train_y)
        self.test_size_ = len(test_y)

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
            calibration_time_s=self.calibration_time_s_,
            model_card=self.model_card(),
        )

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("cannot save an unfitted decoder")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "estimator": self.estimator,
                    "decoder_name": self.decoder_name,
                    "model": self.model,
                    "classes": self.classes_,
                    "feature_names": self.feature_names_,
                    "calibration_fraction": self.calibration_fraction,
                    "calibration_time_s": self.calibration_time_s_,
                },
                handle,
            )

    @staticmethod
    def load(path: Path) -> "SklearnDecoder":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        decoder = SklearnDecoder(
            estimator=payload["estimator"],
            calibration_fraction=float(payload["calibration_fraction"]),
        )
        decoder.model = payload["model"]
        decoder.decoder_name = payload["decoder_name"]
        decoder.classes_ = list(payload["classes"])
        decoder.feature_names_ = tuple(payload["feature_names"])
        decoder.calibration_time_s_ = float(payload["calibration_time_s"])
        return decoder

    def model_card(self) -> dict[str, Any]:
        return {
            "decoder_name": self.decoder_name,
            "adapter": "sklearn",
            "requested_estimator": self.estimator,
            "classes": self.classes_,
            "feature_count": len(self.feature_names_),
            "feature_names": list(self.feature_names_),
            "train_size": self.train_size_,
            "test_size": self.test_size_,
            "calibration_fraction": self.calibration_fraction,
            "calibration_time_s": self.calibration_time_s_,
        }

    def _build_model(self):
        if self.estimator == "lda":
            try:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

                return LinearDiscriminantAnalysis(), "sklearn_lda"
            except Exception:
                return _NearestCentroid(), "nearest_centroid_fallback"
        if self.estimator in {"logistic", "logistic_regression", "lr"}:
            try:
                from sklearn.linear_model import LogisticRegression

                return LogisticRegression(max_iter=1000), "sklearn_logistic_regression"
            except Exception:
                return _NearestCentroid(), "nearest_centroid_fallback"
        if self.estimator in {"nearest_centroid", "centroid"}:
            return _NearestCentroid(), "nearest_centroid"
        raise ValueError(f"unsupported estimator: {self.estimator}")

