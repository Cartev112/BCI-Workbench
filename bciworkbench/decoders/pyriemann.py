from __future__ import annotations

from time import perf_counter

import numpy as np

from bciworkbench.decoders.base import DecoderResult
from bciworkbench.ontology.packets import FeaturePacket, IntentPacket


class PyRiemannDecoder:
    """Optional pyRiemann MDM adapter.

    This adapter expects each FeaturePacket to contain a flattened square
    covariance matrix. It is intentionally optional and is not imported by the
    runtime unless requested by user code.
    """

    def __init__(self, calibration_fraction: float = 0.6, metric: str = "riemann") -> None:
        if not 0 < calibration_fraction < 1:
            raise ValueError("calibration_fraction must be between 0 and 1")
        self.calibration_fraction = calibration_fraction
        self.metric = metric
        self.decoder_name = "pyriemann_mdm"
        self.model = None
        self.calibration_time_s_ = 0.0

    @classmethod
    def from_params(cls, params: dict) -> "PyRiemannDecoder":
        return cls(
            calibration_fraction=float(params.get("calibration_fraction", 0.6)),
            metric=str(params.get("metric", "riemann")),
        )

    def fit_predict(self, features: list[FeaturePacket]) -> DecoderResult:
        try:
            from pyriemann.classification import MDM
        except Exception as exc:
            raise ImportError(
                "PyRiemannDecoder requires the optional pyriemann dependency. "
                'Install with: pip install "bciworkbench[pyriemann]"'
            ) from exc

        labeled = [item for item in features if item.label is not None]
        if len(labeled) < 4:
            raise ValueError("at least 4 labeled feature packets are required")
        x = np.stack([_feature_to_covariance(item) for item in labeled])
        y = np.asarray([item.label for item in labeled])
        split = max(2, min(len(labeled) - 1, int(round(len(labeled) * self.calibration_fraction))))
        train_x, test_x = x[:split], x[split:]
        train_y, test_y = y[:split], y[split:]

        model = MDM(metric=self.metric)
        calibration_start = perf_counter()
        model.fit(train_x, train_y)
        self.calibration_time_s_ = perf_counter() - calibration_start
        self.model = model

        start = perf_counter()
        pred_y = model.predict(test_x)
        elapsed_ms = (perf_counter() - start) * 1000.0
        per_prediction_ms = elapsed_ms / max(len(test_y), 1)

        predictions = [
            IntentPacket(
                intent_id=f"prediction-{index:04d}",
                intent=str(pred),
                confidence=1.0,
                posterior={str(pred): 1.0},
                latency_ms=per_prediction_ms,
                window_id=feature.window_id,
                decoder_id=self.decoder_name,
                label=str(label),
            )
            for index, (feature, pred, label) in enumerate(zip(labeled[split:], pred_y, test_y, strict=True))
        ]
        return DecoderResult(
            predictions=predictions,
            train_size=len(train_y),
            test_size=len(test_y),
            decoder_name=self.decoder_name,
            calibration_time_s=self.calibration_time_s_,
            model_card=self.model_card(),
        )

    def model_card(self) -> dict:
        return {
            "decoder_name": self.decoder_name,
            "adapter": "pyriemann",
            "metric": self.metric,
            "calibration_fraction": self.calibration_fraction,
            "calibration_time_s": self.calibration_time_s_,
        }


def _feature_to_covariance(feature: FeaturePacket) -> np.ndarray:
    size = int(np.sqrt(feature.features.shape[0]))
    if size * size != feature.features.shape[0]:
        raise ValueError("PyRiemannDecoder requires flattened square covariance features")
    covariance = feature.features.reshape(size, size)
    return (covariance + covariance.T) / 2.0

