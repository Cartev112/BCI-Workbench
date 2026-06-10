from __future__ import annotations

from typing import Any

from bciworkbench.decoders.sklearn import SklearnDecoder
from bciworkbench.graph.context import RunContext
from bciworkbench.graph.node import Node
from bciworkbench.ontology.packets import FeaturePacket, SignalPacket, WindowPacket
from bciworkbench.sources.factory import build_source
from bciworkbench.transforms.features import BandpowerTransform, ERPFeatureTransform
from bciworkbench.transforms.windowing import TrialWindowTransform


class SourceNode(Node):
    def __init__(self, source_type: str, params: dict[str, Any]) -> None:
        super().__init__(f"source.{source_type}", "source", params)
        self.source_type = source_type

    def process(self, payload: Any, context: RunContext) -> SignalPacket:
        if payload is not None:
            raise ValueError("source node expected no input payload")
        packet = build_source(context.spec.source, seed=context.spec.random_seed).read()
        context.artifacts["signal"] = packet
        return packet


class TrialWindowNode(Node):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__("transform.window", "transform", params)

    def process(self, payload: SignalPacket, context: RunContext) -> list[WindowPacket]:
        if not isinstance(payload, SignalPacket):
            raise TypeError("TrialWindowNode expected a SignalPacket")
        windows = TrialWindowTransform.from_params(self.params).transform(payload)
        context.artifacts["windows"] = windows
        return windows


class FeatureNode(Node):
    def __init__(self, step_type: str, params: dict[str, Any]) -> None:
        super().__init__(f"transform.{step_type}", "transform", params)
        self.step_type = step_type

    def process(self, payload: list[WindowPacket], context: RunContext) -> list[FeaturePacket]:
        signal = context.artifacts.get("signal")
        if signal is None:
            raise ValueError("FeatureNode requires signal artifact for sampling rate")
        transform = _feature_transform(self.step_type, self.params)
        features = transform.transform(
            payload,
            sampling_rate=signal.channel_schema.sampling_rate,
        )
        context.artifacts["features"] = features
        return features


class DecoderNode(Node):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__("decoder.supervised", "decoder", params)

    def process(self, payload: list[FeaturePacket], context: RunContext):
        decoder = SklearnDecoder.from_params(self.params)
        result = decoder.fit_predict(payload)
        model_path = context.run_dir / "model" / "decoder.pkl"
        decoder.save(model_path)
        result = type(result)(
            predictions=result.predictions,
            train_size=result.train_size,
            test_size=result.test_size,
            decoder_name=result.decoder_name,
            calibration_time_s=result.calibration_time_s,
            model_card={**decoder.model_card(), "model_path": str(model_path)},
            model_path=model_path,
        )
        context.artifacts["decoder_result"] = result
        context.artifacts["predictions"] = result.predictions
        return result


def _feature_transform(step_type: str, params: dict[str, Any]):
    if step_type == "bandpower":
        return BandpowerTransform.from_params(params)
    if step_type == "erp_features":
        return ERPFeatureTransform.from_params(params)
    raise ValueError(f"unsupported feature transform: {step_type}")
