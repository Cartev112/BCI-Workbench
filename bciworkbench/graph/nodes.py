from __future__ import annotations

from typing import Any

from bciworkbench.decoders.simple import SupervisedDecoder
from bciworkbench.graph.context import RunContext
from bciworkbench.graph.node import Node
from bciworkbench.ontology.packets import FeaturePacket, SignalPacket, WindowPacket
from bciworkbench.sources.synthetic import SyntheticMotorImagerySource
from bciworkbench.transforms.features import BandpowerTransform
from bciworkbench.transforms.windowing import TrialWindowTransform


class SyntheticMotorImagerySourceNode(Node):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__("source.synthetic_motor_imagery", "source", params)

    def process(self, payload: Any, context: RunContext) -> SignalPacket:
        if payload is not None:
            raise ValueError("source node expected no input payload")
        packet = SyntheticMotorImagerySource.from_params(self.params, seed=context.spec.random_seed).read()
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


class BandpowerNode(Node):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__("transform.bandpower", "transform", params)

    def process(self, payload: list[WindowPacket], context: RunContext) -> list[FeaturePacket]:
        signal = context.artifacts.get("signal")
        if signal is None:
            raise ValueError("BandpowerNode requires signal artifact for sampling rate")
        features = BandpowerTransform.from_params(self.params).transform(
            payload,
            sampling_rate=signal.channel_schema.sampling_rate,
        )
        context.artifacts["features"] = features
        return features


class DecoderNode(Node):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__("decoder.supervised", "decoder", params)

    def process(self, payload: list[FeaturePacket], context: RunContext):
        result = SupervisedDecoder.from_params(self.params).fit_predict(payload)
        context.artifacts["decoder_result"] = result
        context.artifacts["predictions"] = result.predictions
        return result

