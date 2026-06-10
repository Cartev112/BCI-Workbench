from __future__ import annotations

from typing import Any

from bciworkbench.adaptation.factory import build_adaptation_adapter
from bciworkbench.decoders.base import DecoderResult
from bciworkbench.decoders.pyriemann import PyRiemannDecoder
from bciworkbench.decoders.sklearn import SklearnDecoder
from bciworkbench.graph.context import RunContext
from bciworkbench.graph.node import Node
from bciworkbench.ontology.packets import FeaturePacket, SignalPacket, WindowPacket
from bciworkbench.sources.factory import build_source
from bciworkbench.tasks.cursor import run_cursor_task
from bciworkbench.transforms.features import BandpowerTransform, CovarianceFeatureTransform, ERPFeatureTransform
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
        decoder = _decoder_adapter(self.params)
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


class AdaptationNode(Node):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(f"adaptation.{params.get('type', 'none')}", "adaptation", params)

    def process(self, payload: DecoderResult, context: RunContext) -> DecoderResult:
        if not isinstance(payload, DecoderResult):
            raise TypeError("AdaptationNode expected a DecoderResult")
        result = build_adaptation_adapter(self.params).adapt(payload.predictions)
        adapted_decoder = type(payload)(
            predictions=result.predictions,
            train_size=payload.train_size,
            test_size=payload.test_size,
            decoder_name=payload.decoder_name,
            calibration_time_s=payload.calibration_time_s,
            model_card={**payload.model_card, "adaptation": self.params},
            model_path=payload.model_path,
        )
        context.artifacts["adaptation_packets"] = result.packets
        context.artifacts["adaptation_metrics"] = result.metrics
        context.artifacts["predictions_before_adaptation"] = payload.predictions
        context.artifacts["predictions"] = result.predictions
        context.artifacts["decoder_result"] = adapted_decoder
        return adapted_decoder


class TaskNode(Node):
    def __init__(self, task_type: str, params: dict[str, Any]) -> None:
        super().__init__(f"task.{task_type}", "task", params)
        self.task_type = task_type

    def process(self, payload: DecoderResult, context: RunContext) -> DecoderResult:
        if not isinstance(payload, DecoderResult):
            raise TypeError("TaskNode expected a DecoderResult")
        if self.task_type != "cursor_1d":
            return payload
        result = run_cursor_task(payload.predictions, self.params)
        context.artifacts["task_states"] = result.states
        context.artifacts["feedback"] = result.feedback
        context.artifacts["task_metrics"] = result.metrics
        context.artifacts["task_rows"] = result.rows
        return payload


def _feature_transform(step_type: str, params: dict[str, Any]):
    if step_type == "bandpower":
        return BandpowerTransform.from_params(params)
    if step_type == "erp_features":
        return ERPFeatureTransform.from_params(params)
    if step_type == "covariance":
        return CovarianceFeatureTransform.from_params(params)
    raise ValueError(f"unsupported feature transform: {step_type}")


def _decoder_adapter(params: dict[str, Any]):
    adapter = str(params.get("adapter", "sklearn"))
    estimator = str(params.get("estimator", "lda"))
    if adapter == "pyriemann" or estimator in {"mdm", "pyriemann_mdm"}:
        return PyRiemannDecoder.from_params(params)
    if adapter == "sklearn":
        return SklearnDecoder.from_params(params)
    raise ValueError(f"unsupported decoder adapter: {adapter}")
