from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bciworkbench.decoders.base import DecoderResult
from bciworkbench.eval.metrics import decoder_metrics
from bciworkbench.graph.context import RunContext
from bciworkbench.graph.nodes import DecoderNode, FeatureNode, SourceNode, TaskNode, TrialWindowNode
from bciworkbench.graph.runtime import LinearRuntime
from bciworkbench.ontology.schemas import ExperimentSpec, load_experiment_spec
from bciworkbench.ontology.schema_export import ontology_json_schema
from bciworkbench.reports import (
    provenance,
    write_events,
    write_features,
    write_html_report,
    write_json,
    write_jsonl,
    write_latency_trace,
    write_predictions,
    write_task_feedback,
    write_task_states,
    write_windows,
)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    metrics: dict
    decoder: DecoderResult


class Experiment:
    def __init__(self, spec: ExperimentSpec) -> None:
        self.spec = spec

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Experiment":
        return cls(load_experiment_spec(path))

    def run(self) -> RunResult:
        spec = self.spec
        run_id = self._run_id(spec.name, Path(spec.output_dir))
        run_dir = Path(spec.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        context = RunContext(
            run_id=run_id,
            run_dir=run_dir,
            spec=spec,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        runtime = self._build_runtime()
        decoder = runtime.run(context)
        if not isinstance(decoder, DecoderResult):
            raise TypeError("runtime did not produce a DecoderResult")

        signal = context.artifacts["signal"]
        windows = context.artifacts["windows"]
        features = context.artifacts["features"]
        metrics = decoder_metrics(decoder.predictions)
        metrics.update(
            {
                "train_size": decoder.train_size,
                "test_size": decoder.test_size,
                "decoder": decoder.decoder_name,
                "calibration_time_s": decoder.calibration_time_s,
                "n_events": len(signal.events),
                "n_windows": len(windows),
                "n_features": len(features),
                "source": signal.source_id,
            }
        )
        task_metrics = context.artifacts.get("task_metrics")
        if task_metrics:
            metrics.update(task_metrics)
            if metrics.get("balanced_accuracy") is not None and metrics.get("target_acquisition_rate") is not None:
                metrics["decoder_task_gap"] = float(metrics["balanced_accuracy"]) - float(metrics["target_acquisition_rate"])

        write_json(run_dir / "resolved_config.json", spec.to_dict())
        write_json(run_dir / "ontology_schema.json", ontology_json_schema())
        write_json(run_dir / "graph.json", runtime.describe_graph())
        write_json(run_dir / "channel_schema.json", signal.channel_schema.to_dict())
        source_metadata = dict(signal.metadata)
        latency_trace = source_metadata.pop("latency_trace", None)
        if latency_trace:
            source_metadata["latency_trace_artifact"] = "latency_trace.csv"
            write_json(run_dir / "latency_trace.json", {"rows": latency_trace})
            write_latency_trace(run_dir / "latency_trace.csv", latency_trace)
        if signal.metadata.get("stream_health"):
            write_json(run_dir / "stream_health.json", signal.metadata["stream_health"])
        write_json(run_dir / "source_metadata.json", source_metadata)
        write_json(run_dir / "metrics.json", metrics)
        if task_metrics:
            write_json(run_dir / "task_metrics.json", task_metrics)
        write_json(run_dir / "model" / "model_card.json", decoder.model_card)
        write_json(run_dir / "provenance.json", provenance(spec))
        write_jsonl(run_dir / "telemetry.jsonl", [record.to_dict() for record in runtime.telemetry])
        write_events(run_dir / "events.csv", signal.events)
        write_windows(run_dir / "windows.csv", windows)
        write_features(run_dir / "features.csv", features)
        write_predictions(run_dir / "predictions.csv", decoder.predictions)
        if context.artifacts.get("task_states"):
            write_task_states(run_dir / "task_states.csv", context.artifacts["task_states"])
        if context.artifacts.get("feedback"):
            write_task_feedback(run_dir / "feedback.csv", context.artifacts["feedback"])
        write_html_report(run_dir / "report.html", spec, metrics, decoder)

        return RunResult(run_id=run_id, run_dir=run_dir, metrics=metrics, decoder=decoder)

    def _build_runtime(self) -> LinearRuntime:
        spec = self.spec
        nodes = [
            SourceNode(spec.source.type, spec.source.params),
            TrialWindowNode(spec.pipeline[0].params),
            FeatureNode(spec.pipeline[1].type, spec.pipeline[1].params),
            DecoderNode(spec.pipeline[2].params),
        ]
        if spec.task.type == "cursor_1d":
            nodes.append(TaskNode(spec.task.type, spec.task.params))
        return LinearRuntime(nodes)

    @staticmethod
    def _run_id(name: str, output_dir: Path) -> str:
        safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in name.lower())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = f"{timestamp}-{safe_name}"
        if not (output_dir / candidate).exists():
            return candidate
        suffix = 1
        while (output_dir / f"{candidate}-{suffix}").exists():
            suffix += 1
        return f"{candidate}-{suffix}"

    @staticmethod
    def clean_runs() -> None:
        shutil.rmtree("runs", ignore_errors=True)
