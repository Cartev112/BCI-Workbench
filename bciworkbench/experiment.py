from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bciworkbench.decoders.base import DecoderResult
from bciworkbench.eval.metrics import decoder_metrics
from bciworkbench.graph.context import RunContext
from bciworkbench.graph.nodes import BandpowerNode, DecoderNode, SourceNode, TrialWindowNode
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
    write_predictions,
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

        write_json(run_dir / "resolved_config.json", spec.to_dict())
        write_json(run_dir / "ontology_schema.json", ontology_json_schema())
        write_json(run_dir / "graph.json", runtime.describe_graph())
        write_json(run_dir / "channel_schema.json", signal.channel_schema.to_dict())
        write_json(run_dir / "source_metadata.json", signal.metadata)
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "model" / "model_card.json", decoder.model_card)
        write_json(run_dir / "provenance.json", provenance(spec))
        write_jsonl(run_dir / "telemetry.jsonl", [record.to_dict() for record in runtime.telemetry])
        write_events(run_dir / "events.csv", signal.events)
        write_windows(run_dir / "windows.csv", windows)
        write_features(run_dir / "features.csv", features)
        write_predictions(run_dir / "predictions.csv", decoder.predictions)
        write_html_report(run_dir / "report.html", spec, metrics, decoder)

        return RunResult(run_id=run_id, run_dir=run_dir, metrics=metrics, decoder=decoder)

    def _build_runtime(self) -> LinearRuntime:
        spec = self.spec
        return LinearRuntime(
            [
                SourceNode(spec.source.type, spec.source.params),
                TrialWindowNode(spec.pipeline[0].params),
                BandpowerNode(spec.pipeline[1].params),
                DecoderNode(spec.pipeline[2].params),
            ]
        )

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
