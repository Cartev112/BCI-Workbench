from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bciworkbench.decoders.simple import DecoderResult, SupervisedDecoder
from bciworkbench.eval.metrics import decoder_metrics
from bciworkbench.ontology.schemas import ExperimentSpec, load_experiment_spec
from bciworkbench.reports import (
    provenance,
    write_events,
    write_features,
    write_html_report,
    write_json,
    write_predictions,
    write_windows,
)
from bciworkbench.sources.synthetic import SyntheticMotorImagerySource
from bciworkbench.transforms.features import BandpowerTransform
from bciworkbench.transforms.windowing import TrialWindowTransform


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
        run_id = self._run_id(spec.name)
        run_dir = Path(spec.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        source = SyntheticMotorImagerySource.from_params(spec.source.params, seed=spec.random_seed)
        signal = source.read()

        window_step = spec.pipeline[0]
        feature_step = spec.pipeline[1]
        decoder_step = spec.pipeline[2]
        windows = TrialWindowTransform.from_params(window_step.params).transform(signal)
        features = BandpowerTransform.from_params(feature_step.params).transform(
            windows,
            sampling_rate=signal.channel_schema.sampling_rate,
        )
        decoder = SupervisedDecoder.from_params(decoder_step.params).fit_predict(features)
        metrics = decoder_metrics(decoder.predictions)
        metrics.update(
            {
                "train_size": decoder.train_size,
                "test_size": decoder.test_size,
                "decoder": decoder.decoder_name,
                "n_events": len(signal.events),
                "n_windows": len(windows),
                "n_features": len(features),
                "source": signal.source_id,
            }
        )

        write_json(run_dir / "resolved_config.json", spec.to_dict())
        write_json(run_dir / "channel_schema.json", signal.channel_schema.to_dict())
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "provenance.json", provenance(spec))
        write_events(run_dir / "events.csv", signal.events)
        write_windows(run_dir / "windows.csv", windows)
        write_features(run_dir / "features.csv", features)
        write_predictions(run_dir / "predictions.csv", decoder.predictions)
        write_html_report(run_dir / "report.html", spec, metrics, decoder)

        return RunResult(run_id=run_id, run_dir=run_dir, metrics=metrics, decoder=decoder)

    @staticmethod
    def _run_id(name: str) -> str:
        safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in name.lower())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = f"{timestamp}-{safe_name}"
        if not Path("runs", candidate).exists():
            return candidate
        suffix = 1
        while Path("runs", f"{candidate}-{suffix}").exists():
            suffix += 1
        return f"{candidate}-{suffix}"

    @staticmethod
    def clean_runs() -> None:
        shutil.rmtree("runs", ignore_errors=True)

