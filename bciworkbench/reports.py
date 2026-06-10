from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

from bciworkbench.decoders.simple import DecoderResult
from bciworkbench.ontology.packets import Event, FeaturePacket, IntentPacket, WindowPacket
from bciworkbench.ontology.schemas import ExperimentSpec


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_events(path: Path, events: list[Event]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_id",
                "event_type",
                "name",
                "onset",
                "duration",
                "clock_domain",
                "sample_index",
                "target",
                "source",
            ],
        )
        writer.writeheader()
        for event in events:
            row = event.to_dict()
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


def write_windows(path: Path, windows: list[WindowPacket]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window_id", "start_time", "end_time", "sample_start", "sample_end", "label"],
        )
        writer.writeheader()
        for window in windows:
            writer.writerow(
                {
                    "window_id": window.window_id,
                    "start_time": window.start_time,
                    "end_time": window.end_time,
                    "sample_start": window.sample_start,
                    "sample_end": window.sample_end,
                    "label": window.label,
                }
            )


def write_features(path: Path, features: list[FeaturePacket]) -> None:
    if not features:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = ["feature_id", "window_id", "label", *features[0].feature_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature in features:
            row = {"feature_id": feature.feature_id, "window_id": feature.window_id, "label": feature.label}
            row.update({name: value for name, value in zip(feature.feature_names, feature.features, strict=True)})
            writer.writerow(row)


def write_predictions(path: Path, predictions: list[IntentPacket]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["intent_id", "window_id", "label", "intent", "confidence", "latency_ms", "decoder_id"],
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "intent_id": prediction.intent_id,
                    "window_id": prediction.window_id,
                    "label": prediction.label,
                    "intent": prediction.intent,
                    "confidence": prediction.confidence,
                    "latency_ms": prediction.latency_ms,
                    "decoder_id": prediction.decoder_id,
                }
            )


def provenance(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "experiment_name": spec.name,
        "schema_version": spec.schema_version,
        "python": sys.version,
        "platform": platform.platform(),
        "random_seed": spec.random_seed,
    }


def write_html_report(path: Path, spec: ExperimentSpec, metrics: dict[str, Any], decoder: DecoderResult) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BCI Workbench Report - {spec.name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.45; max-width: 960px; margin: 40px auto; padding: 0 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ text-align: left; background: #f4f4f4; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>{spec.name}</h1>
  <p><strong>Paradigm:</strong> {spec.paradigm}</p>
  <p><strong>Mode:</strong> {spec.mode}</p>
  <p><strong>Decoder:</strong> {decoder.decoder_name}</p>
  <h2>Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Accuracy</td><td>{metrics.get("accuracy")}</td></tr>
    <tr><td>Balanced accuracy</td><td>{metrics.get("balanced_accuracy")}</td></tr>
    <tr><td>Predictions</td><td>{metrics.get("n_predictions")}</td></tr>
    <tr><td>Mean confidence</td><td>{metrics.get("mean_confidence")}</td></tr>
    <tr><td>Mean decoder latency ms</td><td>{metrics.get("mean_decoder_latency_ms")}</td></tr>
  </table>
  <h2>Run Artifacts</h2>
  <p>See <code>metrics.json</code>, <code>events.csv</code>, <code>windows.csv</code>, <code>features.csv</code>, and <code>predictions.csv</code>.</p>
  <h2>Simulation Note</h2>
  <p>This milestone source is synthetic and intended for software plumbing and architecture testing. It is not a validated physiological model.</p>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")

