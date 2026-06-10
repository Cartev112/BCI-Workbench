from __future__ import annotations

import csv
import html
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bciworkbench.decoders.base import DecoderResult
from bciworkbench.ontology.packets import Event, FeaturePacket, FeedbackPacket, IntentPacket, TaskStatePacket, WindowPacket
from bciworkbench.ontology.schemas import ExperimentSpec


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


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


def write_task_states(path: Path, states: list[TaskStatePacket]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "step_index",
                "target",
                "position",
                "target_position",
                "reward",
                "done",
                "success",
                "metadata",
            ],
        )
        writer.writeheader()
        for state in states:
            writer.writerow(
                {
                    "task_id": state.task_id,
                    "step_index": state.state.get("step_index"),
                    "target": state.target,
                    "position": state.state.get("position"),
                    "target_position": state.state.get("target_position"),
                    "reward": state.reward,
                    "done": state.done,
                    "success": state.success,
                    "metadata": json.dumps(state.metadata, sort_keys=True),
                }
            )


def write_task_feedback(path: Path, feedback: list[FeedbackPacket]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["action", "rendered_at", "clock_domain", "reward", "delay_ms", "task_id", "metadata"],
        )
        writer.writeheader()
        for packet in feedback:
            writer.writerow(
                {
                    "action": packet.action,
                    "rendered_at": packet.rendered_at,
                    "clock_domain": packet.clock_domain,
                    "reward": packet.reward,
                    "delay_ms": packet.delay_ms,
                    "task_id": packet.task_state.task_id if packet.task_state else None,
                    "metadata": json.dumps(packet.metadata, sort_keys=True),
                }
            )


def write_latency_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "packet_index",
        "sample_start",
        "sample_end",
        "signal_time_start_s",
        "signal_time_end_s",
        "scheduled_arrival_s",
        "arrival_time_s",
        "arrival_delay_ms",
        "backlog_ms",
        "queue_depth",
        "dropped",
        "sleep_s",
        "speed_mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def provenance(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "experiment_name": spec.name,
        "schema_version": spec.schema_version,
        "python": sys.version,
        "platform": platform.platform(),
        "random_seed": spec.random_seed,
    }


def write_html_report(path: Path, spec: ExperimentSpec, metrics: dict[str, Any], decoder: DecoderResult) -> None:
    run_dir = path.parent
    source_metadata = _read_json_if_exists(run_dir / "source_metadata.json")
    stream_health = _read_json_if_exists(run_dir / "stream_health.json")
    task_metrics = _read_json_if_exists(run_dir / "task_metrics.json")
    model_card = _read_json_if_exists(run_dir / "model" / "model_card.json") or decoder.model_card
    graph = _read_json_if_exists(run_dir / "graph.json")
    provenance_payload = _read_json_if_exists(run_dir / "provenance.json")
    telemetry = _read_jsonl_if_exists(run_dir / "telemetry.jsonl")
    latency = _telemetry_summary(telemetry)
    warnings = _run_warnings(source_metadata, metrics)
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
  <h1>{_esc(spec.name)}</h1>
  <p><strong>Paradigm:</strong> {_esc(spec.paradigm)}</p>
  <p><strong>Mode:</strong> {_esc(spec.mode)}</p>
  <p><strong>Decoder:</strong> {_esc(decoder.decoder_name)}</p>
  <h2>Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Accuracy</td><td>{_fmt(metrics.get("accuracy"))}</td></tr>
    <tr><td>Balanced accuracy</td><td>{_fmt(metrics.get("balanced_accuracy"))}</td></tr>
    <tr><td>Predictions</td><td>{_fmt(metrics.get("n_predictions"))}</td></tr>
    <tr><td>Mean confidence</td><td>{_fmt(metrics.get("mean_confidence"))}</td></tr>
    <tr><td>Calibration time s</td><td>{_fmt(metrics.get("calibration_time_s"))}</td></tr>
    <tr><td>Mean decoder latency ms</td><td>{_fmt(metrics.get("mean_decoder_latency_ms"))}</td></tr>
  </table>
  <h2>Source</h2>
  {_dict_table(source_metadata)}
  <h2>Replay Stream Health</h2>
  {_dict_table(stream_health)}
  <h2>Closed Loop Task</h2>
  {_dict_table(task_metrics)}
  <h2>Model Card</h2>
  {_dict_table(model_card)}
  <h2>Latency And Runtime</h2>
  {_dict_table(latency)}
  <h2>Graph</h2>
  {_graph_table(graph)}
  <h2>Warnings</h2>
  {_warnings_list(warnings)}
  <h2>Provenance</h2>
  {_dict_table(provenance_payload)}
  <h2>Run Artifacts</h2>
  <p>See <code>metrics.json</code>, <code>task_metrics.json</code>, <code>task_states.csv</code>, <code>feedback.csv</code>, <code>model/model_card.json</code>, <code>model/decoder.pkl</code>, <code>graph.json</code>, <code>telemetry.jsonl</code>, <code>source_metadata.json</code>, <code>stream_health.json</code>, <code>latency_trace.csv</code>, <code>events.csv</code>, <code>windows.csv</code>, <code>features.csv</code>, and <code>predictions.csv</code>.</p>
  <h2>Simulation Note</h2>
  <p>This milestone source is synthetic and intended for software plumbing and architecture testing. It is not a validated physiological model.</p>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_metadata = _read_json_if_exists(run_path / "source_metadata.json")
    stream_health = _read_json_if_exists(run_path / "stream_health.json")
    task_metrics = _read_json_if_exists(run_path / "task_metrics.json")
    provenance_payload = _read_json_if_exists(run_path / "provenance.json")
    model_card = _read_json_if_exists(run_path / "model" / "model_card.json")
    telemetry = _read_jsonl_if_exists(run_path / "telemetry.jsonl")
    latency = _telemetry_summary(telemetry)
    return {
        "run_id": run_path.name,
        "run_dir": str(run_path),
        "experiment_name": provenance_payload.get("experiment_name"),
        "source": metrics.get("source"),
        "source_format": source_metadata.get("source_format"),
        "decoder": metrics.get("decoder"),
        "requested_estimator": model_card.get("requested_estimator"),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "mean_confidence": metrics.get("mean_confidence"),
        "calibration_time_s": metrics.get("calibration_time_s"),
        "mean_decoder_latency_ms": metrics.get("mean_decoder_latency_ms"),
        "runtime_total_duration_ms": latency.get("runtime_total_duration_ms"),
        "runtime_slowest_node": latency.get("runtime_slowest_node"),
        "replay_packet_count": stream_health.get("packet_count"),
        "replay_max_backlog_ms": stream_health.get("max_backlog_ms"),
        "replay_max_queue_depth": stream_health.get("max_queue_depth"),
        "target_acquisition_rate": task_metrics.get("target_acquisition_rate"),
        "mean_time_to_target_s": task_metrics.get("mean_time_to_target_s"),
        "decoder_task_gap": metrics.get("decoder_task_gap"),
        "n_events": metrics.get("n_events"),
        "n_windows": metrics.get("n_windows"),
        "n_predictions": metrics.get("n_predictions"),
    }


def compare_runs(run_dirs: list[str | Path], output_dir: str | Path = "runs") -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("compare requires at least one run directory")
    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    comparison_dir = Path(output_dir) / _comparison_id()
    comparison_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(rows),
        "best_run": _best_row(rows),
        "rows": rows,
    }
    write_json(comparison_dir / "comparison_summary.json", payload)
    _write_compare_csv(comparison_dir / "comparison_summary.csv", rows)
    _write_compare_html(comparison_dir / "comparison_report.html", payload)
    return {"comparison_dir": comparison_dir, **payload}


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _telemetry_summary(telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    if not telemetry:
        return {"runtime_total_duration_ms": None, "runtime_slowest_node": None, "runtime_node_count": 0}
    total = sum(float(row.get("duration_ms") or 0.0) for row in telemetry)
    slowest = max(telemetry, key=lambda row: float(row.get("duration_ms") or 0.0))
    return {
        "runtime_total_duration_ms": total,
        "runtime_slowest_node": slowest.get("node_id"),
        "runtime_slowest_node_ms": slowest.get("duration_ms"),
        "runtime_node_count": len(telemetry),
        "runtime_error_count": sum(1 for row in telemetry if row.get("status") != "ok"),
    }


def _run_warnings(source_metadata: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if source_metadata.get("simulation_level"):
        warnings.append(f"Simulation level: {source_metadata['simulation_level']}")
    if metrics.get("balanced_accuracy") is not None and float(metrics["balanced_accuracy"]) < 0.75:
        warnings.append("Balanced accuracy is below 0.75.")
    if source_metadata.get("bad_channels"):
        warnings.append(f"Bad channels present: {', '.join(source_metadata['bad_channels'])}")
    stream_health = source_metadata.get("stream_health") or {}
    if stream_health.get("dropped_packets"):
        warnings.append(f"Replay dropped packets: {stream_health['dropped_packets']}")
    if metrics.get("decoder_task_gap") is not None and float(metrics["decoder_task_gap"]) > 0.25:
        warnings.append("Decoder accuracy is materially higher than closed-loop target acquisition.")
    return warnings


def _dict_table(payload: dict[str, Any]) -> str:
    if not payload:
        return "<p>No artifact found.</p>"
    rows = "\n".join(
        f"<tr><td>{_esc(str(key))}</td><td>{_esc(_compact(value))}</td></tr>"
        for key, value in sorted(payload.items())
    )
    return f"<table><tr><th>Field</th><th>Value</th></tr>{rows}</table>"


def _graph_table(graph: dict[str, Any]) -> str:
    nodes = graph.get("nodes") if graph else None
    if not nodes:
        return "<p>No graph artifact found.</p>"
    rows = "\n".join(
        f"<tr><td>{_esc(node.get('node_id'))}</td><td>{_esc(node.get('node_type'))}</td><td>{_esc(_compact(node.get('params')))}</td></tr>"
        for node in nodes
    )
    return f"<table><tr><th>Node</th><th>Type</th><th>Params</th></tr>{rows}</table>"


def _warnings_list(warnings: list[str]) -> str:
    if not warnings:
        return "<p>No warnings.</p>"
    return "<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in warnings) + "</ul>"


def _write_compare_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_compare_html(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    body_rows = "\n".join(
        "<tr>"
        f"<td>{_esc(row['run_id'])}</td>"
        f"<td>{_esc(row.get('experiment_name'))}</td>"
        f"<td>{_esc(row.get('source'))}</td>"
        f"<td>{_esc(row.get('decoder'))}</td>"
        f"<td>{_fmt(row.get('accuracy'))}</td>"
        f"<td>{_fmt(row.get('balanced_accuracy'))}</td>"
        f"<td>{_fmt(row.get('runtime_total_duration_ms'))}</td>"
        "</tr>"
        for row in rows
    )
    best = payload.get("best_run") or {}
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BCI Workbench Run Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.45; max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ text-align: left; background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>Run Comparison</h1>
  <p><strong>Runs:</strong> {payload['run_count']}</p>
  <p><strong>Best run by balanced accuracy:</strong> {_esc(best.get('run_id'))} ({_fmt(best.get('balanced_accuracy'))})</p>
  <table>
    <tr><th>Run</th><th>Experiment</th><th>Source</th><th>Decoder</th><th>Accuracy</th><th>Balanced Accuracy</th><th>Total Runtime ms</th></tr>
    {body_rows}
  </table>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [row for row in rows if row.get("balanced_accuracy") is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: float(row["balanced_accuracy"]))


def _comparison_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-comparison")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return _esc(str(value))


def _compact(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))
