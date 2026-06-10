from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from bciworkbench.experiment import Experiment
from bciworkbench.ontology.schemas import ConfigError, ExperimentSpec, parse_experiment_spec


@dataclass(frozen=True)
class StressorPreset:
    name: str
    description: str
    source_overrides: dict[str, Any]


BUILTIN_PRESETS: dict[str, StressorPreset] = {
    "clean": StressorPreset(
        name="clean",
        description="Low-artifact reference condition.",
        source_overrides={
            "snr_db": 2,
            "subject": {"attention": 0.95, "fatigue_rate": 0.0},
            "session": {
                "amplitude_drift": 0.0,
                "electrode_shift_mm": 0.0,
                "blink_rate_per_min": 0,
                "muscle_noise": 0.0,
                "channel_dropout_probability": 0.0,
                "marker_jitter_ms": 0.0,
            },
        },
    ),
    "low_snr": StressorPreset(
        name="low_snr",
        description="Reduced class separability through lower SNR.",
        source_overrides={"snr_db": -12},
    ),
    "high_blink": StressorPreset(
        name="high_blink",
        description="Frequent high-amplitude blink contamination.",
        source_overrides={"session": {"blink_rate_per_min": 24, "blink_amplitude_uv": 140}},
    ),
    "muscle_noise": StressorPreset(
        name="muscle_noise",
        description="High-frequency EMG-like bursts.",
        source_overrides={"session": {"muscle_noise": 0.45}},
    ),
    "channel_dropout": StressorPreset(
        name="channel_dropout",
        description="Random channel dropout during the synthetic session.",
        source_overrides={"session": {"channel_dropout_probability": 0.2}},
    ),
    "session_drift": StressorPreset(
        name="session_drift",
        description="Amplitude drift and electrode shift across the session.",
        source_overrides={"session": {"amplitude_drift": 0.3, "electrode_shift_mm": 9.0, "spectral_drift_hz": 0.4, "spatial_covariance_drift": 0.4}},
    ),
    "electrode_shift": StressorPreset(
        name="electrode_shift",
        description="Spatial shift in class topographies without major amplitude drift.",
        source_overrides={"session": {"electrode_shift_mm": 12.0, "spatial_covariance_drift": 0.25}},
    ),
    "jittery_markers": StressorPreset(
        name="jittery_markers",
        description="Marker timestamp jitter that perturbs trial windows.",
        source_overrides={"session": {"marker_jitter_ms": 35.0}},
    ),
    "fatigue": StressorPreset(
        name="fatigue",
        description="Attention drop through trial-index-linked fatigue.",
        source_overrides={"subject": {"attention": 0.75, "fatigue_rate": 0.01}},
    ),
    "delayed_feedback": StressorPreset(
        name="delayed_feedback",
        description="Feedback delay stressor logged for closed-loop compatibility.",
        source_overrides={"session": {"feedback_delay_ms": 220.0}},
    ),
}


@dataclass(frozen=True)
class StressBenchSpec:
    name: str
    base_config: Path
    presets: tuple[str, ...]
    output_dir: Path
    repeats: int = 1
    seed_stride: int = 1


@dataclass(frozen=True)
class StressBenchResult:
    summary_dir: Path
    rows: list[dict[str, Any]]
    aggregates: list[dict[str, Any]]
    robustness: dict[str, Any]


def load_stressbench_spec(path: str | Path) -> StressBenchSpec:
    spec_path = Path(path)
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("stressbench config must be a mapping")
    name = _string(raw.get("name"), "name")
    base_config_raw = _string(raw.get("base_config"), "base_config")
    base_config = Path(base_config_raw)
    if not base_config.is_absolute():
        base_config = spec_path.parent / base_config
    presets_raw = raw.get("presets")
    if not isinstance(presets_raw, list) or not presets_raw:
        raise ConfigError("presets must be a non-empty list")
    presets = tuple(_string(item, "presets[]") for item in presets_raw)
    unknown = sorted(set(presets) - set(BUILTIN_PRESETS))
    if unknown:
        raise ConfigError(f"unknown stressbench preset(s): {', '.join(unknown)}")
    output_dir = Path(str(raw.get("output_dir", "runs")))
    repeats = int(raw.get("repeats", 1))
    if repeats <= 0:
        raise ConfigError("repeats must be positive")
    seed_stride = int(raw.get("seed_stride", 1))
    if seed_stride <= 0:
        raise ConfigError("seed_stride must be positive")
    return StressBenchSpec(
        name=name,
        base_config=base_config,
        presets=presets,
        output_dir=output_dir,
        repeats=repeats,
        seed_stride=seed_stride,
    )


def run_stressbench(path: str | Path) -> StressBenchResult:
    spec = load_stressbench_spec(path)
    base_raw = yaml.safe_load(spec.base_config.read_text(encoding="utf-8"))
    base_experiment = parse_experiment_spec(base_raw)
    summary_dir = spec.output_dir / _summary_id(spec.name)
    summary_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    for preset_name in spec.presets:
        preset = BUILTIN_PRESETS[preset_name]
        for repeat in range(spec.repeats):
            run_raw = _build_variant_raw(
                base_raw=base_raw,
                base_experiment=base_experiment,
                preset=preset,
                repeat=repeat,
                spec=spec,
            )
            experiment = Experiment(parse_experiment_spec(run_raw))
            result = experiment.run()
            row = {
                "preset": preset_name,
                "description": preset.description,
                "repeat": repeat,
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "accuracy": result.metrics.get("accuracy"),
                "balanced_accuracy": result.metrics.get("balanced_accuracy"),
                "mean_confidence": result.metrics.get("mean_confidence"),
                "mean_decoder_latency_ms": result.metrics.get("mean_decoder_latency_ms"),
                "n_predictions": result.metrics.get("n_predictions"),
                "n_events": result.metrics.get("n_events"),
            }
            rows.append(row)

    aggregates = aggregate_rows(rows)
    robustness = robustness_summary(aggregates)
    _write_summary(summary_dir, spec, rows, aggregates, robustness)
    return StressBenchResult(summary_dir=summary_dir, rows=rows, aggregates=aggregates, robustness=robustness)


def _build_variant_raw(
    base_raw: dict[str, Any],
    base_experiment: ExperimentSpec,
    preset: StressorPreset,
    repeat: int,
    spec: StressBenchSpec,
) -> dict[str, Any]:
    run_raw = deepcopy(base_raw)
    run_raw["name"] = f"{base_experiment.name}_{preset.name}_r{repeat}"
    run_raw["output_dir"] = str(spec.output_dir)
    run_raw["random_seed"] = base_experiment.random_seed + repeat * spec.seed_stride
    source = run_raw.setdefault("source", {})
    _deep_update(source, preset.source_overrides)
    metadata = run_raw.setdefault("metadata", {})
    metadata["stressbench"] = {
        "name": spec.name,
        "preset": preset.name,
        "description": preset.description,
        "repeat": repeat,
    }
    return run_raw


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated StressBench rows by preset."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["preset"]), []).append(row)

    clean_mean = _mean_metric(grouped.get("clean", []), "balanced_accuracy")
    aggregates: list[dict[str, Any]] = []
    for preset, preset_rows in grouped.items():
        accuracy_mean = _mean_metric(preset_rows, "accuracy")
        balanced_mean = _mean_metric(preset_rows, "balanced_accuracy")
        confidence_mean = _mean_metric(preset_rows, "mean_confidence")
        latency_mean = _mean_metric(preset_rows, "mean_decoder_latency_ms")
        delta_from_clean = None
        normalized_score = balanced_mean
        if clean_mean is not None and balanced_mean is not None:
            delta_from_clean = balanced_mean - clean_mean
            normalized_score = balanced_mean / clean_mean if clean_mean > 0 else balanced_mean
        aggregates.append(
            {
                "preset": preset,
                "description": preset_rows[0].get("description"),
                "runs": len(preset_rows),
                "accuracy_mean": accuracy_mean,
                "balanced_accuracy_mean": balanced_mean,
                "mean_confidence_mean": confidence_mean,
                "mean_decoder_latency_ms_mean": latency_mean,
                "delta_from_clean": delta_from_clean,
                "normalized_score": normalized_score,
            }
        )

    order = {name: index for index, name in enumerate(BUILTIN_PRESETS)}
    return sorted(aggregates, key=lambda row: order.get(str(row["preset"]), len(order)))


def robustness_summary(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in aggregates if row.get("balanced_accuracy_mean") is not None]
    stressed = [row for row in scored if row["preset"] != "clean"] or scored
    if not stressed:
        return {
            "robustness_score": None,
            "weakest_preset": None,
            "worst_balanced_accuracy": None,
            "largest_drop_from_clean": None,
        }
    robustness_score = sum(float(row["balanced_accuracy_mean"]) for row in stressed) / len(stressed)
    weakest = min(stressed, key=lambda row: float(row["balanced_accuracy_mean"]))
    drops = [row for row in stressed if row.get("delta_from_clean") is not None]
    largest_drop = min(drops, key=lambda row: float(row["delta_from_clean"])) if drops else None
    return {
        "robustness_score": robustness_score,
        "weakest_preset": weakest["preset"],
        "worst_balanced_accuracy": weakest["balanced_accuracy_mean"],
        "largest_drop_from_clean": largest_drop["delta_from_clean"] if largest_drop else None,
        "largest_drop_preset": largest_drop["preset"] if largest_drop else None,
    }


def _write_summary(
    summary_dir: Path,
    spec: StressBenchSpec,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    robustness: dict[str, Any],
) -> None:
    payload = {
        "name": spec.name,
        "base_config": str(spec.base_config),
        "presets": list(spec.presets),
        "repeats": spec.repeats,
        "rows": rows,
        "aggregates": aggregates,
        "robustness": robustness,
    }
    (summary_dir / "stressbench_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (summary_dir / "stressbench_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else ["preset", "repeat", "run_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with (summary_dir / "stressbench_aggregates.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(aggregates[0]) if aggregates else ["preset", "runs"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregates:
            writer.writerow(row)
    _write_html(summary_dir / "stressbench_report.html", spec, rows, aggregates, robustness)


def _write_html(
    path: Path,
    spec: StressBenchSpec,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    robustness: dict[str, Any],
) -> None:
    aggregate_rows = "\n".join(
        "<tr>"
        f"<td>{row['preset']}</td>"
        f"<td>{row['runs']}</td>"
        f"<td>{_format_metric(row['balanced_accuracy_mean'])}</td>"
        f"<td>{_format_metric(row['delta_from_clean'])}</td>"
        f"<td>{_format_metric(row['normalized_score'])}</td>"
        "</tr>"
        for row in aggregates
    )
    body_rows = "\n".join(
        "<tr>"
        f"<td>{row['preset']}</td>"
        f"<td>{row['repeat']}</td>"
        f"<td>{row['accuracy']}</td>"
        f"<td>{row['balanced_accuracy']}</td>"
        f"<td>{row['mean_confidence']}</td>"
        f"<td><code>{row['run_id']}</code></td>"
        "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>StressBench - {spec.name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.45; max-width: 1000px; margin: 40px auto; padding: 0 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ text-align: left; background: #f4f4f4; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>StressBench: {spec.name}</h1>
  <p><strong>Base config:</strong> <code>{spec.base_config}</code></p>
  <h2>Robustness Summary</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Robustness score</td><td>{_format_metric(robustness.get("robustness_score"))}</td></tr>
    <tr><td>Weakest preset</td><td>{robustness.get("weakest_preset")}</td></tr>
    <tr><td>Worst balanced accuracy</td><td>{_format_metric(robustness.get("worst_balanced_accuracy"))}</td></tr>
    <tr><td>Largest drop from clean</td><td>{_format_metric(robustness.get("largest_drop_from_clean"))}</td></tr>
    <tr><td>Largest drop preset</td><td>{robustness.get("largest_drop_preset")}</td></tr>
  </table>
  <h2>Preset Aggregates</h2>
  <table>
    <tr><th>Preset</th><th>Runs</th><th>Balanced Accuracy Mean</th><th>Delta From Clean</th><th>Normalized Score</th></tr>
    {aggregate_rows}
  </table>
  <h2>Runs</h2>
  <table>
    <tr><th>Preset</th><th>Repeat</th><th>Accuracy</th><th>Balanced Accuracy</th><th>Mean Confidence</th><th>Run ID</th></tr>
    {body_rows}
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _summary_id(name: str) -> str:
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in name.lower())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{safe_name}"


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return sum(values) / len(values) if values else None


def _format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
