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
from bciworkbench.stressbench_cards import BUILTIN_ARCHITECTURES, ArchitectureCard


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
        source_overrides={
            "session": {
                "amplitude_drift": 0.3,
                "electrode_shift_mm": 9.0,
                "spectral_drift_hz": 0.4,
                "spatial_covariance_drift": 0.4,
            }
        },
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
    base_config: Path | None
    presets: tuple[str, ...]
    output_dir: Path
    repeats: int = 1
    seed_stride: int = 1
    architectures: tuple[str, ...] = ()


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
    base_config = _optional_base_config(raw, spec_path)
    architectures = _architecture_names(raw.get("architectures", []))
    if base_config is None and not architectures:
        raise ConfigError("stressbench config requires base_config or architectures")

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
        architectures=architectures,
    )


def run_stressbench(path: str | Path) -> StressBenchResult:
    spec = load_stressbench_spec(path)
    architecture_cards = _architecture_cards(spec)
    summary_dir = spec.output_dir / _summary_id(spec.name)
    summary_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    for architecture in architecture_cards:
        base_raw = architecture.config_copy()
        base_experiment = parse_experiment_spec(base_raw)
        for preset_name in spec.presets:
            preset = BUILTIN_PRESETS[preset_name]
            for repeat in range(spec.repeats):
                run_raw = _build_variant_raw(
                    base_raw=base_raw,
                    base_experiment=base_experiment,
                    architecture=architecture,
                    preset=preset,
                    repeat=repeat,
                    spec=spec,
                )
                rows.append(_run_variant(run_raw, architecture, preset, preset_name, repeat))

    aggregates = aggregate_rows(rows)
    robustness = robustness_summary(aggregates)
    _write_summary(summary_dir, spec, architecture_cards, rows, aggregates, robustness)
    return StressBenchResult(summary_dir=summary_dir, rows=rows, aggregates=aggregates, robustness=robustness)


def _build_variant_raw(
    base_raw: dict[str, Any],
    base_experiment: ExperimentSpec,
    architecture: ArchitectureCard,
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
        "architecture": architecture.name,
        "preset": preset.name,
        "description": preset.description,
        "repeat": repeat,
    }
    return run_raw


def _run_variant(
    run_raw: dict[str, Any],
    architecture: ArchitectureCard,
    preset: StressorPreset,
    preset_name: str,
    repeat: int,
) -> dict[str, Any]:
    base_row = {
        "architecture": architecture.name,
        "architecture_description": architecture.description,
        "preset": preset_name,
        "description": preset.description,
        "repeat": repeat,
        "status": "ok",
        "run_id": None,
        "run_dir": None,
        "error": None,
        "accuracy": None,
        "balanced_accuracy": None,
        "mean_confidence": None,
        "mean_decoder_latency_ms": None,
        "calibration_time_s": None,
        "target_acquisition_rate": None,
        "adaptation_update_count": None,
        "n_predictions": None,
        "n_events": None,
    }
    try:
        result = Experiment(parse_experiment_spec(run_raw)).run()
    except Exception as exc:
        hint = architecture.expected_failure_hint
        return {
            **base_row,
            "status": "skipped" if architecture.optional_extra else "error",
            "error": f"{exc}" + (f" ({hint})" if hint else ""),
        }
    return {
        **base_row,
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "accuracy": result.metrics.get("accuracy"),
        "balanced_accuracy": result.metrics.get("balanced_accuracy"),
        "mean_confidence": result.metrics.get("mean_confidence"),
        "mean_decoder_latency_ms": result.metrics.get("mean_decoder_latency_ms"),
        "calibration_time_s": result.metrics.get("calibration_time_s"),
        "target_acquisition_rate": result.metrics.get("target_acquisition_rate"),
        "adaptation_update_count": result.metrics.get("adaptation_update_count"),
        "n_predictions": result.metrics.get("n_predictions"),
        "n_events": result.metrics.get("n_events"),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated StressBench rows by architecture and preset."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault((str(row["architecture"]), str(row["preset"])), []).append(row)

    clean_by_architecture = {
        architecture: _mean_metric(preset_rows, "balanced_accuracy")
        for (architecture, preset), preset_rows in grouped.items()
        if preset == "clean"
    }
    aggregates: list[dict[str, Any]] = []
    for (architecture, preset), preset_rows in grouped.items():
        accuracy_mean = _mean_metric(preset_rows, "accuracy")
        balanced_mean = _mean_metric(preset_rows, "balanced_accuracy")
        confidence_mean = _mean_metric(preset_rows, "mean_confidence")
        latency_mean = _mean_metric(preset_rows, "mean_decoder_latency_ms")
        calibration_mean = _mean_metric(preset_rows, "calibration_time_s")
        task_success_mean = _mean_metric(preset_rows, "target_acquisition_rate")
        adaptation_updates_mean = _mean_metric(preset_rows, "adaptation_update_count")
        clean_mean = clean_by_architecture.get(architecture)
        delta_from_clean = None
        normalized_score = balanced_mean
        if clean_mean is not None and balanced_mean is not None:
            delta_from_clean = balanced_mean - clean_mean
            normalized_score = balanced_mean / clean_mean if clean_mean > 0 else balanced_mean
        aggregates.append(
            {
                "architecture": architecture,
                "preset": preset,
                "description": preset_rows[0].get("description"),
                "runs": len(preset_rows),
                "accuracy_mean": accuracy_mean,
                "balanced_accuracy_mean": balanced_mean,
                "mean_confidence_mean": confidence_mean,
                "mean_decoder_latency_ms_mean": latency_mean,
                "calibration_time_s_mean": calibration_mean,
                "target_acquisition_rate_mean": task_success_mean,
                "adaptation_update_count_mean": adaptation_updates_mean,
                "delta_from_clean": delta_from_clean,
                "normalized_score": normalized_score,
                "latency_score": _latency_score(latency_mean),
                "calibration_efficiency_score": _calibration_efficiency_score(calibration_mean),
                "stressbench_score": _stressbench_score(normalized_score, latency_mean, calibration_mean),
            }
        )

    order = {name: index for index, name in enumerate(BUILTIN_PRESETS)}
    return sorted(aggregates, key=lambda row: (str(row["architecture"]), order.get(str(row["preset"]), len(order))))


def robustness_summary(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in aggregates if row.get("balanced_accuracy_mean") is not None]
    stressed = [row for row in scored if row["preset"] != "clean"] or scored
    if not stressed:
        return {
            "robustness_score": None,
            "weakest_preset": None,
            "weakest_architecture": None,
            "worst_balanced_accuracy": None,
            "largest_drop_from_clean": None,
            "architecture_scores": {},
        }
    robustness_score = sum(float(row["balanced_accuracy_mean"]) for row in stressed) / len(stressed)
    weakest = min(stressed, key=lambda row: float(row["balanced_accuracy_mean"]))
    drops = [row for row in stressed if row.get("delta_from_clean") is not None]
    largest_drop = min(drops, key=lambda row: float(row["delta_from_clean"])) if drops else None
    return {
        "robustness_score": robustness_score,
        "weakest_preset": weakest["preset"],
        "weakest_architecture": weakest["architecture"],
        "worst_balanced_accuracy": weakest["balanced_accuracy_mean"],
        "largest_drop_from_clean": largest_drop["delta_from_clean"] if largest_drop else None,
        "largest_drop_preset": largest_drop["preset"] if largest_drop else None,
        "largest_drop_architecture": largest_drop["architecture"] if largest_drop else None,
        "architecture_scores": _architecture_scores(aggregates),
    }


def _write_summary(
    summary_dir: Path,
    spec: StressBenchSpec,
    architecture_cards: tuple[ArchitectureCard, ...],
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    robustness: dict[str, Any],
) -> None:
    payload = {
        "name": spec.name,
        "base_config": str(spec.base_config) if spec.base_config else None,
        "architectures": [card.to_dict() for card in architecture_cards],
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
    _write_csv(summary_dir / "stressbench_summary.csv", rows, ["architecture", "preset", "repeat", "run_id"])
    _write_csv(summary_dir / "stressbench_aggregates.csv", aggregates, ["architecture", "preset", "runs"])
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
        f"<td>{row['architecture']}</td>"
        f"<td>{row['preset']}</td>"
        f"<td>{row['runs']}</td>"
        f"<td>{_format_metric(row['balanced_accuracy_mean'])}</td>"
        f"<td>{_format_metric(row['delta_from_clean'])}</td>"
        f"<td>{_format_metric(row['mean_decoder_latency_ms_mean'])}</td>"
        f"<td>{_format_metric(row['calibration_time_s_mean'])}</td>"
        f"<td>{_format_metric(row['stressbench_score'])}</td>"
        "</tr>"
        for row in aggregates
    )
    body_rows = "\n".join(
        "<tr>"
        f"<td>{row['architecture']}</td>"
        f"<td>{row['preset']}</td>"
        f"<td>{row['repeat']}</td>"
        f"<td>{row['status']}</td>"
        f"<td>{_format_metric(row['accuracy'])}</td>"
        f"<td>{_format_metric(row['balanced_accuracy'])}</td>"
        f"<td>{_format_metric(row['mean_confidence'])}</td>"
        f"<td>{_format_metric(row['mean_decoder_latency_ms'])}</td>"
        f"<td>{_format_metric(row['calibration_time_s'])}</td>"
        f"<td><code>{row['run_id'] or ''}</code></td>"
        f"<td>{row['error'] or ''}</td>"
        "</tr>"
        for row in rows
    )
    architecture_rows = "\n".join(
        "<tr>"
        f"<td>{name}</td>"
        f"<td>{_format_metric(score.get('robustness_score'))}</td>"
        f"<td>{_format_metric(score.get('mean_latency_ms'))}</td>"
        f"<td>{_format_metric(score.get('mean_calibration_time_s'))}</td>"
        f"<td>{_format_metric(score.get('composite_score'))}</td>"
        f"<td>{score.get('weakest_preset') or ''}</td>"
        "</tr>"
        for name, score in (robustness.get("architecture_scores") or {}).items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>StressBench - {spec.name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.45; max-width: 1200px; margin: 40px auto; padding: 0 20px; }}
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
    <tr><td>Weakest architecture</td><td>{robustness.get("weakest_architecture")}</td></tr>
    <tr><td>Worst balanced accuracy</td><td>{_format_metric(robustness.get("worst_balanced_accuracy"))}</td></tr>
    <tr><td>Largest drop from clean</td><td>{_format_metric(robustness.get("largest_drop_from_clean"))}</td></tr>
    <tr><td>Largest drop preset</td><td>{robustness.get("largest_drop_preset")}</td></tr>
    <tr><td>Largest drop architecture</td><td>{robustness.get("largest_drop_architecture")}</td></tr>
  </table>
  <h2>Architecture Scores</h2>
  <table>
    <tr><th>Architecture</th><th>Robustness</th><th>Mean Latency ms</th><th>Mean Calibration s</th><th>Composite Score</th><th>Weakest Preset</th></tr>
    {architecture_rows}
  </table>
  <h2>Preset Aggregates</h2>
  <table>
    <tr><th>Architecture</th><th>Preset</th><th>Runs</th><th>Balanced Accuracy Mean</th><th>Delta From Clean</th><th>Latency ms</th><th>Calibration s</th><th>StressBench Score</th></tr>
    {aggregate_rows}
  </table>
  <h2>Runs</h2>
  <table>
    <tr><th>Architecture</th><th>Preset</th><th>Repeat</th><th>Status</th><th>Accuracy</th><th>Balanced Accuracy</th><th>Mean Confidence</th><th>Latency ms</th><th>Calibration s</th><th>Run ID</th><th>Error</th></tr>
    {body_rows}
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _architecture_cards(spec: StressBenchSpec) -> tuple[ArchitectureCard, ...]:
    cards: list[ArchitectureCard] = []
    if spec.base_config is not None:
        base_raw = yaml.safe_load(spec.base_config.read_text(encoding="utf-8"))
        base_experiment = parse_experiment_spec(base_raw)
        cards.append(
            ArchitectureCard(
                name=base_experiment.name,
                description=f"Custom architecture from {spec.base_config}",
                config=base_raw,
                tags=("custom",),
            )
        )
    cards.extend(BUILTIN_ARCHITECTURES[name] for name in spec.architectures)
    return tuple(cards)


def _optional_base_config(raw: dict[str, Any], spec_path: Path) -> Path | None:
    if raw.get("base_config") is None:
        return None
    base_config = Path(_string(raw.get("base_config"), "base_config"))
    if not base_config.is_absolute():
        base_config = spec_path.parent / base_config
    return base_config


def _architecture_names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError("architectures must be a list when provided")
    architectures = tuple(_string(item, "architectures[]") for item in value)
    unknown = sorted(set(architectures) - set(BUILTIN_ARCHITECTURES))
    if unknown:
        raise ConfigError(f"unknown stressbench architecture(s): {', '.join(unknown)}")
    return architectures


def _write_csv(path: Path, rows: list[dict[str, Any]], fallback: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else fallback
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _latency_score(latency_ms: float | None) -> float | None:
    if latency_ms is None:
        return None
    return 1.0 / (1.0 + max(latency_ms, 0.0) / 100.0)


def _calibration_efficiency_score(calibration_time_s: float | None) -> float | None:
    if calibration_time_s is None:
        return None
    return 1.0 / (1.0 + max(calibration_time_s, 0.0))


def _stressbench_score(normalized_score: float | None, latency_ms: float | None, calibration_time_s: float | None) -> float | None:
    if normalized_score is None:
        return None
    components = [(0.7, normalized_score)]
    latency = _latency_score(latency_ms)
    calibration = _calibration_efficiency_score(calibration_time_s)
    if latency is not None:
        components.append((0.15, latency))
    if calibration is not None:
        components.append((0.15, calibration))
    total_weight = sum(weight for weight, _value in components)
    return sum(weight * float(value) for weight, value in components) / total_weight


def _architecture_scores(aggregates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in aggregates:
        if row.get("preset") != "clean":
            grouped.setdefault(str(row["architecture"]), []).append(row)
    scores: dict[str, dict[str, Any]] = {}
    for architecture, rows in grouped.items():
        score_rows = [row for row in rows if row.get("balanced_accuracy_mean") is not None]
        weakest = min(score_rows, key=lambda row: float(row["balanced_accuracy_mean"]), default={})
        scores[architecture] = {
            "robustness_score": _mean_value(rows, "balanced_accuracy_mean"),
            "mean_latency_ms": _mean_value(rows, "mean_decoder_latency_ms_mean"),
            "mean_calibration_time_s": _mean_value(rows, "calibration_time_s_mean"),
            "composite_score": _mean_value(rows, "stressbench_score"),
            "weakest_preset": weakest.get("preset"),
        }
    return scores


def _mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
