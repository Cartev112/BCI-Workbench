from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment config is invalid."""


@dataclass(frozen=True)
class SourceSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineStepSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    schema_version: str
    name: str
    paradigm: str
    mode: str
    source: SourceSpec
    pipeline: tuple[PipelineStepSpec, ...]
    task: TaskSpec
    metrics: tuple[str, ...] = ()
    random_seed: int = 0
    output_dir: str = "runs"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "paradigm": self.paradigm,
            "mode": self.mode,
            "random_seed": self.random_seed,
            "output_dir": self.output_dir,
            "source": {"type": self.source.type, **self.source.params},
            "pipeline": [{"type": step.type, **step.params} for step in self.pipeline],
            "task": {"type": self.task.type, **self.task.params},
            "metrics": list(self.metrics),
            "metadata": self.metadata,
        }


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _number(value: Any, path: str) -> float:
    if not isinstance(value, int | float):
        raise ConfigError(f"{path} must be a number")
    return float(value)


def _positive_number(value: Any, path: str) -> float:
    number = _number(value, path)
    if number <= 0:
        raise ConfigError(f"{path} must be positive")
    return number


def parse_experiment_spec(raw: dict[str, Any]) -> ExperimentSpec:
    config = _mapping(raw, "config")

    schema_version = str(config.get("schema_version", "0.1"))
    name = _string(config.get("name"), "name")
    paradigm = _string(config.get("paradigm"), "paradigm")
    mode = _string(config.get("mode", "synthetic"), "mode")
    if mode not in {"synthetic", "offline", "replay", "live"}:
        raise ConfigError("mode must be one of: synthetic, offline, replay, live")

    source_raw = _mapping(config.get("source"), "source")
    source_type = _string(source_raw.get("type"), "source.type")
    source_params = {key: value for key, value in source_raw.items() if key != "type"}
    if source_type != "synthetic_motor_imagery":
        raise ConfigError("only source.type=synthetic_motor_imagery is implemented in this milestone")
    if "sampling_rate" in source_params:
        _positive_number(source_params["sampling_rate"], "source.sampling_rate")
    if "duration_s" in source_params:
        _positive_number(source_params["duration_s"], "source.duration_s")

    pipeline_raw = config.get("pipeline")
    if not isinstance(pipeline_raw, list) or not pipeline_raw:
        raise ConfigError("pipeline must be a non-empty list")
    pipeline: list[PipelineStepSpec] = []
    for index, step_raw in enumerate(pipeline_raw):
        step = _mapping(step_raw, f"pipeline[{index}]")
        step_type = _string(step.get("type"), f"pipeline[{index}].type")
        params = {key: value for key, value in step.items() if key != "type"}
        pipeline.append(PipelineStepSpec(type=step_type, params=params))

    implemented_steps = {"window", "bandpower", "decoder"}
    unknown = [step.type for step in pipeline if step.type not in implemented_steps]
    if unknown:
        raise ConfigError(f"unsupported pipeline step(s): {', '.join(unknown)}")
    if [step.type for step in pipeline] != ["window", "bandpower", "decoder"]:
        raise ConfigError("this milestone requires pipeline order: window, bandpower, decoder")

    task_raw = _mapping(config.get("task"), "task")
    task_type = _string(task_raw.get("type"), "task.type")
    if task_type != "motor_imagery_classification":
        raise ConfigError("only task.type=motor_imagery_classification is implemented in this milestone")
    task_params = {key: value for key, value in task_raw.items() if key != "type"}

    metrics_raw = config.get("metrics", [])
    if not isinstance(metrics_raw, list):
        raise ConfigError("metrics must be a list")
    metrics = tuple(_string(item, "metrics[]") for item in metrics_raw)

    random_seed = int(config.get("random_seed", 0))
    output_dir = str(config.get("output_dir", "runs"))
    metadata = config.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ConfigError("metadata must be a mapping")

    return ExperimentSpec(
        schema_version=schema_version,
        name=name,
        paradigm=paradigm,
        mode=mode,
        source=SourceSpec(type=source_type, params=source_params),
        pipeline=tuple(pipeline),
        task=TaskSpec(type=task_type, params=task_params),
        metrics=metrics,
        random_seed=random_seed,
        output_dir=output_dir,
        metadata=metadata,
    )


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_experiment_spec(raw)

