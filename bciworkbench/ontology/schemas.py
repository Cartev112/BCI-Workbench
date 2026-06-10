from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bciworkbench.sim.profiles import SessionProfile, SubjectProfile


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


def _validate_keys(values: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"{path} has unsupported key(s): {', '.join(unknown)}")


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
    if source_type not in {"synthetic_motor_imagery", "mne_raw", "moabb"}:
        raise ConfigError("source.type must be one of: synthetic_motor_imagery, mne_raw, moabb")
    allowed_source_keys = _allowed_source_keys(source_type)
    _validate_keys(source_params, allowed_source_keys, "source")
    _validate_source_params(source_type, source_params)

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
    decoder_params = pipeline[-1].params
    if decoder_params.get("estimator", "lda") not in {
        "lda",
        "logistic",
        "logistic_regression",
        "lr",
        "nearest_centroid",
        "centroid",
    }:
        raise ConfigError("pipeline decoder estimator must be lda, logistic_regression, or nearest_centroid")

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


def _allowed_source_keys(source_type: str) -> set[str]:
    if source_type == "synthetic_motor_imagery":
        return {
            "duration_s",
            "sampling_rate",
            "n_channels",
            "n_trials",
            "trial_duration_s",
            "inter_trial_s",
            "snr_db",
            "drift",
            "line_noise_hz",
            "subject",
            "session",
        }
    if source_type == "mne_raw":
        return {"path", "preload", "event_id_prefix"}
    if source_type == "moabb":
        return {"dataset", "subject", "paradigm"}
    return set()


def _validate_source_params(source_type: str, source_params: dict[str, Any]) -> None:
    if source_type == "synthetic_motor_imagery":
        if "sampling_rate" in source_params:
            _positive_number(source_params["sampling_rate"], "source.sampling_rate")
        if "duration_s" in source_params:
            _positive_number(source_params["duration_s"], "source.duration_s")
        if "subject" in source_params:
            subject = _mapping(source_params["subject"], "source.subject")
            _validate_keys(subject, set(SubjectProfile().__dict__), "source.subject")
        if "session" in source_params:
            session = _mapping(source_params["session"], "source.session")
            _validate_keys(session, set(SessionProfile().__dict__), "source.session")
    elif source_type == "mne_raw":
        _string(source_params.get("path"), "source.path")
    elif source_type == "moabb":
        dataset = _string(source_params.get("dataset"), "source.dataset")
        if dataset != "BNCI2014_001":
            raise ConfigError("only source.dataset=BNCI2014_001 is supported for moabb in this milestone")
        if "subject" in source_params:
            subject = int(source_params["subject"])
            if subject <= 0:
                raise ConfigError("source.subject must be positive")
