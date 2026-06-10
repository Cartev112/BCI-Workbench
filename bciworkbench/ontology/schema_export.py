from __future__ import annotations

from typing import Any

from bciworkbench.ontology.timing import SUPPORTED_CLOCK_DOMAINS
from bciworkbench.ontology.validation import SUPPORTED_CHANNEL_TYPES, SUPPORTED_MODALITIES
from bciworkbench.sim.profiles import SessionProfile, SubjectProfile


SCHEMA_VERSION = "0.1"


def experiment_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for milestone experiment configs."""

    subject_properties = {key: {"type": "number"} for key in SubjectProfile().__dict__}
    session_properties = {key: {"type": "number"} for key in SessionProfile().__dict__}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://cartev112.github.io/bci-workbench/schemas/experiment-0.1.json",
        "title": "BCI Workbench Experiment",
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "paradigm", "source", "pipeline", "task"],
        "properties": {
            "schema_version": {"type": "string", "default": SCHEMA_VERSION},
            "name": {"type": "string", "minLength": 1},
            "paradigm": {"type": "string", "minLength": 1},
            "mode": {"type": "string", "enum": ["synthetic", "offline", "replay", "live"], "default": "synthetic"},
            "random_seed": {"type": "integer", "default": 0},
            "output_dir": {"type": "string", "default": "runs"},
            "source": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["synthetic_motor_imagery", "synthetic_p300", "mne_raw", "moabb", "xdf_replay"],
                    },
                    "duration_s": {"type": "number", "exclusiveMinimum": 0},
                    "sampling_rate": {"type": "number", "exclusiveMinimum": 0},
                    "n_channels": {"type": "integer", "minimum": 1},
                    "n_trials": {"type": "integer", "minimum": 1},
                    "trial_duration_s": {"type": "number", "exclusiveMinimum": 0},
                    "inter_trial_s": {"type": "number", "minimum": 0},
                    "target_probability": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                    "snr_db": {"type": "number"},
                    "drift": {"type": "number"},
                    "line_noise_hz": {"type": "number", "exclusiveMinimum": 0},
                    "subject": {
                        "oneOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": subject_properties,
                            },
                            {"type": "integer", "minimum": 1},
                        ]
                    },
                    "session": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": session_properties,
                    },
                    "path": {"type": "string"},
                    "preload": {"type": "boolean"},
                    "event_id_prefix": {"type": "string"},
                    "dataset": {"type": "string", "enum": ["BNCI2014_001"]},
                    "paradigm": {"type": "string"},
                    "signal_stream": {"type": "string"},
                    "marker_stream": {"type": "string"},
                    "speed_mode": {"type": "string", "enum": ["fastest", "real_time", "scaled", "stepped"]},
                    "speed": {"type": "number", "exclusiveMinimum": 0},
                    "chunk_duration_s": {"type": "number", "exclusiveMinimum": 0},
                    "step_duration_s": {"type": "number", "minimum": 0},
                    "processing_time_ms": {"type": "number", "minimum": 0},
                    "queue_capacity": {"type": "integer", "minimum": 1},
                },
            },
            "pipeline": {
                "type": "array",
                "minItems": 3,
                "items": {
                    "type": "object",
                    "required": ["type"],
                    "properties": {"type": {"type": "string"}},
                    "additionalProperties": True,
                },
            },
            "task": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["motor_imagery_classification", "p300_classification", "cursor_1d"],
                    },
                    "target_position": {"type": "number", "exclusiveMinimum": 0},
                    "target_radius": {"type": "number", "minimum": 0},
                    "target_dwell_steps": {"type": "integer", "minimum": 1},
                    "step_size": {"type": "number", "exclusiveMinimum": 0},
                    "control_interval_s": {"type": "number", "exclusiveMinimum": 0},
                    "feedback_delay_ms": {"type": "number", "minimum": 0},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                    "reset_on_target_change": {"type": "boolean"},
                },
                "additionalProperties": True,
            },
            "adaptation": {
                "type": "object",
                "required": ["type"],
                "additionalProperties": False,
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["none", "noop", "supervised_batch", "confidence_gated", "drift_triggered"],
                        "default": "none",
                    },
                    "batch_size": {"type": "integer", "minimum": 1},
                    "min_samples": {"type": "integer", "minimum": 1},
                    "confidence_gate": {"type": "number", "minimum": 0, "maximum": 1},
                    "accuracy_floor": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence_floor": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            "metrics": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object"},
        },
    }


def ontology_json_schema() -> dict[str, Any]:
    """Return reportable ontology schema fragments for artifacts and adapters."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://cartev112.github.io/bci-workbench/schemas/ontology-0.1.json",
        "title": "BCI Workbench Ontology",
        "type": "object",
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "clock_domains": {"type": "array", "items": {"enum": sorted(SUPPORTED_CLOCK_DOMAINS)}},
            "channel_types": {"type": "array", "items": {"enum": sorted(SUPPORTED_CHANNEL_TYPES)}},
            "modalities": {"type": "array", "items": {"enum": sorted(SUPPORTED_MODALITIES)}},
            "channel_schema": {"$ref": "#/$defs/ChannelSchema"},
            "event": {"$ref": "#/$defs/Event"},
            "signal_packet": {"$ref": "#/$defs/SignalPacket"},
            "window_packet": {"$ref": "#/$defs/WindowPacket"},
            "feature_packet": {"$ref": "#/$defs/FeaturePacket"},
            "intent_packet": {"$ref": "#/$defs/IntentPacket"},
            "task_state_packet": {"$ref": "#/$defs/TaskStatePacket"},
            "feedback_packet": {"$ref": "#/$defs/FeedbackPacket"},
            "adaptation_packet": {"$ref": "#/$defs/AdaptationPacket"},
        },
        "$defs": {
            "ChannelSchema": {
                "type": "object",
                "required": ["names", "types", "units", "sampling_rate"],
                "properties": {
                    "names": {"type": "array", "items": {"type": "string"}},
                    "types": {"type": "array", "items": {"enum": sorted(SUPPORTED_CHANNEL_TYPES)}},
                    "units": {"type": "array", "items": {"type": "string"}},
                    "sampling_rate": {"type": "number", "exclusiveMinimum": 0},
                    "reference": {"type": "string"},
                    "montage": {"type": ["string", "null"]},
                    "bad_channels": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
            },
            "Event": {
                "type": "object",
                "required": ["event_id", "event_type", "name", "onset", "clock_domain"],
                "properties": {
                    "event_id": {"type": "string"},
                    "event_type": {"type": "string"},
                    "name": {"type": "string"},
                    "onset": {"type": "number", "minimum": 0},
                    "duration": {"type": "number", "minimum": 0},
                    "clock_domain": {"enum": sorted(SUPPORTED_CLOCK_DOMAINS)},
                    "sample_index": {"type": ["integer", "null"], "minimum": 0},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "source": {"type": ["string", "null"]},
                    "target": {"type": ["string", "null"]},
                    "metadata": {"type": "object"},
                },
            },
            "SignalPacket": {"type": "object", "required": ["shape", "clock_domain", "modality"]},
            "WindowPacket": {"type": "object", "required": ["window_id", "start_time", "end_time", "label"]},
            "FeaturePacket": {"type": "object", "required": ["feature_id", "feature_names", "window_id"]},
            "IntentPacket": {"type": "object", "required": ["intent_id", "intent", "confidence", "window_id"]},
            "TaskStatePacket": {"type": "object", "required": ["task_id", "state", "observation", "done", "success"]},
            "FeedbackPacket": {"type": "object", "required": ["action", "rendered_at", "clock_domain", "delay_ms"]},
            "AdaptationPacket": {"type": "object", "required": ["adapter_id", "update_type", "input_window_ids"]},
        },
    }
