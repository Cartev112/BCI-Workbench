from __future__ import annotations

import numpy as np
import pytest

from bciworkbench.ontology.packets import (
    AdaptationPacket,
    ChannelSchema,
    Event,
    FeaturePacket,
    FeedbackPacket,
    IntentPacket,
    SignalPacket,
    TaskStatePacket,
    WindowPacket,
)
from bciworkbench.ontology.schema_export import experiment_json_schema, ontology_json_schema
from bciworkbench.ontology.timing import ClockDomain


def test_clock_domain_validation() -> None:
    assert ClockDomain("sample_clock", "sample-derived").to_dict()["name"] == "sample_clock"
    with pytest.raises(ValueError, match="unsupported clock domain"):
        ClockDomain("bad_clock", "bad")


def test_channel_schema_rejects_duplicate_or_unsupported_channels() -> None:
    with pytest.raises(ValueError, match="unique"):
        ChannelSchema(
            names=("C3", "C3"),
            types=("eeg", "eeg"),
            units=("V", "V"),
            sampling_rate=250,
        )
    with pytest.raises(ValueError, match="unsupported channel"):
        ChannelSchema(
            names=("C3",),
            types=("not_real",),
            units=("V",),
            sampling_rate=250,
        )


def test_event_validation_and_serialization() -> None:
    event = Event(
        event_id="cue-1",
        event_type="cue.onset",
        name="left",
        onset=1.0,
        sample_index=250,
        target="left",
    )
    assert event.to_dict()["clock_domain"] == "sample_clock"
    with pytest.raises(ValueError, match="confidence"):
        Event(event_id="bad", event_type="cue.onset", name="bad", onset=0.0, confidence=2.0)


def test_packet_validation_and_artifact_friendly_dicts() -> None:
    channels = ChannelSchema(
        names=("C3", "C4"),
        types=("eeg", "eeg"),
        units=("V", "V"),
        sampling_rate=250,
    )
    signal = SignalPacket(
        data=np.zeros((2, 10)),
        timestamps=np.arange(10) / 250,
        channel_schema=channels,
        modality="EEG",
        events=[],
    )
    assert signal.to_dict()["shape"] == [2, 10]
    with pytest.raises(ValueError, match="monotonic"):
        SignalPacket(
            data=np.zeros((2, 3)),
            timestamps=np.array([0.0, 0.2, 0.1]),
            channel_schema=channels,
            modality="EEG",
        )

    window = WindowPacket(
        window_id="w1",
        data=np.zeros((2, 5)),
        start_time=0.0,
        end_time=0.02,
        sample_start=0,
        sample_end=5,
        label="left",
    )
    assert window.to_dict()["center_time"] == pytest.approx(0.01)

    feature = FeaturePacket(
        feature_id="f1",
        features=np.array([1.0, 2.0]),
        feature_names=("a", "b"),
        window_id="w1",
        label="left",
    )
    assert feature.to_dict()["feature_count"] == 2

    intent = IntentPacket(
        intent_id="p1",
        intent="left",
        confidence=0.8,
        posterior={"left": 0.8, "right": 0.2},
        latency_ms=1.0,
        window_id="w1",
        decoder_id="d1",
        label="left",
    )
    assert intent.to_dict()["posterior"]["left"] == 0.8


def test_task_feedback_and_adaptation_packets_serialize() -> None:
    task_state = TaskStatePacket(
        task_id="cursor_1d",
        state={"x": 0.1},
        observation={"cursor": 0.1},
        target="left",
        reward=1.0,
        done=False,
        success=False,
    )
    feedback = FeedbackPacket(
        action="move_left",
        rendered_at=1.0,
        clock_domain="sim_clock",
        reward=1.0,
        delay_ms=80.0,
        task_state=task_state,
    )
    adaptation = AdaptationPacket(
        adapter_id="confidence_gate",
        update_type="decoder.threshold",
        input_window_ids=("w1",),
        labels=("left",),
        confidence_gate=0.8,
    )
    assert feedback.to_dict()["task_state"]["task_id"] == "cursor_1d"
    assert adaptation.to_dict()["input_window_ids"] == ["w1"]


def test_json_schema_exports_include_core_ontology() -> None:
    experiment_schema = experiment_json_schema()
    ontology_schema = ontology_json_schema()
    assert experiment_schema["properties"]["source"]["properties"]["type"]["enum"] == [
        "synthetic_motor_imagery",
        "mne_raw",
        "moabb",
    ]
    assert "ChannelSchema" in ontology_schema["$defs"]
    assert "FeedbackPacket" in ontology_schema["$defs"]
    assert "sample_clock" in ontology_schema["properties"]["clock_domains"]["items"]["enum"]
