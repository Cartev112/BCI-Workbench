from __future__ import annotations

from pathlib import Path

from bciworkbench.experiment import Experiment
from bciworkbench.sim.profiles import SessionProfile
from bciworkbench.sources.p300 import SyntheticP300Config, SyntheticP300Source


def test_synthetic_p300_source_emits_target_and_non_target_trials() -> None:
    packet = SyntheticP300Source(
        SyntheticP300Config(duration_s=30, n_trials=20, target_probability=0.4, seed=4)
    ).read()
    trials = [event for event in packet.events if event.event_type == "trial.start"]
    labels = {event.target for event in trials}
    assert labels == {"target", "non_target"}
    assert packet.metadata["simulation_level"] == "2_p300_paradigm_with_subject_session_stressors"


def test_synthetic_p300_artifacts_use_p300_source() -> None:
    packet = SyntheticP300Source(
        SyntheticP300Config(
            duration_s=10,
            n_trials=6,
            seed=8,
            session=SessionProfile(
                blink_rate_per_min=60,
                muscle_noise=1.0,
                channel_dropout_probability=0.2,
            ),
        )
    ).read()
    artifact_sources = {event.source for event in packet.events if event.event_type.startswith("artifact.")}
    assert artifact_sources == {"synthetic_p300"}


def test_p300_example_runs_existing_pipeline(tmp_path: Path) -> None:
    experiment = Experiment.from_yaml("examples/p300_synthetic.yml")
    spec = experiment.spec
    patched_spec = type(spec)(
        schema_version=spec.schema_version,
        name=spec.name,
        paradigm=spec.paradigm,
        mode=spec.mode,
        source=spec.source,
        pipeline=spec.pipeline,
        task=spec.task,
        metrics=spec.metrics,
        random_seed=spec.random_seed,
        output_dir=str(tmp_path),
        metadata=spec.metadata,
    )
    result = Experiment(patched_spec).run()
    assert result.metrics["source"] == "synthetic_p300"
    assert result.metrics["n_predictions"] > 0
    assert result.metrics["balanced_accuracy"] is not None
