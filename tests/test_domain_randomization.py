from __future__ import annotations

from bciworkbench.sim.domain_randomization import DomainRandomization, UniformRange


def test_domain_randomization_samples_nested_source_overrides() -> None:
    randomization = DomainRandomization(
        snr_db=UniformRange(-12, -6),
        electrode_shift_mm=UniformRange(0, 12),
        attention=UniformRange(0.6, 0.9),
        feedback_delay_ms=UniformRange(20, 200),
    )
    sample = randomization.sample_source_overrides(seed=1)
    assert -12 <= sample["snr_db"] <= -6
    assert 0 <= sample["session"]["electrode_shift_mm"] <= 12
    assert 0.6 <= sample["subject"]["attention"] <= 0.9
    assert 20 <= sample["session"]["feedback_delay_ms"] <= 200
