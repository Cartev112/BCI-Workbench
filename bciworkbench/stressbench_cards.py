from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArchitectureCard:
    name: str
    description: str
    config: dict[str, Any]
    tags: tuple[str, ...] = ()
    optional_extra: str | None = None
    expected_failure_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "optional_extra": self.optional_extra,
            "expected_failure_hint": self.expected_failure_hint,
            "metadata": self.metadata,
        }

    def config_copy(self) -> dict[str, Any]:
        return deepcopy(self.config)


BUILTIN_ARCHITECTURES: dict[str, ArchitectureCard] = {
    "mi_bandpower_lda": ArchitectureCard(
        name="mi_bandpower_lda",
        description="Motor imagery synthetic EEG with bandpower features and LDA decoder.",
        tags=("motor_imagery", "bandpower", "lda", "synthetic"),
        config={
            "schema_version": "0.1",
            "name": "stressbench_mi_bandpower_lda",
            "paradigm": "motor_imagery",
            "mode": "synthetic",
            "random_seed": 7,
            "output_dir": "runs",
            "source": {
                "type": "synthetic_motor_imagery",
                "duration_s": 120,
                "sampling_rate": 250,
                "n_channels": 16,
                "n_trials": 80,
                "trial_duration_s": 2.5,
                "inter_trial_s": 0.5,
                "snr_db": -4,
            },
            "pipeline": [
                {"type": "window", "length_s": 1.5, "offset_s": 0.3},
                {"type": "bandpower"},
                {"type": "decoder", "estimator": "lda", "calibration_fraction": 0.6},
            ],
            "task": {"type": "motor_imagery_classification", "classes": ["left", "right"]},
            "metrics": ["accuracy", "balanced_accuracy", "mean_confidence", "calibration_time_s"],
        },
    ),
    "mi_covariance_pyriemann_mdm": ArchitectureCard(
        name="mi_covariance_pyriemann_mdm",
        description="Motor imagery covariance features with optional pyRiemann MDM decoder.",
        tags=("motor_imagery", "covariance", "pyriemann", "mdm", "synthetic"),
        optional_extra="pyriemann",
        expected_failure_hint='Install with: pip install "bciworkbench[pyriemann]"',
        config={
            "schema_version": "0.1",
            "name": "stressbench_mi_covariance_pyriemann_mdm",
            "paradigm": "motor_imagery",
            "mode": "synthetic",
            "random_seed": 7,
            "output_dir": "runs",
            "source": {
                "type": "synthetic_motor_imagery",
                "duration_s": 120,
                "sampling_rate": 250,
                "n_channels": 16,
                "n_trials": 80,
                "trial_duration_s": 2.5,
                "inter_trial_s": 0.5,
                "snr_db": -4,
            },
            "pipeline": [
                {"type": "window", "length_s": 1.5, "offset_s": 0.3},
                {"type": "covariance"},
                {"type": "decoder", "adapter": "pyriemann", "estimator": "mdm", "calibration_fraction": 0.6},
            ],
            "task": {"type": "motor_imagery_classification", "classes": ["left", "right"]},
            "metrics": ["accuracy", "balanced_accuracy", "calibration_time_s"],
        },
    ),
    "p300_erp_lda": ArchitectureCard(
        name="p300_erp_lda",
        description="Synthetic P300 oddball with ERP time-bin features and LDA decoder.",
        tags=("p300", "erp", "lda", "synthetic"),
        config={
            "schema_version": "0.1",
            "name": "stressbench_p300_erp_lda",
            "paradigm": "p300",
            "mode": "synthetic",
            "random_seed": 13,
            "output_dir": "runs",
            "source": {
                "type": "synthetic_p300",
                "duration_s": 150,
                "sampling_rate": 250,
                "n_channels": 16,
                "n_trials": 120,
                "trial_duration_s": 0.8,
                "inter_trial_s": 0.2,
                "target_probability": 0.3,
                "snr_db": -4,
            },
            "pipeline": [
                {"type": "window", "length_s": 0.7, "offset_s": 0.0},
                {"type": "erp_features", "n_bins": 10},
                {"type": "decoder", "estimator": "lda", "calibration_fraction": 0.6},
            ],
            "task": {"type": "p300_classification", "classes": ["target", "non_target"]},
            "metrics": ["accuracy", "balanced_accuracy", "mean_confidence", "calibration_time_s"],
        },
    ),
}
