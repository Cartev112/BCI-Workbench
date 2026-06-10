from __future__ import annotations

from pathlib import Path

from bciworkbench.stressbench import BUILTIN_PRESETS, load_stressbench_spec, run_stressbench


def test_builtin_presets_include_core_stressors() -> None:
    assert "clean" in BUILTIN_PRESETS
    assert "low_snr" in BUILTIN_PRESETS
    assert "session_drift" in BUILTIN_PRESETS


def test_load_stressbench_spec_resolves_relative_base_config() -> None:
    spec = load_stressbench_spec("examples/stressbench_mi.yml")
    assert spec.name == "mi_stressbench"
    assert spec.base_config.name == "mi_synthetic.yml"
    assert "low_snr" in spec.presets


def test_run_stressbench_writes_summary(tmp_path: Path) -> None:
    stressbench_config = tmp_path / "stressbench.yml"
    stressbench_config.write_text(
        "\n".join(
            [
                "name: test_stressbench",
                f"base_config: {Path('examples/mi_synthetic.yml').resolve()}",
                f"output_dir: {tmp_path}",
                "repeats: 1",
                "presets:",
                "  - clean",
                "  - low_snr",
            ]
        ),
        encoding="utf-8",
    )

    result = run_stressbench(stressbench_config)

    assert len(result.rows) == 2
    assert (result.summary_dir / "stressbench_summary.json").exists()
    assert (result.summary_dir / "stressbench_summary.csv").exists()
    assert (result.summary_dir / "stressbench_report.html").exists()
