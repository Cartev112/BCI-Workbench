# BCI Workbench

BCI Workbench is a Python-first systems workbench for BCI architecture experiments.

The goal is to define a BCI pipeline once and run it against synthetic subjects, public datasets, recorded streams, replayed real-time streams, and live hardware while preserving comparable ontology, timing, metrics, and run artifacts.

The first implementation slice supports a narrow but real path:

```text
synthetic motor imagery -> trial windows -> bandpower features -> decoder -> metrics/report
```

## Quickstart

```powershell
pip install -e ".[dev]"
bciworkbench validate examples/mi_synthetic.yml
bciworkbench run examples/mi_synthetic.yml
pytest
```

The run command writes artifacts under `runs/<run_id>/`, including `metrics.json`, event/window/prediction CSV files, provenance, and a simple HTML report.

## Current Scope

Implemented now:

- YAML experiment validation.
- Core ontology dataclasses for channels, events, packets, predictions, and task state.
- Deterministic synthetic motor imagery source with ground-truth trial events.
- Windowing and bandpower feature extraction.
- LDA decoder when scikit-learn is available, with a deterministic nearest-centroid fallback.
- Basic decoder metrics and run reports.
- CLI commands: `validate`, `run`, and `report`.

Planned next:

- MNE/MOABB source adapters.
- Richer simulation stressors for drift, artifacts, and timing.
- Replay scheduler and XDF/LSL support.
- Closed-loop cursor task metrics.
- StressBench robustness sweeps.

