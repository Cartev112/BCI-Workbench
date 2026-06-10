# BCI Workbench

BCI Workbench is a Python-first systems workbench for BCI architecture experiments.

The goal is to define a BCI pipeline once and run it against synthetic subjects, public datasets, recorded streams, replayed real-time streams, and live hardware while preserving comparable ontology, timing, metrics, and run artifacts.

The current implementation supports deterministic synthetic BCI paths plus replay timing for recorded streams:

```text
synthetic motor imagery -> trial windows -> bandpower features -> decoder -> metrics/report
synthetic P300 oddball -> event windows -> ERP features -> decoder -> metrics/report
XDF replay source -> replay scheduler -> existing window/feature/decoder pipeline -> latency/backlog artifacts
synthetic motor imagery -> decoder predictions -> 1D cursor task -> feedback/task metrics/report
synthetic motor imagery -> decoder predictions -> adaptation node -> adaptation metrics/update log
```

## Quickstart

```powershell
pip install -e ".[dev]"
bciworkbench validate examples/mi_synthetic.yml
bciworkbench validate examples/p300_synthetic.yml
bciworkbench schema experiment
bciworkbench run examples/mi_synthetic.yml
bciworkbench run examples/p300_synthetic.yml
bciworkbench run examples/mi_synthetic_adaptive.yml
bciworkbench compare runs/<run_a> runs/<run_b>
bciworkbench stressbench examples/stressbench_mi.yml
pytest
```

The run command writes artifacts under `runs/<run_id>/`, including `ontology_schema.json`, `graph.json`, `telemetry.jsonl`, `source_metadata.json`, `metrics.json`, event/window/prediction CSV files, provenance, and a simple HTML report. Replay runs also write `latency_trace.csv`, `latency_trace.json`, and `stream_health.json`. Closed-loop cursor runs also write `task_metrics.json`, `task_states.csv`, and `feedback.csv`. Adaptive runs also write `adaptation_metrics.json`, `adaptation.jsonl`, and `predictions_before_adaptation.csv`.

## Current Scope

Implemented now:

- YAML experiment validation.
- Core ontology dataclasses for channels, events, packets, predictions, and task state.
- JSON Schema export for experiment configs and ontology artifacts.
- Deterministic linear graph runtime with node telemetry.
- Explicit synthetic subject/session profiles with artifact, drift, timing, and ERP parameters.
- StressBench preset matrix with aggregate robustness scoring for synthetic checks.
- Domain randomization helpers for sampling subject/session stressor ranges.
- Deterministic synthetic motor imagery source with ground-truth trial events.
- Deterministic synthetic P300 oddball source with target/non-target stimulus events.
- Windowing, bandpower feature extraction, and simple ERP feature extraction.
- Sklearn-style decoder adapter with LDA, logistic regression, model persistence, model cards, and a deterministic nearest-centroid fallback.
- Optional pyRiemann MDM adapter with dependency guard.
- MNE Raw FIF source adapter and initial MOABB BNCI2014_001 adapter with optional dependency guards.
- XDF replay adapter with deterministic fastest, real-time, scaled, and stepped scheduler modes.
- Replay packet arrival, backlog, queue-depth, latency trace, and stream health artifacts.
- Closed-loop `cursor_1d` task runtime with feedback packets, task states, feedback delay modeling, and task success metrics.
- Adaptation interfaces with no-op, supervised batch, confidence-gated, and drift-triggered recalibration adapters.
- AdaptationPacket logging plus stability metrics and report warnings for harmful updates.
- Basic decoder metrics and run reports.
- CLI commands: `validate`, `schema`, `run`, `report`, `compare`, and `stressbench`.

Planned next:

- LSL and BrainFlow live sources.
- Broader closed-loop task environments and richer online decoder update backends.
- Broader StressBench robustness sweeps and public benchmark presets.
