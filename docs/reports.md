# Report Interpretation Guide

Each run writes a directory under `runs/<run_id>/`. The HTML report is a compact
view over JSON/CSV artifacts; the files are the source of truth.

## Core Artifacts

- `resolved_config.json`: parsed experiment config.
- `ontology_schema.json`: ontology schema snapshot.
- `graph.json`: runtime nodes and edges.
- `telemetry.jsonl`: per-node runtime latency and status.
- `channel_schema.json`: channel metadata.
- `source_metadata.json`: source-level provenance and stressor settings.
- `metrics.json`: decoder, task, adaptation, and system metrics.
- `events.csv`, `windows.csv`, `features.csv`, `predictions.csv`: tabular run
  outputs.
- `model/model_card.json`: decoder adapter metadata.
- `provenance.json`: environment and random seed metadata.

## Replay Artifacts

Replay runs add:

- `latency_trace.csv`
- `latency_trace.json`
- `stream_health.json`

Use these to distinguish original sample or marker timing from simulated packet
arrival, backlog, queue depth, and dropped-packet telemetry.

## Closed-Loop Artifacts

`cursor_1d` runs add:

- `task_metrics.json`
- `task_states.csv`
- `feedback.csv`

The key comparison is often `balanced_accuracy` versus
`target_acquisition_rate`. A decoder can classify windows well and still perform
poorly in a task because confidence, delay, dwell, or false activations change
task success.

## Adaptation Artifacts

Adaptive runs add:

- `adaptation_metrics.json`
- `adaptation.jsonl`
- `predictions_before_adaptation.csv`

Check before/after balanced accuracy, changed prediction rate, update count, and
catastrophic update warnings. The current adapters are deterministic interfaces
for controlled comparison, not production online-learning algorithms.

## StressBench Reports

StressBench writes:

- `stressbench_summary.json`
- `stressbench_summary.csv`
- `stressbench_aggregates.csv`
- `stressbench_report.html`

Read the report from top down:

1. Robustness summary identifies the weakest preset and largest clean-relative
   drop.
2. Architecture scores combine robustness, latency, and calibration efficiency.
3. Preset aggregates show which stressors break each architecture.
4. Run rows link back to individual run directories.
