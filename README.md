# BCI Workbench

BCI Workbench is a Python SDK for defining, simulating, replaying, evaluating,
and reporting brain-computer interface pipelines with consistent runtime
artifacts.

The SDK is designed around one operational contract: define a BCI architecture
once, then run it against synthetic subjects, public datasets, recorded streams,
and replayed real-time streams while preserving comparable ontology, timing,
metrics, provenance, and reports.

> Status: pre-1.0 research and engineering workbench. The current implementation
> is suitable for architecture experiments, controlled synthetic stress tests,
> replay timing evaluation, and reproducible run reports. It is not a clinical
> or medical-device system.

## What It Provides

BCI Workbench packages the core components needed to evaluate BCI systems beyond
offline classifier accuracy:

- A typed ontology for signals, events, windows, features, intents, task states,
  feedback, and adaptation updates.
- A deterministic graph runtime with per-node telemetry.
- Synthetic motor imagery and P300 sources with explicit subject, session,
  artifact, drift, and timing parameters.
- Offline adapters for MNE Raw FIF and MOABB BNCI2014_001.
- XDF replay support with deterministic scheduler modes and backlog telemetry.
- Decoder adapters for sklearn-style models and optional pyRiemann MDM.
- Closed-loop `cursor_1d` task evaluation.
- Adaptation interfaces with update logs and stability metrics.
- StressBench benchmark suites for stressor and architecture robustness.
- Reproducible run directories with JSON, CSV, JSONL, model, provenance, and
  HTML report artifacts.

## Core Workflow

```text
source -> windowing -> features -> decoder -> optional adaptation -> optional task -> reports
```

Implemented paths include:

```text
synthetic motor imagery -> bandpower -> decoder -> metrics/report
synthetic P300 oddball -> ERP features -> decoder -> metrics/report
XDF replay -> scheduler telemetry -> existing decoder pipeline -> latency report
motor imagery decoder -> cursor_1d task -> feedback/task metrics/report
decoder predictions -> adaptation node -> update log/stability metrics
StressBench architecture cards -> stressor matrix -> robustness report
```

## Installation

Core development install:

```powershell
pip install -e ".[dev]"
```

Optional integration extras:

| Extra | Enables |
| --- | --- |
| `bciworkbench[mne]` | MNE Raw FIF source adapter |
| `bciworkbench[moabb]` | MOABB dataset source adapter |
| `bciworkbench[pyriemann]` | pyRiemann MDM decoder |
| `bciworkbench[xdf]` | Real `.xdf` replay via `pyxdf` |

Optional dependencies are guarded. If an adapter is requested without its extra,
the SDK raises an install hint instead of failing at import time.

## Quickstart

Run a deterministic synthetic motor imagery experiment:

```powershell
bciworkbench validate examples/mi_synthetic.yml
bciworkbench run examples/mi_synthetic.yml
```

The `run` command prints the generated report path:

```text
runs/<run_id>/report.html
```

Run the test suite:

```powershell
pytest
```

Generate the experiment JSON Schema:

```powershell
bciworkbench schema experiment
```

## Example Commands

Synthetic P300:

```powershell
bciworkbench validate examples/p300_synthetic.yml
bciworkbench run examples/p300_synthetic.yml
```

Closed-loop cursor task:

```powershell
bciworkbench run examples/mi_cursor_synthetic.yml
```

Adaptive decoder comparison:

```powershell
bciworkbench run examples/mi_synthetic_adaptive.yml
```

StressBench preset matrix:

```powershell
bciworkbench stressbench examples/stressbench_mi.yml
```

StressBench architecture cards:

```powershell
bciworkbench stressbench examples/stressbench_architectures.yml
```

Run comparison:

```powershell
bciworkbench compare runs/<run_a> runs/<run_b>
```

## Configuration Model

Experiments are YAML files with these top-level sections:

| Section | Purpose |
| --- | --- |
| `source` | Synthetic, offline, or replay signal source |
| `pipeline` | Windowing, feature extraction, and decoder steps |
| `task` | Classification or closed-loop task definition |
| `adaptation` | Optional decoder/prediction update policy |
| `metrics` | Requested metrics for reporting |
| `metadata` | User-defined experiment metadata |

Example pipeline:

```yaml
source:
  type: synthetic_motor_imagery
  duration_s: 120
  sampling_rate: 250

pipeline:
  - type: window
    length_s: 1.5
    offset_s: 0.3
  - type: bandpower
  - type: decoder
    estimator: lda
    calibration_fraction: 0.6

task:
  type: motor_imagery_classification
  classes: [left, right]
```

## Run Artifacts

Every run writes a self-contained directory under `runs/<run_id>/`.

Core artifacts:

- `resolved_config.json`
- `ontology_schema.json`
- `graph.json`
- `telemetry.jsonl`
- `channel_schema.json`
- `source_metadata.json`
- `metrics.json`
- `events.csv`
- `windows.csv`
- `features.csv`
- `predictions.csv`
- `model/model_card.json`
- `model/decoder.pkl`
- `provenance.json`
- `report.html`

Replay runs also write:

- `latency_trace.csv`
- `latency_trace.json`
- `stream_health.json`

Closed-loop task runs also write:

- `task_metrics.json`
- `task_states.csv`
- `feedback.csv`

Adaptive runs also write:

- `adaptation_metrics.json`
- `adaptation.jsonl`
- `predictions_before_adaptation.csv`

## StressBench

StressBench evaluates architecture robustness under controlled stressors. It can
run a custom base config or built-in architecture cards.

Built-in architecture cards:

- `mi_bandpower_lda`
- `mi_covariance_pyriemann_mdm`
- `p300_erp_lda`

Built-in stressor presets include clean, low SNR, blink contamination, muscle
noise, channel dropout, electrode shift, session drift, marker jitter, fatigue,
and delayed feedback.

StressBench reports include:

- clean-relative robustness
- weakest preset
- largest drop from clean
- decoder latency
- calibration efficiency
- composite StressBench score
- per-run links and errors or skipped optional architectures

## Documentation

- [Ontology Guide](docs/ontology.md)
- [Source Adapter Guide](docs/source_adapters.md)
- [Simulation Realism Guide](docs/simulation_realism.md)
- [Report Interpretation Guide](docs/reports.md)
- [Non-Goals And Limitations](docs/limitations.md)

## Current Boundaries

The SDK currently uses a deterministic linear runtime. Live LSL and BrainFlow
sources are planned but not implemented in this branch. Synthetic sources and
closed-loop task models are explicit engineering models for controlled testing;
they are not validated physiological models.
