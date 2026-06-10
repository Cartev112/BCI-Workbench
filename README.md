# BCI Workbench

BCI Workbench is a Python SDK for defining, simulating, replaying, evaluating,
and reporting brain-computer interface pipelines with consistent runtime
artifacts.

The SDK is built around one operational contract: define a BCI architecture
once, then run it against synthetic subjects, public datasets, recorded streams,
and replayed real-time streams while preserving comparable ontology, timing,
metrics, provenance, and reports.

BCI Workbench is intended for architecture research and engineering: comparing
signal sources, feature pipelines, decoders, adaptation policies, timing
behavior, closed-loop tasks, and stressor robustness under a single reproducible
artifact model.

> Status: pre-1.0 research and engineering workbench. The current implementation
> is suitable for controlled experiments, synthetic stress tests, replay timing
> evaluation, and reproducible reports. It is not a clinical system, diagnostic
> system, therapeutic product, or medical-device implementation.

## Why It Exists

Most BCI evaluation stacks are strong in one narrow region: offline dataset
benchmarking, real-time acquisition, signal processing experiments, or task
simulation. BCI Workbench is the integration layer between those regions.

The project focuses on questions that offline accuracy alone does not answer:

- Does an architecture remain usable under drift, noise, artifacts, marker
  jitter, channel dropout, or feedback delay?
- Are source timestamps, marker timestamps, packet arrival times, decoder
  latency, and feedback timing kept separate and inspectable?
- Does closed-loop task performance diverge from offline decoder accuracy?
- Can static and adaptive systems be compared under the same stressor settings?
- Can a run be reproduced and audited from saved config, graph, metrics,
  telemetry, model card, provenance, and report artifacts?

The SDK therefore treats reports and artifacts as first-class outputs, not as
afterthoughts.

## What It Provides

BCI Workbench packages the core components needed to evaluate BCI systems beyond
offline classifier accuracy:

- Typed ontology for signals, events, windows, features, intents, task states,
  feedback packets, and adaptation updates.
- YAML experiment configuration and JSON Schema export.
- Deterministic graph runtime with per-node setup/process/teardown telemetry.
- Synthetic motor imagery and P300 sources with explicit subject, session,
  artifact, drift, and timing parameters.
- Offline adapters for MNE Raw FIF and MOABB BNCI2014_001.
- XDF replay source with deterministic scheduler modes and stream health
  telemetry.
- Windowing, bandpower, ERP, and covariance feature transforms.
- Decoder adapters for sklearn-style estimators and optional pyRiemann MDM.
- Closed-loop `cursor_1d` task evaluation.
- Adaptation interfaces with update logs and stability metrics.
- StressBench robustness suites for stressor presets and architecture cards.
- Reproducible run directories containing JSON, CSV, JSONL, model, provenance,
  and HTML report artifacts.

## Architecture Overview

The conceptual lifecycle is:

```text
source -> synchronization/timing -> windowing -> features -> decoder
       -> optional adaptation -> optional task -> feedback -> reports
```

The current runtime is a deterministic linear graph. The implementation keeps
the graph simple while preserving the conceptual node boundaries needed for
future live sources and richer schedulers.

Implemented paths include:

```text
synthetic motor imagery -> trial windows -> bandpower -> decoder -> metrics/report
synthetic P300 oddball -> event windows -> ERP features -> decoder -> metrics/report
XDF replay -> replay scheduler -> existing decoder pipeline -> latency/backlog report
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
| `bciworkbench[moabb]` | MOABB dataset adapter |
| `bciworkbench[pyriemann]` | pyRiemann MDM decoder |
| `bciworkbench[xdf]` | Real `.xdf` replay through `pyxdf` |

Optional dependencies are guarded. If an adapter is requested without its extra,
the SDK raises a clear install hint instead of failing at import time.

## Ten-Minute Quickstart

Run a deterministic synthetic motor imagery experiment:

```powershell
bciworkbench validate examples/mi_synthetic.yml
bciworkbench run examples/mi_synthetic.yml
```

The run command prints the generated report path:

```text
runs/<run_id>/report.html
```

Run the test suite:

```powershell
pytest
```

Generate the current experiment JSON Schema:

```powershell
bciworkbench schema experiment
```

## CLI Surface

| Command | Purpose |
| --- | --- |
| `bciworkbench validate <config>` | Validate an experiment YAML config. |
| `bciworkbench schema experiment` | Print the current experiment JSON Schema. |
| `bciworkbench schema ontology` | Print ontology artifact schema fragments. |
| `bciworkbench run <config>` | Execute an experiment and write run artifacts. |
| `bciworkbench replay <config>` | Execute replay-mode configs with optional speed overrides. |
| `bciworkbench report <run_dir>` | Print report path and metrics JSON for a run. |
| `bciworkbench compare <run_a> <run_b>` | Build a comparison report across completed runs. |
| `bciworkbench stressbench <config>` | Run a StressBench matrix and write aggregate reports. |

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

Experiments are YAML files with explicit top-level sections:

| Section | Purpose |
| --- | --- |
| `schema_version` | Config format version. |
| `name` | Human-readable run name used in run ids. |
| `paradigm` | Paradigm label such as `motor_imagery` or `p300`. |
| `mode` | `synthetic`, `offline`, `replay`, or future `live`. |
| `source` | Synthetic, offline, or replay signal source. |
| `pipeline` | Windowing, feature extraction, and decoder steps. |
| `task` | Classification or closed-loop task definition. |
| `adaptation` | Optional decoder/prediction update policy. |
| `metrics` | Requested metrics for report readability. |
| `metadata` | User-defined experiment metadata. |

Minimal motor imagery example:

```yaml
schema_version: "0.1"
name: mi_synthetic_baseline
paradigm: motor_imagery
mode: synthetic
random_seed: 7
output_dir: runs

source:
  type: synthetic_motor_imagery
  duration_s: 120
  sampling_rate: 250
  n_channels: 16
  n_trials: 80
  snr_db: -4

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

metrics:
  - accuracy
  - balanced_accuracy
  - mean_confidence
  - calibration_time_s
```

## Sources

| Source type | Mode | Notes |
| --- | --- | --- |
| `synthetic_motor_imagery` | synthetic | Deterministic MI-like EEG with trial and cue events. |
| `synthetic_p300` | synthetic | Oddball source with target and non-target stimulus events. |
| `mne_raw` | offline | MNE Raw FIF conversion with annotation-derived events. |
| `moabb` | offline | Initial MOABB BNCI2014_001 support. |
| `xdf_replay` | replay | Signal and marker streams with replay scheduler telemetry. |

Synthetic sources expose subject/session controls for attention, fatigue, alpha
and beta peaks, P300 amplitude and latency, amplitude drift, spectral drift,
spatial covariance drift, electrode shift, blink artifacts, muscle noise,
channel dropout, marker jitter, and feedback delay.

## Feature And Decoder Pipeline

Supported feature steps:

- `bandpower`: log bandpower features across configurable frequency bands.
- `erp_features`: time-bin ERP features for P300-style windows.
- `covariance`: flattened channel covariance features for Riemannian decoders.

Supported decoder paths:

- sklearn LDA.
- sklearn logistic regression.
- deterministic nearest-centroid fallback.
- optional pyRiemann MDM for covariance features.

Decoder runs produce `IntentPacket`s with intent, confidence, posterior,
latency, window id, decoder id, and label when available.

## Replay Timing

`xdf_replay` separates original signal/marker timestamps from simulated packet
arrival timing. This matters because a real-time system can fail due to backlog
or packet timing even when the offline signal content is unchanged.

Replay modes:

- `fastest`: deterministic replay without real-time delays.
- `real_time`: packet arrival follows original sample timing.
- `scaled`: packet arrival is faster or slower than original timing.
- `stepped`: fixed step duration for packet-by-packet debugging.

Replay runs write `latency_trace.csv`, `latency_trace.json`, and
`stream_health.json`.

## Closed-Loop Tasks

The current task runtime includes `cursor_1d`, a simple one-dimensional cursor
environment driven by decoder intents.

It records:

- cursor position
- target position
- reward
- dwell state
- success state
- feedback delay
- false activations
- time to target
- path efficiency
- target acquisition rate

The key metric comparison is often `balanced_accuracy` versus
`target_acquisition_rate`. A decoder can classify windows correctly and still
perform poorly as a control system because confidence, delay, dwell time, and
false activations change task success.

## Adaptation

Adaptation is represented as an optional graph node after the decoder and before
task evaluation. Current adapters operate on prediction streams and emit
`AdaptationPacket`s.

Implemented adapters:

- `none`: no adaptation node.
- `noop`: explicit no-op adapter for static comparisons.
- `supervised_batch`: deterministic batch recalibration using labeled chunks.
- `confidence_gated`: batch updates using only sufficiently confident examples.
- `drift_triggered`: update hook triggered by accuracy or confidence floors.

Adaptive runs write:

- `adaptation_metrics.json`
- `adaptation.jsonl`
- `predictions_before_adaptation.csv`

Adaptation metrics include update count, changed prediction rate, before/after
accuracy, before/after balanced accuracy, confidence change, and catastrophic
update warning.

## Run Artifacts

Every run writes a self-contained directory under `runs/<run_id>/`.

Core artifacts:

| Artifact | Purpose |
| --- | --- |
| `resolved_config.json` | Parsed experiment config used for the run. |
| `ontology_schema.json` | Ontology schema snapshot. |
| `graph.json` | Runtime nodes and edges. |
| `telemetry.jsonl` | Per-node timing and status records. |
| `channel_schema.json` | Channel metadata and sampling metadata. |
| `source_metadata.json` | Source provenance, simulation settings, and stressors. |
| `metrics.json` | Decoder, task, adaptation, and system metrics. |
| `events.csv` | Event table. |
| `windows.csv` | Trial/window table. |
| `features.csv` | Feature matrix table. |
| `predictions.csv` | Decoder output table. |
| `model/model_card.json` | Decoder metadata. |
| `model/decoder.pkl` | Persisted decoder where supported. |
| `provenance.json` | Environment and random seed metadata. |
| `report.html` | Human-readable run report. |

Additional replay artifacts:

- `latency_trace.csv`
- `latency_trace.json`
- `stream_health.json`

Additional task artifacts:

- `task_metrics.json`
- `task_states.csv`
- `feedback.csv`

Additional adaptation artifacts:

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

Built-in stressor presets:

- `clean`
- `low_snr`
- `high_blink`
- `muscle_noise`
- `channel_dropout`
- `electrode_shift`
- `session_drift`
- `jittery_markers`
- `fatigue`
- `delayed_feedback`

StressBench writes:

- `stressbench_summary.json`
- `stressbench_summary.csv`
- `stressbench_aggregates.csv`
- `stressbench_report.html`

StressBench reports include clean-relative robustness, weakest preset, largest
drop from clean, decoder latency, calibration efficiency, composite StressBench
score, per-architecture scores, per-run links, and errors or skipped optional
architectures.

## Documentation

- [Ontology Guide](docs/ontology.md)
- [Source Adapter Guide](docs/source_adapters.md)
- [Simulation Realism Guide](docs/simulation_realism.md)
- [Report Interpretation Guide](docs/reports.md)
- [Non-Goals And Limitations](docs/limitations.md)

## Extension Points

BCI Workbench is structured so new capabilities can be added behind stable
contracts:

- Add a source adapter that returns `SignalPacket`.
- Add a feature transform that returns `FeaturePacket` lists.
- Add a decoder adapter that returns `DecoderResult`.
- Add a task environment that emits `TaskStatePacket` and `FeedbackPacket`.
- Add an adaptation adapter that emits `AdaptationPacket`.
- Add a StressBench architecture card for repeatable benchmark comparisons.

New integrations should preserve source metadata, clock semantics, optional
dependency guards, and artifact writing conventions.

## Current Boundaries

The SDK currently uses a deterministic linear runtime. Live LSL and BrainFlow
sources are planned but not implemented in this branch. Synthetic sources and
closed-loop task models are explicit engineering models for controlled testing;
they are not validated physiological models.

The project is pre-1.0. Config fields and artifact names should be treated as
milestone contracts rather than permanent public API guarantees. Additive changes
are preferred over changing the meaning of existing artifacts.
