# BCI Workbench

BCI Workbench is a Python SDK for defining, simulating, replaying, evaluating,
and reporting complete brain-computer interface architectures with one
consistent experiment contract.

The SDK is built around a practical systems promise:

> Define a BCI architecture once, then run it against synthetic subjects, public
> datasets, recorded streams, replayed real-time streams, closed-loop tasks, and
> robustness stressors while preserving comparable ontology, timing, metrics,
> provenance, and reports.

BCI Workbench is not a replacement for MNE, MOABB, pyRiemann, BrainFlow, LSL,
Braindecode, Gymnasium, or other mature neurotechnology libraries. It is the
integration and reproducibility layer between those tools: the place where an
architecture becomes a measurable system rather than a loose notebook pipeline.

> Status: pre-1.0 research and engineering SDK. The current implementation is
> suitable for controlled experiments, synthetic stress testing, replay timing
> evaluation, closed-loop task simulation, and reproducible report generation.
> It is not a clinical system, diagnostic system, therapeutic product, or
> medical-device implementation.

## What The SDK Is For

BCI Workbench is designed for teams that need to evaluate BCI systems as systems.
Offline decoder accuracy is necessary, but it is not enough. A BCI architecture
also has to survive drift, artifacts, timing jitter, delayed feedback,
calibration limits, closed-loop control demands, and integration boundaries
between source data, features, decoders, tasks, adaptation, and reporting.

The SDK focuses on questions such as:

- Can the same architecture run on synthetic data, public datasets, and replayed
  recordings without rewriting the pipeline?
- Are cue, target, response, decoded intent, feedback, and adaptation events
  represented as different concepts instead of collapsed into one label column?
- Are sample timestamps, marker timestamps, packet arrival timestamps, and
  decoder latency preserved separately?
- Does a decoder that performs well offline still perform well when connected
  to a closed-loop control task?
- Which stressor breaks an architecture first: noise, drift, artifacts, marker
  jitter, channel dropout, electrode shift, fatigue, or feedback delay?
- Can a run be audited from structured artifacts instead of reconstructed from
  console output and ad hoc plots?

BCI Workbench treats those questions as first-class SDK behavior.

## Core Capabilities

The current implementation includes:

| Capability | Implemented Surface |
| --- | --- |
| Experiment configuration | YAML experiment specs with schema validation and JSON Schema export. |
| Ontology | Typed models for signals, events, channels, windows, features, intents, task states, feedback, and adaptation updates. |
| Runtime | Deterministic linear graph execution with node setup, process, teardown, and telemetry artifacts. |
| Synthetic sources | Motor imagery and P300 generators with subject, session, artifact, drift, timing, and stressor controls. |
| Offline sources | MNE Raw FIF and MOABB BNCI2014_001 adapters behind optional extras. |
| Replay | XDF replay source with fastest, real-time, scaled, and stepped scheduler modes. |
| Features | Windowing, bandpower, ERP feature bins, and covariance features. |
| Decoders | sklearn-style LDA/logistic regression, nearest-centroid fallback, and optional pyRiemann MDM. |
| Closed-loop tasks | `cursor_1d` task with feedback packets, task states, and closed-loop metrics. |
| Adaptation | No-op, supervised batch, confidence-gated, and drift-triggered adaptation interfaces. |
| StressBench | Stressor preset matrices, architecture cards, robustness scoring, aggregate CSV/JSON, and HTML reports. |
| Reporting | Per-run artifacts, provenance, model cards, telemetry, comparison reports, and StressBench reports. |

Live LSL and BrainFlow sources are planned but intentionally skipped in this
branch. Replay support was implemented first so timing semantics could be tested
before live hardware introduces nondeterministic failure modes.

## Why Ontology Comes First

BCI systems fail when semantic boundaries are vague. In many pipelines, a cue
label, target class, user response, latent intent, decoded intent, feedback
action, and training label all become a single string. That is convenient for a
classifier, but it is not good enough for system evaluation.

BCI Workbench uses an explicit ontology so each component can preserve the
meaning of the data it emits:

- `ChannelSchema` records channel names, types, units, sampling rate, reference,
  montage information, bad channels, quality fields, and channel metadata.
- `Event` records typed events with onset, duration, sample index, clock domain,
  confidence, source, target, and metadata.
- `SignalPacket` carries channel x sample signal arrays, timestamps, clock
  domain, source id, events, quality metadata, and source metadata.
- `WindowPacket` represents trial or event-aligned signal windows with labels
  and context.
- `FeaturePacket` carries extracted features and feature schema information.
- `IntentPacket` represents decoder output with intent, posterior, confidence,
  latency, decoder id, window id, task state id, and optional label.
- `TaskStatePacket` records closed-loop state, observation, target, reward,
  completion, success, and metadata.
- `FeedbackPacket` records what action or feedback was delivered, when it was
  rendered, and what delay was modeled or measured.
- `AdaptationPacket` records decoder or policy updates with input windows,
  labels, parameter-change summaries, and before/after metrics.

This ontology is intentionally narrower than the full BCI universe. It covers
the current wedge: EEG motor imagery, P300, synthetic sources, offline dataset
adapters, XDF replay, cursor control, adaptation experiments, and StressBench.
Future additions should extend the ontology only when the new fields can be
tested and reported.

## Architecture

The conceptual architecture is:

```text
source
  -> synchronization and timing
  -> windowing
  -> feature extraction
  -> decoder
  -> optional adaptation
  -> optional task environment
  -> feedback
  -> metrics and reports
```

The current runtime is a deterministic linear graph. That is a deliberate
engineering choice for this milestone: it keeps execution reproducible while
preserving the same node and packet boundaries needed for replay and future live
sources.

Implemented execution paths include:

```text
synthetic motor imagery
  -> trial windows
  -> bandpower features
  -> decoder
  -> metrics and report

synthetic P300 oddball
  -> event windows
  -> ERP features
  -> decoder
  -> metrics and report

XDF replay
  -> replay scheduler
  -> existing pipeline
  -> latency trace, stream health, metrics, and report

motor imagery decoder
  -> cursor_1d task
  -> feedback packets
  -> closed-loop task metrics and report

decoder predictions
  -> adaptation node
  -> update log
  -> adaptation stability metrics and report

StressBench architecture cards
  -> stressor preset matrix
  -> per-run artifacts
  -> aggregate robustness report
```

## Package Layout

The repository is organized around stable contracts rather than one-off scripts:

```text
bciworkbench/
  adaptation/       adaptation adapters, metrics, and factories
  cli/              command-line entry point
  decoders/         decoder adapter contracts and implementations
  eval/             decoder and task metric helpers
  graph/            runtime context, nodes, telemetry, and replay scheduler
  ontology/         schemas, packets, events, channels, and timing models
  sim/              subject/session profiles and domain randomization
  sources/          synthetic, offline, and replay source adapters
  tasks/            closed-loop task environments and user model
  transforms/       windowing and feature extraction

docs/               detailed implementation guides
examples/           runnable experiment and StressBench configs
tests/              unit and integration-style regression tests
```

## Installation

BCI Workbench requires Python 3.11 or newer.

For local development:

```powershell
pip install -e ".[dev]"
```

For a fuller local environment with classical ML, offline, replay, and optional
Riemannian adapters:

```powershell
pip install -e ".[dev,sklearn,mne,moabb,xdf,pyriemann]"
```

Optional extras:

| Extra | Enables |
| --- | --- |
| `bciworkbench[dev]` | Test dependencies. |
| `bciworkbench[sklearn]` | sklearn LDA and logistic regression decoder paths. |
| `bciworkbench[mne]` | MNE Raw FIF source adapter. |
| `bciworkbench[moabb]` | MOABB BNCI2014_001 source adapter. |
| `bciworkbench[xdf]` | Real `.xdf` replay through `pyxdf`. |
| `bciworkbench[pyriemann]` | pyRiemann MDM decoder architecture card. |

Optional integrations are guarded. If a config requests an adapter whose extra
is not installed, the SDK raises a direct install hint instead of failing during
package import.

## Ten-Minute Quickstart

Validate and run a deterministic synthetic motor imagery experiment:

```powershell
bciworkbench validate examples/mi_synthetic.yml
bciworkbench run examples/mi_synthetic.yml
```

The run command prints a generated run directory and report path:

```text
run_id: 20260617T000000Z-mi-synthetic-baseline
run_dir: runs/20260617T000000Z-mi-synthetic-baseline
accuracy: 0.875
report: runs/20260617T000000Z-mi-synthetic-baseline/report.html
```

The exact run id and metrics depend on the current timestamp and config seed.
The important point is that the run directory is self-contained: the report is
human-readable, and the JSON/CSV artifacts are the source of truth.

Run the test suite:

```powershell
pytest
```

Generate the experiment schema:

```powershell
bciworkbench schema experiment
```

Generate the ontology schema fragments:

```powershell
bciworkbench schema ontology
```

## Python SDK Entry Point

The current public Python surface is intentionally small:

```python
from bciworkbench import Experiment, ExperimentSpec, load_experiment_spec

spec = load_experiment_spec("examples/mi_synthetic.yml")
experiment = Experiment(spec)
result = experiment.run()

print(result.run_id)
print(result.run_dir)
print(result.metrics)
```

Equivalent shortcut:

```python
from bciworkbench import Experiment

result = Experiment.from_yaml("examples/mi_synthetic.yml").run()
print(result.run_dir / "report.html")
```

Most users should start with YAML configs and the CLI. The Python API is useful
when running experiments from notebooks, internal benchmark harnesses, or custom
automation.

## CLI Reference

| Command | Purpose |
| --- | --- |
| `bciworkbench validate <config>` | Parse and validate an experiment YAML file. |
| `bciworkbench run <config>` | Execute an experiment and write a run directory. |
| `bciworkbench replay <config>` | Execute a replay-mode experiment with optional scheduler overrides. |
| `bciworkbench report <run_dir>` | Print the report path and formatted metrics JSON for a run. |
| `bciworkbench compare <run_a> <run_b> ...` | Build a comparison report across completed runs. |
| `bciworkbench stressbench <config>` | Run a StressBench matrix and write aggregate reports. |
| `bciworkbench schema experiment` | Print or write the experiment JSON Schema. |
| `bciworkbench schema ontology` | Print or write ontology schema fragments. |

Replay command overrides:

```powershell
bciworkbench replay path\to\replay_config.yml --speed-mode fastest
bciworkbench replay path\to\replay_config.yml --speed-mode real_time
bciworkbench replay path\to\replay_config.yml --speed-mode scaled --speed 2.0
bciworkbench replay path\to\replay_config.yml --speed-mode stepped --speed 1.0
```

Replay configs use `mode: replay` and `source.type: xdf_replay`. The test suite
also exercises a deterministic JSON fixture format for replay behavior. Real
`.xdf` replay requires installing `bciworkbench[xdf]`.

## Example Workflows

Synthetic motor imagery baseline:

```powershell
bciworkbench validate examples/mi_synthetic.yml
bciworkbench run examples/mi_synthetic.yml
```

Synthetic motor imagery with logistic regression:

```powershell
bciworkbench run examples/mi_synthetic_logistic.yml
```

Synthetic P300 oddball:

```powershell
bciworkbench validate examples/p300_synthetic.yml
bciworkbench run examples/p300_synthetic.yml
```

Closed-loop cursor task:

```powershell
bciworkbench run examples/mi_cursor_synthetic.yml
```

Adaptive decoder experiment:

```powershell
bciworkbench run examples/mi_synthetic_adaptive.yml
```

MOABB BNCI2014_001 offline config:

```powershell
pip install -e ".[moabb,sklearn]"
bciworkbench validate examples/mi_moabb_bnci2014_001.yml
bciworkbench run examples/mi_moabb_bnci2014_001.yml
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

## Experiment Configuration

Experiments are YAML files with explicit top-level sections:

| Section | Purpose |
| --- | --- |
| `schema_version` | Version of the experiment config contract. |
| `name` | Human-readable experiment name used in generated run ids. |
| `paradigm` | Paradigm label, such as `motor_imagery` or `p300`. |
| `mode` | Execution mode: `synthetic`, `offline`, `replay`, or future `live`. |
| `random_seed` | Seed used by deterministic synthetic and runtime components. |
| `output_dir` | Root directory for generated run artifacts. |
| `source` | Source adapter type and source-specific parameters. |
| `pipeline` | Ordered window, feature, and decoder steps. |
| `task` | Classification task or closed-loop task definition. |
| `adaptation` | Optional adaptation policy. Defaults to `none`. |
| `metrics` | Requested metric labels for report readability. |
| `metadata` | User-defined run metadata carried into resolved config artifacts. |

Minimal synthetic motor imagery config:

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

adaptation:
  type: none

metrics:
  - accuracy
  - balanced_accuracy
  - mean_confidence
  - calibration_time_s
```

The runtime expects the current pipeline shape to contain:

1. A `window` step.
2. A feature step: `bandpower`, `erp_features`, or `covariance`.
3. A `decoder` step.

That shape is intentionally constrained in the current milestone so reports,
tests, and artifact contracts remain stable while the SDK grows.

## Source Adapters

BCI Workbench source adapters are responsible for converting external or
synthetic data into ontology packets without discarding timing and provenance.

| Source type | Mode | Description |
| --- | --- | --- |
| `synthetic_motor_imagery` | synthetic | EEG-like motor imagery source with trial events and class-specific effects. |
| `synthetic_p300` | synthetic | Oddball/P300 source with target and non-target stimulus events. |
| `mne_raw` | offline | MNE Raw FIF source adapter with annotation-derived events. |
| `moabb` | offline | MOABB BNCI2014_001 adapter with subject/session/run metadata. |
| `xdf_replay` | replay | XDF or fixture replay source with signal and marker timing preservation. |

Source adapters should preserve:

- sampling rate
- channel names and units
- channel order
- bad-channel and quality metadata where available
- event names and event types
- sample timestamps
- marker timestamps
- source-specific provenance
- optional dependency requirements

The goal is not to hide MNE, MOABB, or XDF behind a vague wrapper. The goal is to
make their outputs comparable under BCI Workbench's artifact and timing model.

## Timing Model

Timing is a first-class part of the SDK contract. BCI Workbench distinguishes:

- sample-derived timestamps
- recording timestamps
- replay arrival timestamps
- local system timestamps
- simulated task timestamps
- decoder processing latency
- feedback delay

Replay sources preserve original marker and signal timing separately from
simulated packet arrival. This prevents a common reporting failure: treating a
recorded stream as if offline file order were equivalent to real-time delivery.

Replay scheduler modes:

| Mode | Behavior |
| --- | --- |
| `fastest` | Deterministic replay without sleeping for original timing. |
| `real_time` | Packet arrival follows original timing. |
| `scaled` | Packet arrival is faster or slower than original timing. |
| `stepped` | Fixed-step progression for packet-by-packet debugging. |

Replay runs write:

- `latency_trace.csv`
- `latency_trace.json`
- `stream_health.json`

Use these artifacts when diagnosing backlog, timing drift, packet arrival
behavior, or marker alignment.

## Synthetic Simulation

The synthetic sources are designed for controlled BCI engineering experiments,
not for claiming validated physiology. Their value is that they expose known
ground truth, controllable stressors, and deterministic seeds.

Current synthetic coverage:

| Area | Implemented Controls |
| --- | --- |
| Background signal | EEG-like rhythms, channel-specific noise, SNR, and sampling rate. |
| Motor imagery | Class-specific trial effects, subject separability, mu/beta effects, and trial labels. |
| P300 | Target/non-target events, ERP amplitude, ERP latency, and attention/fatigue modulation. |
| Subject profile | Alpha/beta peaks, motor imagery vividness, P300 amplitude/latency, attention, fatigue. |
| Session profile | Amplitude drift, spectral drift, spatial drift, electrode shift, artifacts, dropout, jitter, delay. |
| Artifacts | Blink events, muscle noise, line noise, channel dropout, and quality/stressor metadata. |
| Domain randomization | Bounded sampling of source parameter overrides for robustness sweeps. |

Simulation artifacts label realism where applicable. The current MI and P300
sources are paradigm-aware engineering simulations around levels 2/3, while
closed-loop task artifacts add level 5 system behavior. They are not validated
human-in-the-loop physiological models.

## Feature Pipeline And Decoders

Supported feature steps:

| Feature step | Use case |
| --- | --- |
| `bandpower` | Motor imagery and spectral baselines. |
| `erp_features` | P300/event-locked ERP decoding. |
| `covariance` | Riemannian-style architecture cards and covariance baselines. |

Supported decoder paths:

| Decoder path | Notes |
| --- | --- |
| `lda` | sklearn LDA when sklearn is installed, with fallback behavior guarded by adapter code. |
| `logistic_regression` | sklearn logistic regression path. |
| nearest centroid | Deterministic lightweight fallback used when configured paths allow it. |
| `pyriemann_mdm` | Optional pyRiemann MDM decoder for covariance features. |

Decoder outputs are stored as `IntentPacket`-like prediction records and written
to `predictions.csv`. Reports include accuracy, balanced accuracy, confidence,
calibration time, decoder identity, latency, and model-card metadata where
available.

## Closed-Loop Task Evaluation

BCI Workbench includes a `cursor_1d` task environment for whole-system testing.
It is deliberately simple: the point is not to build a game, but to expose the
gap between decoder metrics and control metrics.

The cursor task records:

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

Closed-loop task runs write:

- `task_metrics.json`
- `task_states.csv`
- `feedback.csv`

The most important comparison is often:

```text
balanced_accuracy - target_acquisition_rate = decoder_task_gap
```

A decoder can score well on held-out windows while still performing poorly in a
task because confidence thresholds, delay, dwell time, and false activations
change control outcomes.

## Adaptation

Adaptation is represented as an optional graph node after the decoder and before
closed-loop task evaluation. This preserves a clean comparison between static
and adaptive systems.

Implemented adapters:

| Adapter | Purpose |
| --- | --- |
| `none` | No adaptation node. |
| `noop` | Explicit no-op adapter for controlled static baselines. |
| `supervised_batch` | Deterministic recalibration from labeled batches. |
| `confidence_gated` | Batch updates gated by prediction confidence. |
| `drift_triggered` | Update hook triggered by accuracy or confidence floors. |

Adaptive runs write:

- `adaptation_metrics.json`
- `adaptation.jsonl`
- `predictions_before_adaptation.csv`

Metrics include update count, changed prediction rate, before/after accuracy,
before/after balanced accuracy, confidence change, and catastrophic update
warning. These adapters are controlled research interfaces, not production
online-learning algorithms.

## StressBench

StressBench is the architecture robustness suite included with BCI Workbench. It
complements offline dataset benchmarking by testing whole architectures under
controlled stressor presets.

Built-in architecture cards:

| Card | Pipeline |
| --- | --- |
| `mi_bandpower_lda` | Synthetic MI, bandpower features, LDA decoder. |
| `mi_covariance_pyriemann_mdm` | Synthetic MI, covariance features, optional pyRiemann MDM. |
| `p300_erp_lda` | Synthetic P300, ERP features, LDA decoder. |

Built-in stressor presets:

| Preset | Intended stressor |
| --- | --- |
| `clean` | Reference condition. |
| `low_snr` | Lower signal-to-noise ratio. |
| `high_blink` | Increased blink artifacts. |
| `muscle_noise` | Increased EMG-like contamination. |
| `channel_dropout` | Channel loss/dropout events. |
| `electrode_shift` | Spatial pattern perturbation. |
| `session_drift` | Drift across the synthetic session. |
| `jittery_markers` | Marker timing jitter. |
| `fatigue` | Reduced attention or increasing fatigue. |
| `delayed_feedback` | Feedback delay for closed-loop stress. |

StressBench writes:

- `stressbench_summary.json`
- `stressbench_summary.csv`
- `stressbench_aggregates.csv`
- `stressbench_report.html`

StressBench reports include:

- clean-relative robustness
- weakest preset
- largest drop from clean
- decoder latency
- calibration efficiency
- composite StressBench score
- per-architecture scores
- skipped optional architectures and install hints
- per-run links back to individual run directories

## Run Artifact Contract

Every run writes a self-contained directory under `runs/<run_id>/`.

Core artifacts:

| Artifact | Purpose |
| --- | --- |
| `resolved_config.json` | Parsed experiment config used for the run. |
| `ontology_schema.json` | Ontology schema snapshot for the run. |
| `graph.json` | Runtime node list and graph edges. |
| `telemetry.jsonl` | Per-node setup/process/teardown timing and status records. |
| `channel_schema.json` | Channel metadata and sampling metadata. |
| `source_metadata.json` | Source provenance, simulation settings, and stressors. |
| `metrics.json` | Decoder, task, adaptation, and system metrics. |
| `events.csv` | Event table. |
| `windows.csv` | Window/trial table. |
| `features.csv` | Feature matrix table. |
| `predictions.csv` | Decoder output table. |
| `model/model_card.json` | Decoder metadata and training summary. |
| `provenance.json` | Environment, seed, and SDK provenance metadata. |
| `report.html` | Human-readable report generated from structured artifacts. |

Replay-specific artifacts:

| Artifact | Purpose |
| --- | --- |
| `latency_trace.csv` | Tabular packet timing and latency trace. |
| `latency_trace.json` | JSON representation of replay timing rows. |
| `stream_health.json` | Replay health, backlog, and stream diagnostics. |

Task-specific artifacts:

| Artifact | Purpose |
| --- | --- |
| `task_metrics.json` | Closed-loop task metric summary. |
| `task_states.csv` | Step-by-step task state records. |
| `feedback.csv` | Feedback/action records emitted by the task loop. |

Adaptation-specific artifacts:

| Artifact | Purpose |
| --- | --- |
| `adaptation_metrics.json` | Adaptation summary and stability metrics. |
| `adaptation.jsonl` | One adaptation update record per line. |
| `predictions_before_adaptation.csv` | Baseline prediction stream before adaptation changes. |

Reports are views over these files. The artifacts are the durable interface for
analysis, regression testing, internal dashboards, and future SDK integrations.

## Report Interpretation

A run report should be read in this order:

1. Confirm the source, paradigm, mode, random seed, and resolved config.
2. Check graph structure and node telemetry.
3. Inspect decoder metrics and calibration size.
4. Compare decoder latency against replay or task timing.
5. For closed-loop runs, compare decoder accuracy against task success metrics.
6. For adaptive runs, inspect changed prediction rate and before/after metrics.
7. For replay runs, inspect stream health and latency trace artifacts.
8. For StressBench runs, inspect the weakest preset and largest clean-relative
   drop before looking at the composite score.

The composite StressBench score is useful for comparing architectures inside the
workbench. It should not be interpreted as a clinical, consumer, or hardware
certification metric.

## Current Limitations

The SDK is intentionally pre-1.0 and has clear boundaries:

- The runtime is deterministic and linear, not a fully asynchronous live graph
  scheduler.
- Live LSL and BrainFlow sources are planned but not implemented in this branch.
- Synthetic sources are controlled engineering simulations, not validated human
  physiology.
- Current public dataset support is intentionally narrow.
- Deep-learning decoders are not implemented in the current branch.
- Config fields and artifact names are milestone contracts rather than permanent
  stable public API guarantees.

These boundaries are part of the design. The project is prioritizing ontology,
timing, artifacts, simulation stressors, and reproducible reports before adding
large integration breadth.

## Extension Points

BCI Workbench is structured so new capabilities can be added behind explicit
contracts:

| Extension | Contract |
| --- | --- |
| Source adapter | Emit `SignalPacket` data with channel schema, events, timestamps, quality, and source metadata. |
| Feature transform | Convert `WindowPacket` lists into `FeaturePacket` records without losing window provenance. |
| Decoder adapter | Train and predict through the decoder contract and emit prediction records with confidence and latency. |
| Task environment | Emit task states, feedback records, rewards, success flags, and task metrics. |
| Adaptation adapter | Emit adaptation update packets and stability metrics. |
| StressBench card | Define a reproducible architecture recipe and optional dependency requirements. |
| Report section | Render from structured artifacts rather than private in-memory state. |

New integrations should preserve:

- ontology semantics
- timing domains
- source provenance
- optional dependency guards
- deterministic tests where possible
- artifact compatibility
- clear install hints

## Documentation

Detailed guides:

- [Ontology Guide](docs/ontology.md)
- [Source Adapter Guide](docs/source_adapters.md)
- [Simulation Realism Guide](docs/simulation_realism.md)
- [Report Interpretation Guide](docs/reports.md)
- [Non-Goals And Limitations](docs/limitations.md)

Useful examples:

- [Synthetic Motor Imagery](examples/mi_synthetic.yml)
- [Synthetic Motor Imagery Logistic Regression](examples/mi_synthetic_logistic.yml)
- [Synthetic P300](examples/p300_synthetic.yml)
- [Closed-Loop Cursor Task](examples/mi_cursor_synthetic.yml)
- [Adaptive Motor Imagery](examples/mi_synthetic_adaptive.yml)
- [MOABB BNCI2014_001](examples/mi_moabb_bnci2014_001.yml)
- [StressBench Presets](examples/stressbench_mi.yml)
- [StressBench Architecture Cards](examples/stressbench_architectures.yml)

## Development Verification

Recommended checks before submitting changes:

```powershell
pytest -q
bciworkbench validate examples/mi_synthetic.yml
bciworkbench run examples/mi_synthetic.yml
bciworkbench schema experiment
```

For documentation-only changes, `pytest -q` and `bciworkbench schema experiment`
are usually sufficient to confirm the package still imports and the schema
contract still renders.

## Project Direction

The near-term direction is to keep strengthening the architecture reproducibility
layer:

- tighter ontology validation
- richer simulation validation tests
- stronger golden artifact tests
- broader but guarded optional integrations
- better replay-to-live parity
- clearer report diagnostics
- more StressBench architecture cards

The long-term product shape is a proprietary-grade SDK experience for BCI
architecture engineering: a disciplined contract for defining systems,
executing them across data regimes, measuring them under stress, and producing
artifacts that can be audited by researchers, engineers, and technical
stakeholders.
