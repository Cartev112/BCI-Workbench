# Simulation Realism Guide

The synthetic sources are designed for software architecture testing, controlled
stress sweeps, and report plumbing. They are not validated physiological human
models.

## Simulation Levels

Artifacts label simulation realism with `simulation_level` where applicable.

- `0_plumbing`: shape-correct data for exercising code paths only.
- `1_spectral`: EEG-like rhythms or spectra, but no task physiology.
- `2_paradigm`: task events and class effects are present.
- `3_subject_session`: subject and session parameters are explicit.
- `4_source_space`: source-space or head-model realism. Not implemented yet.
- `5_closed_loop`: task feedback and state affect whole-system evaluation.

Current synthetic MI and P300 sources are labeled around level 2/3. The
`cursor_1d` task metrics are level 5 closed-loop artifacts, but the source signal
itself is still not a validated human-in-the-loop generator.

## Subject And Session Parameters

`SubjectProfile` captures stable synthetic subject parameters such as:

- alpha and beta peak frequencies
- motor imagery vividness
- P300 amplitude and latency
- attention and fatigue rate

`SessionProfile` captures recording-specific stressors such as:

- amplitude drift
- spectral drift
- spatial covariance drift
- electrode shift
- blink rate and amplitude
- muscle noise
- channel dropout probability
- marker jitter
- feedback delay

These are logged in `source_metadata.json` so robustness failures can be traced
back to ground-truth stressor settings.

## StressBench Presets

Built-in presets include clean, low SNR, high blink, muscle noise, channel
dropout, electrode shift, session drift, jittery markers, fatigue, and delayed
feedback.

Run the basic MI stress suite:

```powershell
bciworkbench stressbench examples/stressbench_mi.yml
```

Run architecture cards:

```powershell
bciworkbench stressbench examples/stressbench_architectures.yml
```

StressBench scores include robustness against clean, decoder latency, and
calibration efficiency. A good StressBench score is useful for comparing
architectures inside this workbench; it is not a claim of real-world clinical or
consumer-device performance.

## Domain Randomization

`DomainRandomization` samples source override dictionaries for synthetic sweeps.
Use it when you need randomized but bounded stressor settings. Keep the sampled
settings in run metadata so results remain auditable.
