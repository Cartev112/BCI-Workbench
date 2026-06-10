from __future__ import annotations

import argparse
import json
from pathlib import Path

from bciworkbench.experiment import Experiment
from bciworkbench.ontology.schema_export import experiment_json_schema, ontology_json_schema
from bciworkbench.ontology.schemas import ConfigError, load_experiment_spec
from bciworkbench.reports import compare_runs
from bciworkbench.stressbench import run_stressbench


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bciworkbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate an experiment YAML file.")
    validate_parser.add_argument("config")

    run_parser = subparsers.add_parser("run", help="Run an experiment YAML file.")
    run_parser.add_argument("config")

    replay_parser = subparsers.add_parser("replay", help="Run a replay-mode experiment with optional speed overrides.")
    replay_parser.add_argument("config")
    replay_parser.add_argument("--speed-mode", choices=["fastest", "real_time", "scaled", "stepped"])
    replay_parser.add_argument("--speed", type=float)

    report_parser = subparsers.add_parser("report", help="Print a run report path and metrics summary.")
    report_parser.add_argument("run_dir")

    compare_parser = subparsers.add_parser("compare", help="Compare completed run directories.")
    compare_parser.add_argument("run_dirs", nargs="+")
    compare_parser.add_argument("--output-dir", "-o", default="runs")

    stressbench_parser = subparsers.add_parser("stressbench", help="Run a StressBench preset matrix.")
    stressbench_parser.add_argument("config")

    schema_parser = subparsers.add_parser("schema", help="Print a JSON Schema.")
    schema_parser.add_argument("kind", choices=["experiment", "ontology"])
    schema_parser.add_argument("--output", "-o")

    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            spec = load_experiment_spec(args.config)
            print(f"valid: {spec.name}")
            return 0

        if args.command == "run":
            result = Experiment.from_yaml(args.config).run()
            print(f"run_id: {result.run_id}")
            print(f"run_dir: {result.run_dir}")
            print(f"accuracy: {result.metrics.get('accuracy')}")
            print(f"report: {result.run_dir / 'report.html'}")
            return 0

        if args.command == "replay":
            spec = _load_replay_spec(args.config, speed_mode=args.speed_mode, speed=args.speed)
            result = Experiment(spec).run()
            print(f"run_id: {result.run_id}")
            print(f"run_dir: {result.run_dir}")
            print(f"accuracy: {result.metrics.get('accuracy')}")
            print(f"stream_health: {result.run_dir / 'stream_health.json'}")
            print(f"latency_trace: {result.run_dir / 'latency_trace.csv'}")
            print(f"report: {result.run_dir / 'report.html'}")
            return 0

        if args.command == "report":
            run_dir = Path(args.run_dir)
            metrics_path = run_dir / "metrics.json"
            report_path = run_dir / "report.html"
            if not metrics_path.exists():
                raise FileNotFoundError(f"missing metrics file: {metrics_path}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"report: {report_path}")
            print(json.dumps(metrics, indent=2, sort_keys=True))
            return 0

        if args.command == "compare":
            result = compare_runs(args.run_dirs, output_dir=args.output_dir)
            print(f"comparison_dir: {result['comparison_dir']}")
            print(f"runs: {result['run_count']}")
            best_run = result.get("best_run") or {}
            print(f"best_run: {best_run.get('run_id')}")
            print(f"report: {result['comparison_dir'] / 'comparison_report.html'}")
            return 0

        if args.command == "stressbench":
            result = run_stressbench(args.config)
            print(f"summary_dir: {result.summary_dir}")
            print(f"runs: {len(result.rows)}")
            print(f"robustness_score: {result.robustness.get('robustness_score')}")
            print(f"weakest_preset: {result.robustness.get('weakest_preset')}")
            print(f"report: {result.summary_dir / 'stressbench_report.html'}")
            return 0

        if args.command == "schema":
            schema = experiment_json_schema() if args.kind == "experiment" else ontology_json_schema()
            rendered = json.dumps(schema, indent=2, sort_keys=True)
            if args.output:
                Path(args.output).write_text(rendered + "\n", encoding="utf-8")
                print(f"schema: {args.output}")
            else:
                print(rendered)
            return 0

    except (ConfigError, FileNotFoundError, ValueError, TypeError) as exc:
        print(f"error: {exc}")
        return 2

    return 1


def _load_replay_spec(config: str, speed_mode: str | None = None, speed: float | None = None):
    spec = load_experiment_spec(config)
    if spec.mode != "replay":
        raise ConfigError("replay command requires mode: replay")
    params = dict(spec.source.params)
    if speed_mode is not None:
        params["speed_mode"] = speed_mode
    if speed is not None:
        if speed <= 0:
            raise ConfigError("--speed must be positive")
        params["speed"] = speed
    return type(spec)(
        schema_version=spec.schema_version,
        name=spec.name,
        paradigm=spec.paradigm,
        mode=spec.mode,
        source=type(spec.source)(type=spec.source.type, params=params),
        pipeline=spec.pipeline,
        task=spec.task,
        metrics=spec.metrics,
        random_seed=spec.random_seed,
        output_dir=spec.output_dir,
        metadata=spec.metadata,
    )


if __name__ == "__main__":
    raise SystemExit(main())
