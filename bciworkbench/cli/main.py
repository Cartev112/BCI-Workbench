from __future__ import annotations

import argparse
import json
from pathlib import Path

from bciworkbench.experiment import Experiment
from bciworkbench.ontology.schemas import ConfigError, load_experiment_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bciworkbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate an experiment YAML file.")
    validate_parser.add_argument("config")

    run_parser = subparsers.add_parser("run", help="Run an experiment YAML file.")
    run_parser.add_argument("config")

    report_parser = subparsers.add_parser("report", help="Print a run report path and metrics summary.")
    report_parser.add_argument("run_dir")

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

    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

