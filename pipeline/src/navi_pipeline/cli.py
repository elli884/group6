"""Small local CLI. Data preparation/validation never loads a VLM."""

import argparse
import json
import sys
from pathlib import Path

from .artifacts import write_json
from .config import load_config, project_path
from .data import prepare_manifest
from .evaluation import compare_runs
from .mission_logs import CASES
from .runner import run_experiment, run_one_image_mock, run_selected_mission_log, validate_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="NAVI-style ground research pipeline")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd(),
                        help="pipeline directory; all configured paths are relative to this")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare a content-hashed dataset manifest")
    prepare.add_argument("--dataset", required=True, help="Project-relative class-folder directory")
    prepare.add_argument("--output", required=True, help="Project-relative output JSONL")
    prepare.add_argument("--taxonomy", choices=["aid18", "eurosat", "folders"], required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--heldout-fraction", type=float, default=0.2)
    prepare.add_argument("--train-fraction", type=float, default=0.0)
    prepare.add_argument("--validation-fraction", type=float, default=0.0)
    validate = commands.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--allow-heldout", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--allow-heldout", action="store_true")
    run.add_argument(
        "--mock-mission-log", nargs="?", const="Forest", default=None, metavar="LABEL",
        help="Ignore each identified class and put LABEL in the mission log the "
             "dialogue prompt receives. Bare flag uses Forest.",
    )
    run.add_argument(
        "--mission-log", choices=[case.case_id for case in CASES], metavar="CASE",
        help="Classify --image with the config model, then ask dialogue from this catalog log.",
    )
    run.add_argument(
        "--image", type=Path,
        help="With --mission-log, the image fed to the config model.",
    )
    mock = commands.add_parser(
        "mock-mission-log",
        help="Load one image, ignore its class, and ask dialogue from a hardcoded label",
    )
    mock.add_argument("--config", required=True)
    mock.add_argument("--image", required=True, type=Path, help="Image file to load")
    mock.add_argument("--label", default="Forest", help="Hardcoded mission-log label")
    compare = commands.add_parser("compare", help="Compare two completed runs on identical images")
    compare.add_argument("--clean", required=True, help="Project-relative run directory")
    compare.add_argument("--other", required=True, help="Project-relative run directory")
    compare.add_argument("--target-label")
    args = parser.parse_args()
    root = args.project_dir.resolve()
    try:
        if not (root / "pyproject.toml").is_file():
            raise ValueError("Run from pipeline/ or pass --project-dir pointing to pipeline/")
        if args.command == "prepare":
            audit_path = project_path(root, args.output + ".audit.json")
            if audit_path.exists():
                raise FileExistsError(f"Refusing to overwrite audit: {audit_path}")
            audit = prepare_manifest(root, args.dataset, args.output, args.taxonomy, args.seed,
                                     args.heldout_fraction, args.train_fraction,
                                     args.validation_fraction)
            write_json(audit_path, audit)
            print(json.dumps(audit, indent=2))
        elif args.command == "compare":
            print(json.dumps(compare_runs(project_path(root, args.clean),
                                          project_path(root, args.other), args.target_label), indent=2))
        else:
            config = load_config(project_path(root, args.config))
            if args.command == "validate":
                selected, _, _ = validate_experiment(config, root, args.allow_heldout)
                print(json.dumps({"valid": True, "selected": len(selected), "split": config.split,
                                  "model_loaded": False}, indent=2))
            elif args.command == "run":
                if args.mission_log and args.mock_mission_log:
                    raise ValueError("Choose either --mission-log or --mock-mission-log")
                if args.mission_log and args.image is None:
                    raise ValueError("--mission-log requires --image")
                if args.image is not None and not args.mission_log:
                    raise ValueError("--image on run is only valid with --mission-log")
                if args.mission_log:
                    run_dir, report = run_selected_mission_log(
                        config, root, args.mission_log, args.image,
                    )
                    print(report)
                    print(f"run: {run_dir.relative_to(root)}")
                else:
                    run_dir = run_experiment(
                        config, root, args.allow_heldout, mock_mission_log=args.mock_mission_log,
                    )
                    if args.mock_mission_log is not None:
                        print(f"mock mission log label: {args.mock_mission_log}")
                    print(run_dir.relative_to(root))
            else:
                run_dir, report = run_one_image_mock(config, root, args.image, args.label)
                print(report)
                print(f"run: {run_dir.relative_to(root)}")
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
