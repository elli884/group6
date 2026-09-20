"""Small local CLI. Data preparation/validation never loads a VLM."""

import argparse
import json
import sys
from pathlib import Path

from .artifacts import write_json
from .config import load_config, project_path
from .data import prepare_manifest
from .evaluation import compare_runs
from .runner import run_experiment, validate_experiment


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
    for name in ("validate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--allow-heldout", action="store_true")
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
            else:
                print(run_experiment(config, root, args.allow_heldout).relative_to(root))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
