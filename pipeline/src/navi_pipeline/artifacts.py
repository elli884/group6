"""Filesystem-only experiment storage and provenance, separate from model code."""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import project_path
from .data import sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, ensure_ascii=True, allow_nan=False)
        target.write("\n")


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def new_run_dir(root: Path, output: str, name: str) -> Path:
    parent = project_path(root, output)
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / f"{name}-{uuid4().hex}"
    run.mkdir()
    return run


def provenance(root: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  text=True, check=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    paths = [root / "pyproject.toml", root / "uv.lock", root / ".python-version"]
    for directory, pattern in (("src", "*.py"), ("prompts", "*.txt")):
        paths.extend(sorted((root / directory).rglob(pattern)))
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in paths if path.is_file()}
    return {
        "created_at": utc_now(), "python": sys.version, "platform": platform.platform(),
        "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "project_diff": git("diff", "HEAD", "--", "."),
        "source_sha256": hashes,
        "packages": {dist.metadata["Name"]: dist.version
                     for dist in importlib.metadata.distributions() if dist.metadata["Name"]},
    }


def peak_process_rss_bytes() -> int | None:
    """Process lifetime high-water RSS, not per-image or child-process memory."""
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        return None
