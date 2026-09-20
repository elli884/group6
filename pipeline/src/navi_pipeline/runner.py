"""Conductor: explicit stage ordering and persistence, not model reasoning."""

import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from string import Formatter

from .artifacts import (
    append_jsonl,
    new_run_dir,
    peak_process_rss_bytes,
    provenance,
    utc_now,
    write_json,
)
from .config import ExperimentConfig, project_path
from .data import ManifestEntry, load_image, load_manifest, sha256_file
from .detector import detect
from .dialogue import ask_questions
from .evaluation import evaluate, save_metrics
from .models import Backend, create_backend
from .records import ModelInput
from .transforms import defend_input, transform_input


def _template(path: Path, fields: set[str]) -> str:
    text = path.read_text(encoding="utf-8")
    observed = {name for _, name, _, _ in Formatter().parse(text) if name is not None}
    if observed != fields:
        raise ValueError(f"{path}: expected exactly these prompt fields: {sorted(fields)}")
    text.format(**dict.fromkeys(fields, "validation"))
    return text


def validate_experiment(
    config: ExperimentConfig, root: Path, allow_heldout: bool = False,
) -> tuple[list[ManifestEntry], str, str | None]:
    if config.split == "heldout" and not allow_heldout:
        raise ValueError("Held-out evaluation requires --allow-heldout; never use it for tuning")
    if config.split in {"train", "validation"}:
        raise ValueError("The clean evaluation runner accepts development or heldout only")
    project_path(root, config.output_dir)
    if config.model.adapter_path is not None:
        project_path(root, config.model.adapter_path)
    entries = load_manifest(project_path(root, config.manifest_path), root, config.labels)
    selected = [entry for entry in entries if entry.split == config.split]
    if not selected:
        raise ValueError(f"No images in split {config.split}")
    if config.limit is not None and config.limit < len(selected):
        # Balanced round-robin selection; preserve manifest order after selection.
        groups = defaultdict(list)
        for entry in selected:
            groups[entry.label].append(entry)
        rng = random.Random(config.seed)
        for label in sorted(groups):
            rng.shuffle(groups[label])
        label_order = sorted(groups)
        rng.shuffle(label_order)
        chosen = set()
        while len(chosen) < config.limit:
            for label in label_order:
                if groups[label] and len(chosen) < config.limit:
                    chosen.add(groups[label].pop().image_id)
        selected = [entry for entry in selected if entry.image_id in chosen]
    detection = _template(project_path(root, config.detection.prompt_path), {"user_label_list"})
    dialogue = (_template(project_path(root, config.dialogue.prompt_path),
                          {"user_label_list", "classification_results", "user_input"})
                if config.dialogue.enabled else None)
    return selected, detection, dialogue


def run_experiment(config: ExperimentConfig, root: Path, allow_heldout: bool = False) -> Path:
    root = root.resolve()
    selected, detection_template, dialogue_template = validate_experiment(
        config, root, allow_heldout,
    )
    run = new_run_dir(root, config.output_dir, config.name)
    backend: Backend | None = None
    started = time.perf_counter()
    summary = {"status": "running", "scheduled": len(selected), "processed": 0,
               "dialogue_calls": 0, "started_at": utc_now(), "error": None}
    try:
        write_json(run / "config.json", config.model_dump(mode="json"))
        write_json(run / "provenance.json", provenance(root))
        manifest_path = project_path(root, config.manifest_path)
        # Both the full split manifest and the selected subset are frozen before inference.
        (run / "manifest.jsonl").write_bytes(manifest_path.read_bytes())
        write_json(run / "inputs.json", {
            "manifest_sha256": sha256_file(manifest_path),
            "detection_template": detection_template, "dialogue_template": dialogue_template,
            "preprocessing": "EXIF transpose; RGB conversion; metadata stripped; model processor",
            "transform": "identity", "defence": "identity", "allow_heldout": allow_heldout,
        })
        references = [entry.model_dump(mode="json") for entry in selected]
        for reference in references:
            append_jsonl(run / "references.jsonl", reference)
        os.environ["HF_HOME"] = str(project_path(root, ".cache/huggingface"))
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        model_started = time.perf_counter()
        backend = create_backend(config.model, root)
        summary["model_load_seconds"] = time.perf_counter() - model_started
        model_metadata = backend.metadata()
        write_json(run / "model.json", model_metadata)
        replay_config = config.model_dump(mode="json")
        if "resolved_revision" in model_metadata:
            replay_config["model"]["revision"] = model_metadata["resolved_revision"]
        write_json(run / "replay-config.json", replay_config)
        detections = []
        prompt = detection_template.format(user_label_list=", ".join(config.labels))
        for entry in selected:
            # Recheck just before loading: source data must stay immutable throughout a run.
            if sha256_file(project_path(root, entry.path)) != entry.sha256:
                raise ValueError(f"Image changed after validation: {entry.image_id}")
            with load_image(entry, root) as image:
                item = ModelInput(entry.image_id, image, prompt)
                item = defend_input(transform_input(item))
                result = detect(
                    backend, item, config.labels, config.detection.max_attempts, config.seed,
                    on_attempt=lambda attempt, image_id=entry.image_id: append_jsonl(
                        run / "attempts.jsonl", {"image_id": image_id, **asdict(attempt)},
                    ),
                )
            append_jsonl(run / "detections.jsonl", result.to_dict())
            detections.append(result)
            summary["processed"] += 1
            if result.status == "inference_error":
                raise RuntimeError(f"Inference failed for {entry.image_id}; inspect attempts.jsonl")
        metrics = evaluate(references, [item.to_dict() for item in detections], config.labels)
        metrics["synthetic"] = bool(model_metadata["synthetic"])
        save_metrics(run, metrics)
        if dialogue_template is not None:
            for record in ask_questions(backend, detections, config.labels, config.dialogue,
                                        dialogue_template, config.seed):
                append_jsonl(run / "dialogue.jsonl", record)
                summary["dialogue_calls"] += 1
                if record["status"] == "inference_error":
                    raise RuntimeError("Dialogue inference failed; inspect dialogue.jsonl")
        summary["status"] = "completed"
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["finished_at"] = utc_now()
        summary["peak_process_rss_bytes"] = peak_process_rss_bytes()
        summary["rss_scope"] = "process-lifetime high-water mark, excludes child processes"
        # Persist even if accelerator cleanup fails; a killed process may have traces only.
        write_json(run / "summary.json", summary)
        if backend is not None:
            backend.close()
    return run
