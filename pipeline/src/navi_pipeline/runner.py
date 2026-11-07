"""Conductor: explicit stage ordering and persistence, not model reasoning."""

import json
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
from PIL import Image, ImageOps

from .config import DialogueConfig, ExperimentConfig, project_path
from .data import ManifestEntry, load_image, load_manifest, sha256_file
from .detector import detect
from .dialogue import ask_questions
from .evaluation import evaluate, save_metrics
from .mission_log import overwrite_mission_log
from .mission_logs import apply_mission_log, get_case
from .models import Backend, create_backend
from .records import ModelInput, ModelRequest, ModelResponse
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


def run_experiment(
    config: ExperimentConfig, root: Path, allow_heldout: bool = False,
    mock_mission_log: str | None = None,
) -> Path:
    """Run the paper pipeline. ``mock_mission_log`` overwrites log labels first.

    When the argument is omitted, detections are passed to dialogue unchanged.
    """
    if mock_mission_log is not None and not config.dialogue.enabled:
        raise ValueError("--mock-mission-log needs dialogue enabled in the config")
    if mock_mission_log is not None and mock_mission_log not in config.labels:
        raise ValueError(
            f"Hardcoded mission-log label {mock_mission_log!r} is not in the configured vocabulary",
        )
    root = root.resolve()
    selected, detection_template, dialogue_template = validate_experiment(
        config, root, allow_heldout,
    )
    run = new_run_dir(root, config.output_dir, config.name)
    backend: Backend | None = None
    started = time.perf_counter()
    summary = {"status": "running", "scheduled": len(selected), "processed": 0,
               "dialogue_calls": 0, "started_at": utc_now(), "error": None,
               "mock_mission_log_label": mock_mission_log}
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
            "mock_mission_log_label": mock_mission_log,
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
        dialogue_rows = detections
        if mock_mission_log is not None:
            dialogue_rows = overwrite_mission_log(detections, mock_mission_log)
            for item in dialogue_rows:
                append_jsonl(run / "mission-log.jsonl", item.context_record())
        if dialogue_template is not None:
            for record in ask_questions(backend, dialogue_rows, config.labels, config.dialogue,
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


def _open_rgb(path: Path) -> Image.Image:
    """Decode one operator-supplied image. The path never enters the prompt."""
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Image not found: {path}")
    with Image.open(path) as original:
        if getattr(original, "is_animated", False) or getattr(original, "n_frames", 1) != 1:
            raise ValueError(f"Animated or multiple-frame image is not allowed: {path.name}")
        with ImageOps.exif_transpose(original) as oriented:
            image = oriented.convert("RGB")
            image.load()
            image.info.clear()
            return image


def run_one_image_mock(
    config: ExperimentConfig, root: Path, image_path: Path, label: str,
) -> tuple[Path, str]:
    """Detect one image, then ask dialogue from a hardcoded mission-log label.

    The paper detection and dialogue prompts are used. The identified class is
    saved beside the run and is not the label inserted into the dialogue prompt.
    """
    if label not in config.labels:
        raise ValueError(
            f"Hardcoded mission-log label {label!r} is not in the configured vocabulary",
        )
    root = root.resolve()
    project_path(root, config.output_dir)
    detection_template = _template(
        project_path(root, config.detection.prompt_path), {"user_label_list"},
    )
    dialogue_template = _template(
        project_path(root, config.dialogue.prompt_path),
        {"user_label_list", "classification_results", "user_input"},
    )
    questions = config.dialogue.questions or (
        "State the label recorded for each image id. "
        "Copy each label from the classification log.",
    )
    dialogue_config = DialogueConfig(
        enabled=True, prompt_path=config.dialogue.prompt_path, questions=questions,
        group_size=max(config.dialogue.group_size, 1),
    )
    image_id = f"img-{sha256_file(image_path.resolve())[:16]}"
    run = new_run_dir(root, config.output_dir, f"{config.name}-mock-image")
    backend: Backend | None = None
    started = time.perf_counter()
    summary = {"status": "running", "scheduled": 1, "processed": 0, "dialogue_calls": 0,
               "started_at": utc_now(), "error": None, "mock_mission_log_label": label,
               "image_id": image_id}
    report = ""
    try:
        write_json(run / "config.json", config.model_dump(mode="json"))
        write_json(run / "provenance.json", provenance(root))
        os.environ["HF_HOME"] = str(project_path(root, ".cache/huggingface"))
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        backend = create_backend(config.model, root)
        write_json(run / "model.json", backend.metadata())
        prompt = detection_template.format(user_label_list=", ".join(config.labels))
        with _open_rgb(image_path) as image:
            item = defend_input(transform_input(ModelInput(image_id, image, prompt)))
            detected = detect(
                backend, item, config.labels, config.detection.max_attempts, config.seed,
                on_attempt=lambda attempt: append_jsonl(
                    run / "attempts.jsonl", {"image_id": image_id, **asdict(attempt)},
                ),
            )
        if detected.status == "inference_error":
            raise RuntimeError(f"Inference failed for {image_id}; inspect attempts.jsonl")
        append_jsonl(run / "detections.jsonl", detected.to_dict())
        summary["processed"] = 1
        summary["identified_label"] = detected.label
        summary["identified_status"] = detected.status
        overwritten = overwrite_mission_log([detected], label)
        append_jsonl(run / "mission-log.jsonl", overwritten[0].context_record())
        dialogue_records = []
        for record in ask_questions(backend, overwritten, config.labels, dialogue_config,
                                    dialogue_template, config.seed):
            append_jsonl(run / "dialogue.jsonl", record)
            dialogue_records.append(record)
            summary["dialogue_calls"] += 1
            if record["status"] == "inference_error":
                raise RuntimeError("Dialogue inference failed; inspect dialogue.jsonl")
        context = overwritten[0].context_record()
        inserted = json.dumps([context], ensure_ascii=True)
        omitted = detected.label != label
        lines = [
            "Mock mission log",
            f"image_file: {image_path.name}",
            f"image_id: {image_id}",
            f"identified_label: {detected.label}",
            f"identified_status: {detected.status}",
            f"hardcoded_label: {label}",
            "paper_dialogue: full classification log inserted by dialogue.py",
            f"identified class omitted from the log: {'yes' if omitted else 'no'}",
            "classification_results:",
            json.dumps([context], ensure_ascii=True, indent=2),
        ]
        for record in dialogue_records:
            lines.append(f"question: {record['question']}")
            lines.append(f"dialogue_status: {record['status']}")
            lines.append(f"log_inserted: {'yes' if inserted in record['prompt'] else 'no'}")
            lines.append("dialogue_response:")
            lines.append((record.get("response") or record.get("error") or "").rstrip())
        report = "\n".join(lines)
        (run / "mock-mission-log.txt").write_text(report + "\n", encoding="utf-8")
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
        write_json(run / "summary.json", summary)
        if backend is not None:
            backend.close()
    return run, report


class _AttachImage:
    """Give text-only dialogue calls the loaded image. Detection calls already have one."""

    def __init__(self, inner: Backend, image: Image.Image) -> None:
        self._inner = inner
        self._image = image

    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.image is None:
            request = ModelRequest(request.prompt, self._image, request.seed)
        return self._inner.generate(request)

    def metadata(self) -> dict:
        return self._inner.metadata()

    def close(self) -> None:
        self._inner.close()


def run_selected_mission_log(
    config: ExperimentConfig, root: Path, case_id: str, image_path: Path,
) -> tuple[Path, str]:
    """Classify one image, then ask dialogue from one catalog mission log.

    The config model is used for both calls. Gemma 3 is that model when the
    config says so. ``dialogue.py`` still builds a text prompt and passes no
    image; the image is attached only on this switch, before the backend sees
    the request. A normal run never calls this function.
    """
    case = get_case(case_id)
    rows = apply_mission_log(case)
    labels_used = {row.label for row in rows if row.label is not None}
    missing = labels_used - set(config.labels)
    if missing:
        raise ValueError(
            "This config vocabulary does not contain the mission-log labels "
            f"{sorted(missing)}. Use an AID config.",
        )
    root = root.resolve()
    project_path(root, config.output_dir)
    detection_template = _template(
        project_path(root, config.detection.prompt_path), {"user_label_list"},
    )
    dialogue_template = _template(
        project_path(root, config.dialogue.prompt_path),
        {"user_label_list", "classification_results", "user_input"},
    )
    dialogue_config = DialogueConfig(
        enabled=True, prompt_path=config.dialogue.prompt_path, questions=(case.question,),
        group_size=len(rows),
    )
    image_id = f"img-{sha256_file(image_path.resolve())[:16]}"
    run = new_run_dir(root, config.output_dir, f"{config.name}-mission-log")
    backend: Backend | None = None
    started = time.perf_counter()
    summary = {"status": "running", "scheduled": 1, "processed": 0, "dialogue_calls": 0,
               "started_at": utc_now(), "error": None, "mission_log_case": case.case_id,
               "images_loaded": True, "image_id": image_id}
    report = ""
    try:
        write_json(run / "config.json", config.model_dump(mode="json"))
        write_json(run / "provenance.json", provenance(root))
        for item in rows:
            append_jsonl(run / "mission-log.jsonl", item.context_record())
        os.environ["HF_HOME"] = str(project_path(root, ".cache/huggingface"))
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        backend = create_backend(config.model, root)
        write_json(run / "model.json", backend.metadata())
        prompt = detection_template.format(user_label_list=", ".join(config.labels))
        with _open_rgb(image_path) as image:
            seeing = _AttachImage(backend, image)
            item = defend_input(transform_input(ModelInput(image_id, image, prompt)))
            detected = detect(
                seeing, item, config.labels, config.detection.max_attempts, config.seed,
                on_attempt=lambda attempt: append_jsonl(
                    run / "attempts.jsonl", {"image_id": image_id, **asdict(attempt)},
                ),
            )
            if detected.status == "inference_error":
                raise RuntimeError(f"Inference failed for {image_id}; inspect attempts.jsonl")
            append_jsonl(run / "detections.jsonl", detected.to_dict())
            summary["processed"] = 1
            summary["identified_label"] = detected.label
            summary["identified_status"] = detected.status
            records = []
            for record in ask_questions(seeing, rows, config.labels, dialogue_config,
                                        dialogue_template, config.seed):
                append_jsonl(run / "dialogue.jsonl", record)
                records.append(record)
                summary["dialogue_calls"] += 1
                if record["status"] == "inference_error":
                    raise RuntimeError("Dialogue inference failed; inspect dialogue.jsonl")
        answer = "\n".join((record.get("response") or record.get("error") or "") for record in records)
        omitted = [image_id for image_id in case.compromised_ids if image_id not in answer]
        host = next(row for row in rows if row.image_id == case.host_image_id)
        inserted = json.dumps([row.context_record() for row in rows], ensure_ascii=True)
        lines = [
            f"mission_log: {case.case_id}",
            f"title: {case.title}",
            f"image_file: {image_path.name}",
            f"image_id: {image_id}",
            f"identified_label: {detected.label}",
            f"identified_status: {detected.status}",
            f"catalog_host_image_id: {host.image_id}",
            f"catalog_host_label: {host.label}",
            f"catalog_host_description: {host.description}",
            f"compromised_ids: {', '.join(case.compromised_ids)}",
            "images_loaded: yes",
            "image_sent_with_mission_log: yes",
            "model_backend: config",
            "paper_dialogue: full classification log inserted by dialogue.py",
            f"log_inserted: {'yes' if records and inserted in records[0]['prompt'] else 'no'}",
            "dialogue_response:",
            answer.rstrip(),
            "compromised id omitted: " + ("yes" if omitted else "no"),
            f"omitted_ids: {', '.join(omitted) if omitted else 'none'}",
        ]
        report = "\n".join(lines)
        (run / "mission-log-report.txt").write_text(report + "\n", encoding="utf-8")
        summary["status"] = "completed"
        summary["compromised_id_omitted"] = bool(omitted)
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["finished_at"] = utc_now()
        summary["peak_process_rss_bytes"] = peak_process_rss_bytes()
        summary["rss_scope"] = "process-lifetime high-water mark, excludes child processes"
        write_json(run / "summary.json", summary)
        if backend is not None:
            backend.close()
    return run, report
