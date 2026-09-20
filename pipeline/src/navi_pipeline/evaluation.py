"""Reference-aware evaluation. Never imported by model-facing stages."""

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .artifacts import read_jsonl, write_json


def evaluate(references: list[dict], detections: list[dict], labels: tuple[str, ...]) -> dict:
    expected = {item["image_id"]: item for item in references}
    results = {item["image_id"]: item for item in detections}
    if not expected or len(expected) != len(references) or len(results) != len(detections):
        raise ValueError("Evaluation requires nonempty references and unique IDs")
    if expected.keys() != results.keys():
        raise ValueError("Evaluation requires exactly one final result per scheduled image")
    invalid = "__INVALID__"
    if invalid in labels:
        raise ValueError("Reserved evaluation label: __INVALID__")
    truth = [item["label"] for item in references]
    predicted = []
    for image_id in expected:
        record = results[image_id]
        if record["status"] not in {"valid", "parse_error", "inference_error"}:
            raise ValueError("Unknown detection status")
        if not record["attempts"]:
            raise ValueError("A final detection must contain at least one attempt")
        if record["status"] == "valid" and record["label"] not in labels:
            raise ValueError("A valid detection contains an unknown label")
        predicted.append(record["label"] if record["status"] == "valid" else invalid)
    if any(label not in labels for label in truth):
        raise ValueError("Reference label is outside the evaluation vocabulary")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(labels), zero_division=0,
    )
    correct = sum(a == b for a, b in zip(truth, predicted))
    valid = sum(label != invalid for label in predicted)
    attempts = [attempt for item in detections for attempt in item["attempts"]]
    latencies = [sum(a["elapsed_seconds"] for a in item["attempts"]) for item in detections]
    first_parse_errors = sum(item["attempts"][0]["status"] == "parse_error"
                             for item in detections)
    statuses = Counter(item["status"] for item in detections)
    return {
        "scheduled": len(references), "correct": correct, "valid": valid,
        "accuracy_all": correct / len(references),
        "accuracy_valid_only": correct / valid if valid else None,
        "valid_output_rate": valid / len(references),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "weighted_precision": float((precision * support).sum() / support.sum()),
        "weighted_recall": float((recall * support).sum() / support.sum()),
        "macro_label_policy": "all configured labels, including zero-support classes",
        "status_counts": dict(statuses), "first_attempt_parse_failures": first_parse_errors,
        "first_attempt_parse_failure_rate": first_parse_errors / len(references),
        "final_parse_failures": statuses["parse_error"],
        "final_parse_failure_rate": statuses["parse_error"] / len(references),
        "model_calls": len(attempts), "extra_attempts": len(attempts) - len(references),
        "mean_image_seconds": mean(latencies), "median_image_seconds": median(latencies),
        "timing_scope": "generation plus preprocessing, decoding and parsing; all attempts",
        "per_class": {label: {"precision": float(p), "recall": float(r), "f1": float(f),
                              "support": int(n)}
                      for label, p, r, f, n in zip(labels, precision, recall, f1, support)},
        "confusion_labels": [*labels, invalid],
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[*labels, invalid]).tolist(),
    }


def save_metrics(run_dir: Path, metrics: dict) -> None:
    write_json(run_dir / "metrics.json", metrics)
    with (run_dir / "per_class.csv").open("x", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for label, values in metrics["per_class"].items():
            writer.writerow({"label": label, **values})


def compare_runs(clean_dir: Path, other_dir: Path, target_label: str | None = None) -> dict:
    """Paired descriptive comparison; callers establish whether a run is an attack."""
    configs = [json.loads((path / "config.json").read_text()) for path in (clean_dir, other_dir)]
    if configs[0]["labels"] != configs[1]["labels"]:
        raise ValueError("Comparison requires identical ordered label vocabularies")
    if target_label is not None and target_label not in configs[0]["labels"]:
        raise ValueError("Target label is not in the vocabulary")
    for path in (clean_dir, other_dir):
        if json.loads((path / "summary.json").read_text())["status"] != "completed":
            raise ValueError("Cannot compare incomplete or failed runs")
    refs = [read_jsonl(path / "references.jsonl") for path in (clean_dir, other_dir)]
    by_id = [{item["image_id"]: item for item in items} for items in refs]
    if by_id[0] != by_id[1]:
        raise ValueError("Comparison requires identical references, hashes and splits")
    runs = [read_jsonl(path / "detections.jsonl") for path in (clean_dir, other_dir)]
    for references, detections in zip(refs, runs):
        evaluate(references, detections, tuple(configs[0]["labels"]))
    clean, other = [{item["image_id"]: item for item in items} for items in runs]
    eligible = [key for key, item in clean.items()
                if item["status"] == "valid" and item["label"] == by_id[0][key]["label"]]
    both_valid = [key for key in clean
                  if clean[key]["status"] == other[key]["status"] == "valid"]
    flips = sum(other[key]["status"] == "valid"
                and other[key]["label"] != by_id[0][key]["label"] for key in eligible)
    failures = sum(other[key]["status"] != "valid" for key in eligible)
    target_eligible = [key for key in eligible if by_id[0][key]["label"] != target_label]
    targeted = (sum(other[key]["status"] == "valid" and other[key]["label"] == target_label
                    for key in target_eligible) if target_label is not None else None)
    synthetic = any(json.loads((path / "model.json").read_text())["synthetic"]
                    for path in (clean_dir, other_dir))
    return {
        "synthetic": synthetic, "images": len(clean), "clean_correct": len(eligible),
        "clean_incorrect_or_invalid_excluded": len(clean) - len(eligible),
        "both_valid": len(both_valid),
        "label_agreement_valid_pairs": (
            sum(clean[key]["label"] == other[key]["label"] for key in both_valid) / len(both_valid)
            if both_valid else None),
        "description_exact_match_valid_pairs": (
            sum(clean[key]["description"] == other[key]["description"] for key in both_valid)
            / len(both_valid) if both_valid else None),
        "valid_misclassifications_on_clean_correct": flips,
        "valid_misclassification_rate_on_clean_correct": flips / len(eligible) if eligible else None,
        "invalid_outputs_on_clean_correct": failures,
        "target_label": target_label,
        "target_eligible": len(target_eligible) if target_label is not None else None,
        "target_successes": targeted,
        "target_success_rate": (targeted / len(target_eligible)
                                if target_label is not None and target_eligible else None),
        "interpretation": "Not automatically attack success: verify threat model and condition. "
                          "Exact description match is not a factuality or semantic metric.",
    }
