"""End-to-end filesystem contracts, using synthetic images and the fake backend only."""

import json

import pytest
from PIL import Image

from navi_pipeline.artifacts import read_jsonl
from navi_pipeline.config import ExperimentConfig, ModelConfig, project_path
from navi_pipeline.data import prepare_manifest
from navi_pipeline.evaluation import compare_runs, evaluate
from navi_pipeline.runner import run_experiment, validate_experiment


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / ".cache/huggingface"))
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    for class_index, label in enumerate(("Forest", "River")):
        folder = tmp_path / "data" / label
        folder.mkdir(parents=True)
        for index in range(4):
            with Image.new("RGB", (3, 2), (class_index * 100, index * 30, 20)) as image:
                image.save(folder / f"{index}.png")
    prepare_manifest(tmp_path, "data", "manifest.jsonl", "folders", 42,
                     heldout_fraction=0.25)
    (tmp_path / "detection.txt").write_text("Labels: {user_label_list}")
    (tmp_path / "dialogue.txt").write_text(
        "Labels: {user_label_list}\n{classification_results}\nQuestion: {user_input}"
    )
    config = ExperimentConfig.model_validate({
        "name": "fake-test", "manifest_path": "manifest.jsonl", "labels": ["Forest", "River"],
        "model": {"backend": "fake", "device": "cpu", "dtype": "float32"},
        "detection": {"prompt_path": "detection.txt"},
        "dialogue": {"enabled": True, "prompt_path": "dialogue.txt",
                     "questions": ["List the scenes."], "group_size": 2},
    })
    return tmp_path, config


def test_fake_pipeline_persists_complete_paired_results(experiment):
    root, config = experiment
    first = run_experiment(config, root)
    second = run_experiment(config, root)
    assert first != second
    summary = json.loads((first / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["scheduled"] == summary["processed"] == 6
    assert summary["dialogue_calls"] == 3
    metrics = json.loads((first / "metrics.json").read_text())
    assert metrics["synthetic"] is True
    assert metrics["accuracy_all"] == 0.5
    assert json.loads((first / "replay-config.json").read_text()) == config.model_dump(mode="json")
    assert len(read_jsonl(first / "attempts.jsonl")) == 6
    assert len(read_jsonl(first / "references.jsonl")) == 6
    assert (first / "manifest.jsonl").read_bytes() == (root / "manifest.jsonl").read_bytes()
    dialogue = read_jsonl(first / "dialogue.jsonl")
    assert all("data/" not in record["prompt"] for record in dialogue)
    comparison = compare_runs(first, second, target_label="River")
    assert comparison["synthetic"] is True
    assert comparison["label_agreement_valid_pairs"] == 1
    assert comparison["description_exact_match_valid_pairs"] == 1
    assert comparison["clean_correct"] == 3
    assert comparison["target_successes"] == 0


def test_validation_is_balanced_repeatable_and_guards_heldout(experiment):
    root, config = experiment
    limited = config.model_copy(update={"limit": 2})
    first, _, _ = validate_experiment(limited, root)
    second, _, _ = validate_experiment(limited, root)
    assert first == second
    assert {entry.label for entry in first} == {"Forest", "River"}
    heldout = config.model_copy(update={"split": "heldout"})
    with pytest.raises(ValueError, match="allow-heldout"):
        validate_experiment(heldout, root)
    selected, _, _ = validate_experiment(heldout, root, allow_heldout=True)
    assert len(selected) == 2
    assert all(entry.split == "heldout" for entry in selected)
    assert not (root / ".cache").exists()


def test_failed_backend_saves_error_and_does_not_retry_remaining_images(experiment, monkeypatch):
    root, config = experiment

    class BrokenBackend:
        def metadata(self):
            return {"synthetic": True}

        def generate(self, request):
            raise RuntimeError("synthetic infrastructure failure")

        def close(self):
            pass

    monkeypatch.setattr("navi_pipeline.runner.create_backend", lambda *_: BrokenBackend())
    with pytest.raises(RuntimeError, match="Inference failed"):
        run_experiment(config, root)
    run = next((root / "runs").iterdir())
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["processed"] == 1
    assert len(read_jsonl(run / "attempts.jsonl")) == 1
    assert not (run / "metrics.json").exists()


def test_comparison_rejects_changed_references(experiment):
    root, config = experiment
    first = run_experiment(config, root)
    second = run_experiment(config, root)
    path = second / "references.jsonl"
    references = read_jsonl(path)
    references[0]["sha256"] = "0" * 64
    path.write_text("\n".join(json.dumps(item) for item in references))
    with pytest.raises(ValueError, match="identical references"):
        compare_runs(first, second)


def test_evaluation_rejects_missing_or_duplicate_results():
    refs = [{"image_id": "one", "label": "Forest"}]
    with pytest.raises(ValueError, match="exactly one final result"):
        evaluate(refs, [], ("Forest", "River"))
    with pytest.raises(ValueError, match="unique IDs"):
        evaluate(refs + refs, [], ("Forest", "River"))


def test_paths_and_precision_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="project-relative"):
        project_path(tmp_path, "../outside")
    with pytest.raises(ValueError, match="CUDA"):
        ModelConfig(device="cpu", dtype="float32", quantization="4bit")
    with pytest.raises(ValueError, match="float32"):
        ModelConfig(device="cpu", dtype="float16")
