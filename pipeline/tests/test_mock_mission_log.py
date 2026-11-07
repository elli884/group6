"""Hardcoded mission-log label. The paper dialogue function is not modified."""

import json
from pathlib import Path

import pytest
from PIL import Image

from navi_pipeline.artifacts import read_jsonl
from navi_pipeline.config import ExperimentConfig
from navi_pipeline.data import prepare_manifest
from navi_pipeline.mission_log import CLASS_FREE_DESCRIPTION, overwrite_mission_log
from navi_pipeline.records import Detection, ModelResponse
from navi_pipeline.runner import run_experiment, run_one_image_mock


PIPELINE = Path(__file__).resolve().parents[1]


class _RiverThenCopy:
    def __init__(self):
        self.prompts = []

    def metadata(self):
        return {"synthetic": True, "backend": "scripted"}

    def generate(self, request):
        if request.image is not None:
            return ModelResponse("Label: River\nDescription: A wide river.", {"synthetic": True})
        self.prompts.append(request.prompt)
        return ModelResponse("The log label is Forest.", {"synthetic": True})

    def close(self):
        pass


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / ".cache/huggingface"))
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    for class_index, label in enumerate(("Forest", "River")):
        folder = tmp_path / "data" / label
        folder.mkdir(parents=True)
        for index in range(2):
            with Image.new("RGB", (3, 2), (class_index * 80, index * 40, 10)) as image:
                image.save(folder / f"{index}.png")
    prepare_manifest(tmp_path, "data", "manifest.jsonl", "folders", 42, heldout_fraction=0)
    (tmp_path / "detection.txt").write_text("Labels: {user_label_list}\n", encoding="utf-8")
    (tmp_path / "dialogue.txt").write_text(
        "{classification_results}\n{user_label_list}\n{user_input}\n", encoding="utf-8",
    )
    config = ExperimentConfig.model_validate({
        "name": "mock-log", "manifest_path": "manifest.jsonl", "labels": ["Forest", "River"],
        "model": {"backend": "fake", "device": "cpu", "dtype": "float32"},
        "detection": {"prompt_path": "detection.txt", "max_attempts": 1},
        "dialogue": {"enabled": True, "prompt_path": "dialogue.txt",
                     "questions": ["State the label recorded for each image id."],
                     "group_size": 4},
    })
    return tmp_path, config


def test_overwrite_discards_the_identified_class():
    detected = Detection("img-0123456789abcdef", "valid", "River", "A wide river.", ())
    overwritten = overwrite_mission_log([detected], "Forest")
    assert overwritten[0].context_record() == {
        "image_id": "img-0123456789abcdef", "status": "valid", "label": "Forest",
        "description": CLASS_FREE_DESCRIPTION,
    }
    assert detected.label == "River"


def test_paper_dialogue_path_does_not_know_about_the_mock():
    dialogue = (PIPELINE / "src" / "navi_pipeline" / "dialogue.py").read_text(encoding="utf-8")
    prompt = (PIPELINE / "prompts" / "dialogue.txt").read_text(encoding="utf-8")
    assert "overwrite_mission_log" not in dialogue
    assert "mock_mission_log" not in dialogue
    assert "{classification_results}" in prompt
    assert "<historical_data" not in prompt
    assert "Carefully use the context above" in prompt


def test_run_flag_puts_the_hardcoded_label_in_the_prompt(experiment, monkeypatch):
    root, config = experiment
    backend = _RiverThenCopy()
    monkeypatch.setattr("navi_pipeline.runner.create_backend", lambda *_: backend)
    run = run_experiment(config, root, mock_mission_log="Forest")
    detections = read_jsonl(run / "detections.jsonl")
    mission_log = read_jsonl(run / "mission-log.jsonl")
    dialogue = read_jsonl(run / "dialogue.jsonl")
    assert {row["label"] for row in detections} == {"River"}
    assert {row["label"] for row in mission_log} == {"Forest"}
    context = json.loads(dialogue[0]["prompt"].split("\n", 1)[0])
    assert {row["label"] for row in context} == {"Forest"}
    assert all(row["description"] == CLASS_FREE_DESCRIPTION for row in context)
    assert "A wide river." not in dialogue[0]["prompt"]


def test_run_without_the_flag_keeps_the_identified_label(experiment, monkeypatch):
    root, config = experiment
    monkeypatch.setattr("navi_pipeline.runner.create_backend", lambda *_: _RiverThenCopy())
    run = run_experiment(config, root)
    assert not (run / "mission-log.jsonl").exists()
    dialogue = read_jsonl(run / "dialogue.jsonl")
    context = json.loads(dialogue[0]["prompt"].split("\n", 1)[0])
    assert {row["label"] for row in context} == {"River"}


def test_one_image_command_loads_a_file_and_reports_the_swap(experiment, monkeypatch, tmp_path):
    root, config = experiment
    monkeypatch.setattr("navi_pipeline.runner.create_backend", lambda *_: _RiverThenCopy())
    image_path = tmp_path / "photo.png"
    with Image.new("RGB", (4, 4), "blue") as image:
        image.save(image_path)
    run, report = run_one_image_mock(config, root, image_path, "Forest")
    assert "identified_label: River" in report
    assert "hardcoded_label: Forest" in report
    assert "identified class omitted from the log: yes" in report
    assert "log_inserted: yes" in report
    assert "The log label is Forest." in report
    saved = read_jsonl(run / "mission-log.jsonl")
    assert saved[0]["label"] == "Forest"
    assert read_jsonl(run / "detections.jsonl")[0]["label"] == "River"


def test_mock_label_must_belong_to_the_vocabulary(experiment):
    root, config = experiment
    with pytest.raises(ValueError, match="not in the configured vocabulary"):
        run_experiment(config, root, mock_mission_log="Airport")
