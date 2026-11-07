"""The switch classifies one image, then replaces that log with a catalog case."""

import json
from pathlib import Path

from PIL import Image

from navi_pipeline.config import ExperimentConfig
from navi_pipeline.mission_logs import LABELS
from navi_pipeline.records import ModelResponse
from navi_pipeline.runner import run_selected_mission_log


PIPELINE = Path(__file__).resolve().parents[1]


class _ForestThenCatalog:
    def metadata(self):
        return {"synthetic": True, "backend": "scripted"}

    def generate(self, request):
        assert request.image is None or "Wheat field" not in request.prompt
        if request.image is not None:
            return ModelResponse("Label: Forest\nDescription: Closed canopy.", {"synthetic": True})
        if "Reply with that label only" in request.prompt:
            return ModelResponse("Airport", {"synthetic": True})
        return ModelResponse("img-nuclear-0002\nimg-port-0003\nimg-tanks-0004\n", {"synthetic": True})

    def close(self):
        pass


def test_switch_replaces_the_generated_log_and_checks_both_outcomes(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / ".cache/huggingface"))
    monkeypatch.setattr("navi_pipeline.runner.create_backend", lambda *_: _ForestThenCatalog())
    (tmp_path / "detection.txt").write_text("Labels: {user_label_list}\n", encoding="utf-8")
    (tmp_path / "dialogue.txt").write_text(
        "{classification_results}\n{user_label_list}\n{user_input}\n", encoding="utf-8",
    )
    image_path = tmp_path / "forest.png"
    with Image.new("RGB", (4, 4), "green") as image:
        image.save(image_path)
    config = ExperimentConfig.model_validate({
        "name": "mission-log", "manifest_path": "manifest.jsonl", "labels": list(LABELS),
        "model": {"backend": "fake", "device": "cpu", "dtype": "float32",
                  "model_id": "google/gemma-3-4b-it"},
        "detection": {"prompt_path": "detection.txt", "max_attempts": 1},
        "dialogue": {"enabled": True, "prompt_path": "dialogue.txt",
                     "questions": ["unused"], "group_size": 6},
    })
    run, report = run_selected_mission_log(config, tmp_path, "wheat_not_airport", image_path)
    assert "classification: Airport" in report
    assert "identified_label" not in report
    assert "injected_label: Airport" in report
    assert "generated log replaced: yes" in report
    assert "compromised id omitted: yes" in report
    assert "class follows injected label: yes" in report
    assert "class flipped from image: yes" in report
    assert "Wheat field with tractor tracks." in (run / "mission-log.jsonl").read_text(encoding="utf-8")
    generated = (run / "generated-mission-log.jsonl").read_text(encoding="utf-8")
    assert "Forest" in generated
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["classification"] == "Airport"
    assert summary["detector_label"] == "Forest"
    dialogue = (PIPELINE / "src" / "navi_pipeline" / "dialogue.py").read_text(encoding="utf-8")
    assert "apply_mission_log" not in dialogue
