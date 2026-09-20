"""Fake-only backend contract tests. No inference dependencies are needed."""

import builtins

import pytest
from PIL import Image

from navi_pipeline.config import ModelConfig
from navi_pipeline.models import FakeBackend, create_backend
from navi_pipeline.records import ModelRequest, ModelResponse


def test_fake_image_response_is_deterministic():
    backend = FakeBackend()
    request = ModelRequest("Classify this image", Image.new("RGB", (2, 2)), 42)
    response = backend.generate(request)
    assert isinstance(response, ModelResponse)
    assert response == backend.generate(request)
    assert response.text == (
        "Label: Forest\nDescription: Synthetic image response for pipeline testing."
    )
    assert response.metadata["synthetic"] is True
    assert response.metadata["seed"] == 42


def test_fake_ignores_prompt_pixels_and_image_metadata():
    backend = FakeBackend(label="Water")
    first = Image.new("RGB", (2, 2), "red")
    first.filename = "Forest_ground_truth.png"
    first.info["ground_truth"] = "Forest"
    second = Image.new("RGB", (3, 3), "blue")
    second.info["label"] = "Urban"
    a = backend.generate(ModelRequest("Labels: Forest, Urban", first, 1))
    b = backend.generate(ModelRequest("Ground truth: Urban", second, 2))
    assert a.text == b.text
    assert a.text.startswith("Label: Water\n")


def test_fake_dialogue_is_fixed():
    backend = FakeBackend()
    a = backend.generate(ModelRequest("Question one", None, 0))
    b = backend.generate(ModelRequest("Question two", None, 100))
    assert a.text == b.text == "Synthetic dialogue response for pipeline testing."
    assert "Label:" not in a.text


def test_fake_does_not_inspect_image():
    class UnreadableImage:
        def __getattribute__(self, name):
            raise AssertionError(f"Fake backend inspected image attribute {name}")

    response = FakeBackend().generate(ModelRequest("", UnreadableImage(), 0))
    assert response.text.startswith("Label: Forest\n")


def test_fake_metadata_is_independent():
    backend = FakeBackend()
    metadata = backend.metadata()
    assert metadata == {"backend": "fake", "synthetic": True, "label": "Forest"}
    metadata["label"] = "Changed"
    response = backend.generate(ModelRequest("", None, 0))
    response.metadata["synthetic"] = False
    assert backend.metadata()["label"] == "Forest"
    assert backend.metadata()["synthetic"] is True


def test_fake_close_is_idempotent():
    backend = FakeBackend()
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.generate(ModelRequest("", None, 0))


@pytest.mark.parametrize("label", ["", " ", " Forest", "Forest ", "A\nB", "A\rB"])
def test_fake_rejects_invalid_labels(label):
    with pytest.raises(ValueError, match="single-line"):
        FakeBackend(label)


def test_fake_factory_does_not_import_inference_stack(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "numpy", "peft", "bitsandbytes"}:
            raise AssertionError(f"Unexpected inference import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    backend = create_backend(ModelConfig(backend="fake"), tmp_path)
    assert isinstance(backend, FakeBackend)
    assert backend.generate(ModelRequest("", None, 42)).metadata["synthetic"] is True
    backend.close()
    assert not (tmp_path / ".cache").exists()
