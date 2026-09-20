"""Strict experiment configuration. Paths are relative to the project directory."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    backend: Literal["transformers", "fake"] = "transformers"
    fake_label: str = "Forest"
    model_id: str = "google/gemma-3-4b-it"
    revision: str = "main"
    device: Literal["cpu", "cuda", "mps"] = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    quantization: Literal["none", "4bit", "8bit"] = "none"
    adapter_path: str | None = None
    local_files_only: bool = True
    max_input_tokens: int = Field(default=20000, ge=1)
    max_new_tokens: int = Field(default=1000, ge=1)
    do_sample: bool = False
    temperature: float = Field(default=0.2, gt=0)

    @model_validator(mode="after")
    def check_precision(self):
        if self.quantization != "none" and self.device != "cuda":
            raise ValueError("Quantized inference requires explicit CUDA selection")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("Use float32 for the supported CPU profile")
        if self.device == "mps" and self.dtype == "bfloat16":
            raise ValueError("Use float16 or float32 for the supported MPS profile")
        return self


class DetectionConfig(StrictModel):
    prompt_path: str = "prompts/detection.txt"
    max_attempts: int = Field(default=3, ge=1, le=10)


class DialogueConfig(StrictModel):
    enabled: bool = False
    prompt_path: str = "prompts/dialogue.txt"
    questions: tuple[str, ...] = ()
    group_size: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def check_questions(self):
        if self.enabled and (not self.questions or any(not q.strip() for q in self.questions)):
            raise ValueError("Enabled dialogue needs nonempty scripted questions")
        return self


class ExperimentConfig(StrictModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    manifest_path: str
    labels: tuple[str, ...] = Field(min_length=2)
    split: Literal["development", "heldout", "train", "validation"] = "development"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    limit: int | None = Field(default=None, ge=1)
    output_dir: str = "runs"
    model: ModelConfig = ModelConfig()
    detection: DetectionConfig = DetectionConfig()
    dialogue: DialogueConfig = DialogueConfig()

    @model_validator(mode="after")
    def check_labels(self):
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Labels must be unique")
        if "__INVALID__" in self.labels:
            raise ValueError("__INVALID__ is reserved for evaluation failures")
        if self.model.backend == "fake" and self.model.fake_label not in self.labels:
            raise ValueError("model.fake_label must be a permitted label for synthetic runs")
        if any(not label.strip() or label != label.strip() or "\n" in label
               or "\r" in label for label in self.labels):
            raise ValueError("Labels must be nonempty, stripped, single-line strings")
        return self


def project_path(root: Path, value: str) -> Path:
    """Resolve a project-local path, rejecting traversal and escaping symlinks."""
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Expected a project-relative path inside {root}: {value}")
    return path


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
