"""Data-only contracts shared between pipeline stages."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from PIL import Image


@dataclass(frozen=True)
class ModelRequest:
    prompt: str
    image: Image.Image | None
    seed: int


@dataclass(frozen=True)
class ModelResponse:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelInput:
    """No file paths or evaluation labels may cross this boundary."""

    image_id: str
    image: Image.Image
    prompt: str


@dataclass(frozen=True)
class Attempt:
    number: int
    seed: int
    prompt: str
    response: str | None
    status: Literal["valid", "parse_error", "inference_error"]
    label: str | None
    description: str | None
    error: str | None
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Detection:
    image_id: str
    status: Literal["valid", "parse_error", "inference_error"]
    label: str | None
    description: str | None
    attempts: tuple[Attempt, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def context_record(self) -> dict[str, Any]:
        return {"image_id": self.image_id, "status": self.status,
                "label": self.label, "description": self.description}
