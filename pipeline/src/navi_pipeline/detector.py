"""Detector stage: generation and bounded format retries, without reference labels."""

import hashlib
import time
from collections.abc import Callable

from .models import Backend
from .parsing import parse_detection
from .records import Attempt, Detection, ModelInput, ModelRequest


def seed_for(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:4], "big")


def detect(
    backend: Backend,
    item: ModelInput,
    labels: tuple[str, ...],
    max_attempts: int,
    seed: int,
    on_attempt: Callable[[Attempt], None] | None = None,
) -> Detection:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    attempts = []
    for number in range(1, max_attempts + 1):
        attempt_seed = seed_for(seed, f"{item.image_id}:attempt:{number}")
        started = time.perf_counter()
        try:
            response = backend.generate(ModelRequest(item.prompt, item.image, attempt_seed))
        except Exception as error:
            attempt = Attempt(number, attempt_seed, item.prompt, None, "inference_error",
                              None, None, f"{type(error).__name__}: {error}",
                              time.perf_counter() - started)
        else:
            try:
                parsed = parse_detection(response.text, labels)
            except ValueError as error:
                attempt = Attempt(number, attempt_seed, item.prompt, response.text, "parse_error",
                                  None, None, str(error), time.perf_counter() - started,
                                  response.metadata)
            else:
                attempt = Attempt(number, attempt_seed, item.prompt, response.text, "valid",
                                  parsed.label, parsed.description, None,
                                  time.perf_counter() - started, response.metadata)
        attempts.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt)
        if attempt.status != "parse_error":
            break
    final = attempts[-1]
    return Detection(item.image_id, final.status, final.label, final.description, tuple(attempts))
