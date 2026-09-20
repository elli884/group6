"""Scripted, text-only dialogue over bounded groups of detection records."""

import json
import time
from collections.abc import Iterator

from .config import DialogueConfig
from .detector import seed_for
from .models import Backend
from .records import Detection, ModelRequest


def ask_questions(
    backend: Backend,
    detections: list[Detection],
    labels: tuple[str, ...],
    config: DialogueConfig,
    template: str,
    seed: int,
) -> Iterator[dict]:
    if not config.enabled:
        return
    for offset in range(0, len(detections), config.group_size):
        group = detections[offset:offset + config.group_size]
        context = json.dumps([item.context_record() for item in group], ensure_ascii=True)
        for index, question in enumerate(config.questions):
            prompt = template.format(user_label_list=", ".join(labels),
                                     classification_results=context, user_input=question)
            request_seed = seed_for(seed, f"dialogue:{offset}:{index}")
            record = {"group": offset // config.group_size,
                      "image_ids": [item.image_id for item in group], "question": question,
                      "prompt": prompt, "seed": request_seed}
            started = time.perf_counter()
            try:
                response = backend.generate(ModelRequest(prompt, None, request_seed))
            except Exception as error:
                record.update(status="inference_error", response=None,
                              error=f"{type(error).__name__}: {error}", metadata={})
            else:
                record.update(status="valid", response=response.text, error=None,
                              metadata=response.metadata)
            record["elapsed_seconds"] = time.perf_counter() - started
            yield record
            if record["status"] == "inference_error":
                return
