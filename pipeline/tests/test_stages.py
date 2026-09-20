"""Stage contracts using scripted responses only; no model inference."""

import json
from copy import deepcopy

import pytest
from PIL import Image

from navi_pipeline.config import DetectionConfig, DialogueConfig
from navi_pipeline.detector import detect
from navi_pipeline.dialogue import ask_questions
from navi_pipeline.evaluation import evaluate
from navi_pipeline.parsing import parse_detection
from navi_pipeline.records import Attempt, Detection, ModelInput, ModelResponse
from navi_pipeline.transforms import defend_input, transform_input


LABELS = ("Forest", "River")
VALID = "Label: Forest\nDescription: Trees beside a clearing."


class ScriptedBackend:
    def __init__(self, *outcomes):
        self.outcomes = iter(outcomes)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return ModelResponse(outcome, {"synthetic": True})


@pytest.fixture
def item():
    with Image.new("RGB", (2, 3), "green") as image:
        yield ModelInput("img-0123456789abcdef", image, "Classify this image exactly.")


@pytest.mark.parametrize("text,description", [
    (VALID, "Trees beside a clearing."),
    ('"Label: Forest"\n"Description: Trees beside a clearing."',
     "Trees beside a clearing."),
    ("'Label: Forest'\n'Description: Trees beside a clearing.'",
     "Trees beside a clearing."),
    ("\n Label: Forest \n\n Description: Trees.\n A clearing. \n", "Trees.\nA clearing."),
    ('"Label: Forest"\n"Description: Trees."\n"A clearing."', "Trees.\nA clearing."),
    ("Label: Forest\nDescription:\nTrees.", "Trees."),
])
def test_parser_accepts_paper_quotes_plain_and_multiline(text, description):
    parsed = parse_detection(text, LABELS)
    assert parsed.label == "Forest"
    assert parsed.description == description


@pytest.mark.parametrize("text", [
    "", "Label: Forest", "Label: forest\nDescription: Trees.",
    "Label: Unknown\nDescription: Trees.", "Label:\nDescription: Trees.",
    "Label: Forest\nDescription: \n \n",
    "Label: Forest\nLabel: River\nDescription: Trees.",
    VALID + "\nLabel: River", VALID + "\nDescription: Another field.",
    VALID + '\n"Label: River"', VALID + "\nDescription : Another field.",
    "Description: Trees.\nLabel: Forest",
    "```text\n" + VALID + "\n```",
    "**Label:** Forest\n**Description:** Trees.",
    "# Label: Forest\nDescription: Trees.",
    "- Label: Forest\n- Description: Trees.",
])
def test_parser_rejects_invalid_labels_duplicate_empty_and_markdown_fields(text):
    with pytest.raises(ValueError):
        parse_detection(text, LABELS)


def test_parser_rejects_markdown_duplicate_after_valid_fields():
    with pytest.raises(ValueError):
        parse_detection(VALID + "\n**Label:** River", LABELS)


def test_detector_retries_same_prompt_with_unique_reproducible_seeds(item):
    config = DetectionConfig(max_attempts=3)
    scripts = ("malformed", "Label: Unknown\nDescription: Trees.", VALID)
    backend = ScriptedBackend(*scripts)
    saved = []
    result = detect(backend, item, LABELS, config.max_attempts, 42, saved.append)
    repeat = ScriptedBackend(*scripts)
    detect(repeat, item, LABELS, config.max_attempts, 42)

    assert result.status == "valid"
    assert result.label == "Forest"
    assert result.description == "Trees beside a clearing."
    assert result.image_id == item.image_id
    assert tuple(saved) == result.attempts
    assert [a.status for a in saved] == ["parse_error", "parse_error", "valid"]
    assert [a.number for a in saved] == [1, 2, 3]
    assert [a.response for a in saved] == list(scripts)
    assert all(a.error and a.label is None and a.description is None for a in saved[:2])
    assert saved[-1].error is None
    assert all(a.metadata == {"synthetic": True} for a in saved)
    assert all(a.elapsed_seconds >= 0 for a in saved)
    assert all(r.prompt == item.prompt and r.image is item.image for r in backend.requests)
    assert all(a.prompt == item.prompt for a in saved)
    seeds = [r.seed for r in backend.requests]
    assert len(set(seeds)) == config.max_attempts
    assert seeds == [a.seed for a in saved] == [r.seed for r in repeat.requests]
    assert all(0 <= seed < 2**32 for seed in seeds)


def test_detector_stops_at_parse_retry_budget(item):
    backend = ScriptedBackend("bad", "still bad", VALID)
    saved = []
    result = detect(backend, item, LABELS, 2, 42, saved.append)
    assert len(backend.requests) == len(saved) == 2
    assert tuple(saved) == result.attempts
    assert result.status == "parse_error"
    assert result.label is result.description is None
    assert all(a.status == "parse_error" for a in saved)


def test_detector_does_not_retry_semantically_wrong_valid_label(item):
    # The reference stays outside ModelInput and is used only after detection.
    reference_label = "Forest"
    backend = ScriptedBackend("Label: River\nDescription: Flowing water.", VALID)
    result = detect(backend, item, LABELS, 3, 42)
    assert result.status == "valid"
    assert result.label == "River" != reference_label
    assert len(result.attempts) == len(backend.requests) == 1
    assert vars(item).keys() == {"image_id", "image", "prompt"}
    assert vars(backend.requests[0]).keys() == {"prompt", "image", "seed"}


@pytest.mark.parametrize("prefix", [(), ("bad format",)])
def test_detector_saves_inference_exception_without_retry(item, prefix):
    backend = ScriptedBackend(*prefix, RuntimeError("backend unavailable"), VALID)
    saved = []
    result = detect(backend, item, LABELS, 3, 42, saved.append)
    assert len(backend.requests) == len(saved) == len(prefix) + 1
    assert tuple(saved) == result.attempts
    assert result.status == "inference_error"
    assert result.label is result.description is None
    final = result.to_dict()["attempts"][-1]
    assert final["status"] == "inference_error"
    assert final["error"] == "RuntimeError: backend unavailable"
    assert final["response"] is None
    assert final["prompt"] == item.prompt
    assert final["seed"] == backend.requests[-1].seed
    assert final["metadata"] == {}
    assert final["elapsed_seconds"] >= 0


def test_dialogue_uses_only_context_records_and_exact_grouped_questions(item):
    secret_attempt = Attempt(
        number=1, seed=99, prompt="PRIVATE_ATTEMPT_PROMPT", response="PRIVATE_RAW_OUTPUT",
        status="parse_error", label=None, description=None, error="PRIVATE_ERROR",
        elapsed_seconds=0.5, metadata={"path": "PRIVATE_ORIGINAL_IMAGE.png"},
    )
    item.image.filename = "PRIVATE_ORIGINAL_IMAGE.png"
    detections = [
        Detection(f"img-{index:016x}", "valid", "Forest", f"Scene {index}.", (secret_attempt,))
        for index in range(3)
    ]
    detections[1] = Detection(detections[1].image_id, "parse_error", None, None, (secret_attempt,))
    questions = ("  Which scenes contain trees?  ", "Compare them.\nUse only records.")
    config = DialogueConfig(enabled=True, group_size=2, questions=questions)
    template = "{classification_results}\nLABELS={user_label_list}\nQUESTION={user_input}"
    backend = ScriptedBackend("answer 0", "answer 1", "answer 2", "answer 3")
    records = list(ask_questions(backend, detections, LABELS, config, template, 42))

    assert len(records) == len(backend.requests) == 4
    assert [r["group"] for r in records] == [0, 0, 1, 1]
    assert [r["question"] for r in records] == list(questions) * 2
    for index, (record, request) in enumerate(zip(records, backend.requests)):
        group = detections[:2] if index < 2 else detections[2:]
        expected = [{"image_id": d.image_id, "status": d.status,
                     "label": d.label, "description": d.description} for d in group]
        assert [d.context_record() for d in group] == expected
        context, suffix = request.prompt.split("\nLABELS=", 1)
        assert json.loads(context) == expected
        assert suffix == "Forest, River\nQUESTION=" + questions[index % 2]
        assert request.image is None
        assert "PRIVATE_" not in request.prompt
        assert record["image_ids"] == [d.image_id for d in group]
        assert record["prompt"] == request.prompt
        assert record["seed"] == request.seed
        assert record["status"] == "valid"
        assert record["response"] == f"answer {index}"
    assert len({r.seed for r in backend.requests}) == 4


def test_disabled_dialogue_makes_no_requests():
    backend = ScriptedBackend()
    assert list(ask_questions(backend, [], LABELS, DialogueConfig(), "", 42)) == []
    assert backend.requests == []


def test_metrics_include_invalid_outcomes_in_denominators_and_false_negatives():
    references = [{"image_id": str(i), "label": label}
                  for i, label in enumerate(("Forest", "Forest", "River", "River"))]
    detections = [
        {"image_id": str(i), "status": status, "label": label,
         "attempts": [{"status": attempt_status, "elapsed_seconds": 1.0}
                      for attempt_status in statuses]}
        for i, (status, label, statuses) in enumerate([
            ("valid", "Forest", ["parse_error", "valid"]),
            ("parse_error", None, ["parse_error", "parse_error"]),
            ("valid", "Forest", ["valid"]),
            ("inference_error", None, ["inference_error"]),
        ])
    ]
    metrics = evaluate(references, list(reversed(detections)), LABELS)
    assert (metrics["scheduled"], metrics["correct"], metrics["valid"]) == (4, 1, 2)
    assert metrics["accuracy_all"] == 0.25
    assert metrics["accuracy_valid_only"] == metrics["valid_output_rate"] == 0.5
    assert metrics["per_class"]["Forest"] == {
        "precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2,
    }
    assert metrics["per_class"]["River"] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2,
    }
    assert metrics["macro_f1"] == metrics["weighted_f1"] == 0.25
    assert metrics["confusion_labels"] == [*LABELS, "__INVALID__"]
    assert metrics["confusion_matrix"] == [[1, 0, 1], [1, 0, 1], [0, 0, 0]]
    assert metrics["status_counts"] == {"valid": 2, "parse_error": 1, "inference_error": 1}
    assert metrics["first_attempt_parse_failures"] == 2
    assert metrics["first_attempt_parse_failure_rate"] == 0.5
    assert metrics["final_parse_failures"] == 1
    assert metrics["final_parse_failure_rate"] == 0.25
    assert metrics["model_calls"] == 6
    assert metrics["extra_attempts"] == 2
    assert metrics["mean_image_seconds"] == metrics["median_image_seconds"] == 1.5


def test_metrics_with_zero_valid_outputs_and_zero_support_class():
    references = [{"image_id": str(i), "label": "Forest"} for i in range(2)]
    detections = [{"image_id": str(i), "status": status, "label": None,
                   "attempts": [{"status": status, "elapsed_seconds": 0.0}]}
                  for i, status in enumerate(("parse_error", "inference_error"))]
    metrics = evaluate(references, detections, LABELS)
    assert metrics["scheduled"] == 2
    assert metrics["valid"] == metrics["correct"] == 0
    assert metrics["accuracy_valid_only"] is None
    for key in ("accuracy_all", "valid_output_rate", "macro_f1", "weighted_f1"):
        assert metrics[key] == 0.0
    for label, support in (("Forest", 2), ("River", 0)):
        assert metrics["per_class"][label] == {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": support,
        }
    assert metrics["confusion_matrix"] == [[0, 0, 2], [0, 0, 0], [0, 0, 0]]


@pytest.mark.parametrize("stage", [transform_input, defend_input])
def test_identity_stage_does_not_mutate_input(item, stage):
    item.image.info["nested"] = {"values": [1, 2]}
    fields = vars(item).copy()
    pixels = item.image.tobytes()
    info = deepcopy(item.image.info)
    mode, size = item.image.mode, item.image.size
    assert stage(item) is item
    assert vars(item) == fields
    assert item.image.tobytes() == pixels
    assert item.image.info == info
    assert (item.image.mode, item.image.size) == (mode, size)
