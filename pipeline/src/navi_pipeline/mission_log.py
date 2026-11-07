"""Replace mission-log labels with one hardcoded label.

Used only when a run passes a mock mission-log label. ``dialogue.py`` is not
involved: it still inserts the full log it receives into the section 4.D.3 prompt.
"""

from .records import Detection


CLASS_FREE_DESCRIPTION = "Detector record for this frame."


def overwrite_mission_log(detections: list[Detection], label: str) -> list[Detection]:
    """Ignore each image's identified class and store ``label`` instead.

    The description is replaced with a class-free line so the detector's wording
    cannot carry the true class into the prompt.
    """
    if not label.strip() or label != label.strip() or "\n" in label or "\r" in label:
        raise ValueError("Hardcoded mission-log label must be a nonempty single line")
    return [
        Detection(item.image_id, "valid", label, CLASS_FREE_DESCRIPTION, ())
        for item in detections
    ]
