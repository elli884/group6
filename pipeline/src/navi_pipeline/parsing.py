"""Strict text-output gate. This validates structure, not factual correctness."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedDetection:
    label: str
    description: str


def parse_detection(text: str, labels: tuple[str, ...]) -> ParsedDetection:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    # The published prompt quotes each field; accept both quoted and plain fields.
    normalized = []
    for line in lines:
        if len(line) >= 2 and line[0] in {"'", '"'} and line[-1] == line[0]:
            line = line[1:-1].strip()
        normalized.append(line)
    if len(normalized) < 2 or not normalized[0].startswith("Label:"):
        raise ValueError("Expected Label: as the first field")
    label = normalized[0][len("Label:"):].strip()
    if label not in labels:
        raise ValueError("Label is not an exact member of the permitted vocabulary")
    if not normalized[1].startswith("Description:"):
        raise ValueError("Expected Description: immediately after Label:")
    if any(re.match(r"^[*#>\-\s]*(Label|Description)[*\s]*:", line)
           for line in normalized[2:]):
        raise ValueError("Duplicate output field")
    description = "\n".join([normalized[1][len("Description:"):].strip(), *normalized[2:]])
    if not description.strip():
        raise ValueError("Description is empty")
    return ParsedDetection(label, description.strip())
