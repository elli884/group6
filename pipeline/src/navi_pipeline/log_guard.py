"""External audit for mission-log poisoning. Not part of the paper dialogue path.

``dialogue.py`` still inserts the full classification log into the paper prompt.
This module only builds an operator report: provenance, a gold-profile decision,
and an XML copy that separates historical records from that decision. Nothing
here is substituted into ``{classification_results}``.
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from .records import Detection


@dataclass(frozen=True)
class ClassProfile:
    label: str
    critical: bool
    terms: tuple[str, ...]


# Terms are distinctive scene words. A record is admitted only when its
# description supports its own label more than any other profile.
BASE_PROFILES: dict[str, ClassProfile] = {
    profile.label: profile
    for profile in (
        ClassProfile(
            "Agricultural", False, ("farm", "crop", "field", "wheat", "orchard", "pasture", "barn"),
        ),
        ClassProfile("Airport", True, ("airport", "runway", "aircraft", "terminal", "taxiway")),
        ClassProfile("Beach", False, ("beach", "shore", "coastline", "swimmer")),
        ClassProfile("Bridge", True, ("bridge", "span")),
        ClassProfile(
            "Commercial District", False, ("commercial", "downtown", "skyscraper", "office"),
        ),
        ClassProfile("Desert", False, ("desert", "dune", "arid")),
        ClassProfile("Forest", False, ("forest", "canopy", "woodland", "tree")),
        ClassProfile("Highway Interchange", True, ("interchange", "highway", "freeway")),
        ClassProfile(
            "Industrial", True,
            (
                "industrial", "factory", "reactor", "nuclear", "refinery",
                "smokestack", "cooling tower",
            ),
        ),
        ClassProfile("Mountain", False, ("mountain", "peak", "ridge", "alpine")),
        ClassProfile("Parking", False, ("parking",)),
        ClassProfile(
            "Port", True,
            ("port", "harbor", "harbour", "crane", "cargo", "dock", "wharf", "pier", "ship"),
        ),
        ClassProfile("Railway Station", True, ("railway", "railroad", "train", "platform")),
        ClassProfile("Residential", False, ("residential", "house", "home", "garden", "apartment")),
        ClassProfile("River", False, ("river", "stream", "canal")),
        ClassProfile("Storage Tanks", True, ("tank", "fuel", "pipeline")),
        ClassProfile("Venue", False, ("stadium", "venue", "arena", "theater", "theatre")),
        ClassProfile("Water Reservoir", True, ("reservoir", "dam")),
    )
}

_COMMAND = re.compile(
    r"\b(?:do not|don't|omit|disregard|must not|you must|ignore)\b|\bdownlink\b",
    re.IGNORECASE,
)
_NOT_CRITICAL = re.compile(
    r"\bnot\s+critical\b|\bno\s+critical\b|\bnon[-\s]critical\b",
    re.IGNORECASE,
)
_SAFE_CLAIM = re.compile(
    r"\bis\s+safe\b|\bsafe\s+to\s+(?:ignore|skip|omit)\b",
    re.IGNORECASE,
)


def content_sha256(detection: Detection) -> str:
    payload = json.dumps(
        {
            "description": detection.description,
            "image_id": detection.image_id,
            "label": detection.label,
            "status": detection.status,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _term_body(term: str) -> str:
    parts = term.split()
    last = re.escape(parts[-1])
    if not parts[-1].endswith("s"):
        last += "s?"
    escaped = [re.escape(part) for part in parts[:-1]]
    return r"\s+".join([*escaped, last])


def _present(text: str, term: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9]){_term_body(term)}(?![A-Za-z0-9])", text, re.IGNORECASE,
    ) is not None


def _negated(text: str, term: str) -> bool:
    return re.search(
        rf"\b(?:no|not|without|isn't|isnt|aren't|arent)\b(?:\s+\S+){{0,4}}\s+{_term_body(term)}",
        text,
        re.IGNORECASE,
    ) is not None


def _support(text: str, terms: tuple[str, ...]) -> tuple[int, bool]:
    hits = 0
    denied = False
    for term in terms:
        if _negated(text, term):
            denied = True
        elif _present(text, term):
            hits += 1
    return hits, denied


def _best_other(text: str, label: str) -> tuple[str | None, int]:
    best_label = None
    best_hits = 0
    for other, profile in BASE_PROFILES.items():
        if other == label:
            continue
        hits, _denied = _support(text, profile.terms)
        if hits > best_hits:
            best_label = other
            best_hits = hits
    return best_label, best_hits


def assess_record(detection: Detection, labels: tuple[str, ...]) -> dict[str, Any]:
    """Decide whether this detector record may enter the dialogue prompt."""
    profile = BASE_PROFILES.get(detection.label or "")
    own_support = 0
    denied = False
    other_label = None
    other_support = 0
    if detection.description and profile is not None:
        own_support, denied = _support(detection.description, profile.terms)
        other_label, other_support = _best_other(detection.description, profile.label)

    if detection.description and re.search(r"[<>]", detection.description):
        reason = "markup_in_description"
        detail = "Scene text contains markup and was not copied into the prompt."
    elif detection.description and _COMMAND.search(detection.description):
        reason = "command_in_description"
        detail = (
            "Scene text contains operational command language and was not copied into the prompt."
        )
    elif detection.status != "valid" or not detection.description or not detection.label:
        reason = "missing_scene_text"
        detail = (
            f"Status {detection.status} has no scene text to check against a base profile."
        )
    elif detection.label not in labels or profile is None:
        reason = "label_not_in_profile"
        detail = f"Label {detection.label!r} is outside the gold-standard base profiles."
    elif profile.critical and (
        _NOT_CRITICAL.search(detection.description) or _SAFE_CLAIM.search(detection.description)
    ):
        reason = "false_operational_claim"
        detail = (
            f"{detection.label} is critical infrastructure in the base profile; "
            "the description says it is not critical or is safe."
        )
    elif denied:
        reason = "profile_contradiction"
        detail = f"Description denies terms in the {detection.label} base profile."
    elif other_support > own_support:
        reason = "profile_mismatch"
        detail = (
            f"Label {detection.label} support {own_support} is below "
            f"{other_label} support {other_support}."
        )
    elif own_support == 0:
        reason = "no_profile_support"
        detail = f"Description shares no terms with the {detection.label} base profile."
    else:
        reason = "gold_profile_match"
        detail = (
            f"Description matches the {detection.label} base profile (support {own_support})."
        )

    decision = "admit" if reason == "gold_profile_match" else "reject"
    return {
        "decision": decision,
        "reason": reason,
        "detail": detail,
        "label": detection.label,
        "matched_profile": detection.label if decision == "admit" else other_label,
        "own_support": own_support,
        "other_label": other_label,
        "other_support": other_support,
        "scene_text": detection.description,
        "scene_text_in_prompt": decision == "admit",
    }


def guard_classification_log(
    detections: list[Detection], labels: tuple[str, ...],
) -> tuple[str, list[dict[str, Any]]]:
    """Return XML historical data plus one provenance row per input record."""
    admitted = ET.Element("classification_log")
    rejected = ET.Element("rejected_log")
    audit: list[dict[str, Any]] = []
    for ordinal, detection in enumerate(detections):
        digest = content_sha256(detection)
        row = assess_record(detection, labels)
        row.update(
            ordinal=ordinal,
            image_id=detection.image_id,
            origin="detector",
            role="observation",
            content_sha256=digest,
        )
        provenance = {
            "origin": "detector",
            "role": "observation",
            "ordinal": str(ordinal),
            "content_sha256": digest,
        }
        if row["decision"] == "admit":
            record = ET.SubElement(admitted, "record", attrib=provenance)
            ET.SubElement(record, "image_id").text = detection.image_id
            ET.SubElement(record, "status").text = detection.status
            ET.SubElement(record, "label").text = detection.label
            ET.SubElement(record, "description").text = detection.description
        else:
            ET.SubElement(rejected, "rejected", attrib={
                **provenance,
                "image_id": detection.image_id,
                "reason": row["reason"],
            })
        audit.append(row)
    historical = ET.Element("historical_data")
    historical.append(admitted)
    historical.append(rejected)
    xml = ET.tostring(historical, encoding="unicode")
    return xml, audit


def explain_audit(audit: list[dict[str, Any]]) -> str:
    """Operator-facing lines. These are not sent to the dialogue model."""
    lines = []
    for row in audit:
        lines.append(
            f"  [{row['ordinal']}] {row['image_id']} audit={row['decision']} "
            f"reason={row['reason']} label={row['label']} "
            f"sha256={row['content_sha256'][:12]} origin={row['origin']} role={row['role']}"
        )
        lines.append(f"      {row['detail']}")
        lines.append(f"      scene text in the paper pipeline: {row['scene_text']}")
        if row["scene_text_in_prompt"]:
            lines.append("      defended copy: would include this scene text")
        else:
            lines.append("      defended copy: would withhold this scene text")
    return "\n".join(lines)
