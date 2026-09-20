"""Project-local image manifests and deterministic, content-disjoint splits.

AID labels follow Table III of sources/NAVI-Orbital-Paper.md. Its curation
description does not fully reconcile with canonical AID: the documented merges
and exclusions produce 17 classes, and no original folder supplies Highway
Interchange. This is a paper-inspired mapping, not an exact benchmark replica.
"""

import hashlib
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import ConfigDict, Field, field_validator

from .config import StrictModel, project_path

AID_LABELS = (
    "Agricultural", "Airport", "Beach", "Bridge", "Commercial District", "Desert",
    "Forest", "Highway Interchange", "Industrial", "Mountain", "Parking", "Port",
    "Railway Station", "Residential", "River", "Storage Tanks", "Venue", "Water Reservoir",
)

_AID_MAPPING = {
    "Airport": "Airport", "BareLand": "Desert", "Beach": "Beach", "Bridge": "Bridge",
    "Commercial": "Commercial District", "DenseResidential": "Residential",
    "Desert": "Desert", "Farmland": "Agricultural", "Forest": "Forest",
    "Industrial": "Industrial", "Meadow": "Agricultural", "MediumResidential": "Residential",
    "Mountain": "Mountain", "Park": "Forest", "Parking": "Parking", "Pond": "Water Reservoir",
    "Port": "Port", "RailwayStation": "Railway Station", "River": "River",
    "SparseResidential": "Residential", "Stadium": "Venue", "StorageTanks": "Storage Tanks",
    "Viaduct": "Bridge",
}
_AID_EXCLUDED = {"BaseballField", "Church", "School", "Center", "Resort", "Square", "Playground"}
_AID_AMBIGUITIES = (
    "Canonical AID has no Highway Interchange folder; no source class is fabricated. "
    "The stated merges and exclusions yield 17 mapped classes, not 18.",
    "The paper calls the excluded baseball class BaseballDiamond; canonical AID uses BaseballField.",
    "Farmland -> Agricultural, Pond -> Water Reservoir, and Commercial -> Commercial District "
    "are semantic correspondences to Table III, not explicit folder mappings in the paper.",
)
_EUROSAT_MAPPING = {
    "AnnualCrop": "Annual Crop", "Forest": "Forest", "HerbaceousVegetation": "Herbaceous Vegetation",
    "Highway": "Highway", "Industrial": "Industrial", "Pasture": "Pasture",
    "PermanentCrop": "Permanent Crop", "Residential": "Residential", "River": "River",
    "SeaLake": "Sea Lake",
}
EUROSAT_LABELS = tuple(sorted(_EUROSAT_MAPPING.values()))
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
_SPLITS = ("heldout", "train", "validation", "development")


class ManifestEntry(StrictModel):
    model_config = ConfigDict(strict=True)

    image_id: str = Field(pattern=r"^img-[0-9a-f]{16}$")
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    original_label: str
    label: str
    split: Literal["development", "heldout", "train", "validation"]

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        if Path(value).is_absolute() or ".." in Path(value).parts or "\x00" in value:
            raise ValueError("Image path must be project-relative without traversal")
        return value

    @field_validator("sha256")
    @classmethod
    def lowercase_digest(cls, value: str) -> str:
        return value.lower()


def _local_path(root: Path, value: str) -> Path:
    resolved = project_path(root, value)
    relative = Path(value)
    if ".." in relative.parts or "\x00" in value:
        raise ValueError(f"Path traversal is not allowed: {value}")
    current = root.resolve()
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {value}")
    return resolved


def sha256_file(path: Path) -> str:
    """Hash file bytes with bounded memory, without decoding or trusting the name."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(entry: ManifestEntry, root: Path) -> Image.Image:
    """Return detached RGB pixels, never a filename, with EXIF orientation applied."""
    path = _local_path(root, entry.path)
    with Image.open(path) as original:
        if getattr(original, "is_animated", False) or getattr(original, "n_frames", 1) != 1:
            raise ValueError(f"Animated or multiple-frame image is not allowed: {entry.path}")
        with ImageOps.exif_transpose(original) as oriented:
            image = oriented.convert("RGB")
            image.load()
            image.info.clear()
            return image


def load_manifest(path: Path, root: Path, labels: tuple[str, ...]) -> list[ManifestEntry]:
    """Validate the entire manifest before any split filtering can hide leakage."""
    root = root.resolve()
    if path.is_absolute():
        try:
            value = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("Manifest must be inside the project") from error
    else:
        value = path.as_posix()
    manifest = _local_path(root, value)
    entries = []
    ids: set[str] = set()
    paths: set[Path] = set()
    hashes: set[str] = set()
    with manifest.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                entry = ManifestEntry.model_validate_json(line)
                image_path = _local_path(root, entry.path)
                if entry.label not in labels:
                    raise ValueError(f"Unknown mapped label: {entry.label}")
                if entry.image_id != f"img-{entry.sha256[:16]}":
                    raise ValueError("Opaque image_id does not match sha256")
                if entry.sha256 in hashes:
                    raise ValueError("Duplicate content (possible split leakage)")
                if entry.image_id in ids:
                    raise ValueError("Duplicate image_id")
                if image_path in paths:
                    raise ValueError("Duplicate image path")
                if not image_path.is_file():
                    raise ValueError(f"Image file does not exist: {entry.path}")
                if sha256_file(image_path) != entry.sha256:
                    raise ValueError(f"Checksum mismatch: {entry.path}")
                with load_image(entry, root):
                    pass
            except (ValueError, OSError, Image.DecompressionBombError) as error:
                raise ValueError(f"Invalid manifest line {line_number}: {error}") from error
            ids.add(entry.image_id)
            paths.add(image_path)
            hashes.add(entry.sha256)
            entries.append(entry)
    if not entries:
        raise ValueError("Manifest must contain at least one entry")
    return entries


def prepare_manifest(
    root: Path,
    dataset_dir: str,
    output: str,
    taxonomy: Literal["aid18", "eurosat", "folders"],
    seed: int,
    heldout_fraction: float = 0.2,
    train_fraction: float = 0.0,
    validation_fraction: float = 0.0,
) -> dict:
    """Discover immediate class folders and recursively collect their images.

    Split each mapped class, allocating floor(n * fraction) in heldout, train,
    validation order and leaving the remainder for development. A hard-link
    publication of a flushed temporary file is atomic and cannot overwrite.
    """
    if taxonomy not in ("aid18", "eurosat", "folders"):
        raise ValueError(f"Unknown taxonomy: {taxonomy}")
    fractions = (heldout_fraction, train_fraction, validation_fraction)
    if any(not math.isfinite(f) or not 0 <= f <= 1 for f in fractions):
        raise ValueError("Split fractions must be finite and between zero and one")
    if sum(fractions) > 1:
        raise ValueError("Split fractions must sum to at most one")
    root = root.resolve()
    dataset = _local_path(root, dataset_dir)
    destination = _local_path(root, output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {destination}")
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    groups: dict[str, list[ManifestEntry]] = defaultdict(list)
    hashes: set[str] = set()
    ids: set[str] = set()
    excluded: Counter[str] = Counter()
    original_counts: Counter[str] = Counter()
    mapping = _AID_MAPPING if taxonomy == "aid18" else _EUROSAT_MAPPING
    for folder in sorted(dataset.iterdir()):
        _local_path(root, folder.relative_to(root).as_posix())
        if not folder.is_dir():
            if folder.suffix.lower() in _IMAGE_EXTENSIONS:
                raise ValueError(f"Image must be inside a class folder: {folder.name}")
            continue
        original_label = folder.name
        omit = taxonomy == "aid18" and original_label in _AID_EXCLUDED
        if taxonomy != "folders" and original_label not in mapping and not omit:
            raise ValueError(f"Unknown {taxonomy} class folder: {original_label}")
        label = original_label if taxonomy == "folders" else mapping.get(original_label)
        files = []

        def walk_error(error: OSError) -> None:
            raise error

        for directory, directories, filenames in os.walk(folder, onerror=walk_error):
            for name in sorted(directories + filenames):
                candidate = Path(directory) / name
                _local_path(root, candidate.relative_to(root).as_posix())
                if name in filenames and candidate.suffix.lower() in _IMAGE_EXTENSIONS:
                    if not candidate.is_file():
                        raise ValueError(f"Not a regular image file: {candidate}")
                    files.append(candidate)
        if omit:
            excluded[original_label] = len(files)
            continue
        for image_path in sorted(files):
            digest = sha256_file(image_path)
            image_id = f"img-{digest[:16]}"
            if digest in hashes:
                raise ValueError(f"Duplicate content (possible split leakage): {image_path}")
            if image_id in ids:
                raise ValueError(f"Duplicate image_id (hash prefix collision): {image_id}")
            entry = ManifestEntry(
                image_id=image_id, path=image_path.relative_to(root).as_posix(), sha256=digest,
                original_label=original_label, label=label, split="development",
            )
            with load_image(entry, root):
                pass
            hashes.add(digest)
            ids.add(image_id)
            groups[entry.label].append(entry)
            original_counts[original_label] += 1
    if not groups:
        raise ValueError("Dataset contains no included images")
    rng = random.Random(seed)
    entries = []
    class_splits = {}
    for label in sorted(groups):
        members = sorted(groups[label], key=lambda entry: entry.path)
        rng.shuffle(members)
        counts = [math.floor(len(members) * fraction) for fraction in fractions]
        counts.append(len(members) - sum(counts))
        class_splits[label] = dict(zip(_SPLITS, counts))
        start = 0
        for split, count in zip(_SPLITS, counts):
            entries.extend(
                entry.model_copy(update={"split": split}) for entry in members[start:start + count]
            )
            start += count
    entries.sort(key=lambda entry: entry.path)
    observed = sorted(groups)
    declared = AID_LABELS if taxonomy == "aid18" else (
        EUROSAT_LABELS if taxonomy == "eurosat" else tuple(observed)
    )
    audit = {
        "taxonomy": taxonomy, "seed": seed, "manifest_path": destination.relative_to(root).as_posix(),
        "total": len(entries), "labels": observed, "declared_labels": list(declared),
        "missing_labels": sorted(set(declared) - set(observed)),
        "mapped_class_counts": {label: len(groups[label]) for label in observed},
        "original_class_counts": dict(sorted(original_counts.items())),
        "excluded_class_counts": dict(sorted(excluded.items())),
        "split_counts": {split: sum(e.split == split for e in entries) for split in _SPLITS},
        "class_split_counts": class_splits,
        "ambiguities": list(_AID_AMBIGUITIES) if taxonomy == "aid18" else [],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as target:
            temporary = Path(target.name)
            for entry in entries:
                target.write(entry.model_dump_json() + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return audit
