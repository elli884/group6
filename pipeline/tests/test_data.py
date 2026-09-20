import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from navi_pipeline.data import (
    AID_LABELS,
    EUROSAT_LABELS,
    ManifestEntry,
    load_image,
    load_manifest,
    prepare_manifest,
    sha256_file,
)


def make_image(root: Path, relative: str, color: int = 0) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", (4, 3), (color, 40, 90)) as image:
        image.save(path)
    return path


def make_entry(root: Path, path: Path, **changes) -> ManifestEntry:
    digest = sha256_file(path)
    values = dict(
        image_id=f"img-{digest[:16]}", path=path.relative_to(root).as_posix(), sha256=digest,
        original_label="Forest", label="Forest", split="development",
    )
    values.update(changes)
    return ManifestEntry(**values)


def write_manifest(root: Path, entries: list[ManifestEntry]) -> Path:
    path = root / "manifest.jsonl"
    path.write_text("".join(entry.model_dump_json() + "\n" for entry in entries), encoding="utf-8")
    return path


def test_manifest_entry_is_strict_frozen_and_forbids_extras(tmp_path):
    entry = make_entry(tmp_path, make_image(tmp_path, "image.png"))
    with pytest.raises(ValidationError):
        entry.label = "River"
    for changes in (
        {"label": 12}, {"extra": "no"}, {"split": "test"}, {"image_id": "Forest-1"},
        {"sha256": "not-a-hash"}, {"path": "../image.png"}, {"path": "/image.png"},
    ):
        with pytest.raises(ValidationError):
            ManifestEntry.model_validate(entry.model_dump() | changes)


def test_sha256_streams_file(tmp_path):
    content = b"test bytes" * 300_000
    path = tmp_path / "large.bin"
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_load_manifest_and_detached_rgb(tmp_path):
    path = make_image(tmp_path, "images/Forest.png")
    entry = make_entry(tmp_path, path)
    manifest = write_manifest(tmp_path, [entry])
    assert load_manifest(manifest, tmp_path, ("Forest",)) == [entry]
    assert load_manifest(Path("manifest.jsonl"), tmp_path, ("Forest",)) == [entry]
    image = load_image(entry, tmp_path)
    try:
        path.unlink()
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (0, 40, 90)
        assert not getattr(image, "filename", None)
        assert image.info == {}
    finally:
        image.close()


def test_checksum_checked_even_in_other_split(tmp_path):
    path = make_image(tmp_path, "image.png")
    entry = make_entry(tmp_path, path, split="heldout")
    manifest = write_manifest(tmp_path, [entry])
    make_image(tmp_path, "image.png", 1)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_manifest(manifest, tmp_path, ("Forest",))


@pytest.mark.parametrize("split", ["development", "heldout", "train", "validation"])
def test_duplicate_content_rejected_across_manifest(tmp_path, split):
    first = make_image(tmp_path, "first.png")
    second = make_image(tmp_path, "second.png")
    entries = [make_entry(tmp_path, first), make_entry(tmp_path, second, split=split)]
    with pytest.raises(ValueError, match="Duplicate content"):
        load_manifest(write_manifest(tmp_path, entries), tmp_path, ("Forest",))


def test_duplicate_path_rejected_with_different_declared_hash(tmp_path):
    path = make_image(tmp_path, "image.png")
    first = make_entry(tmp_path, path)
    second = make_entry(tmp_path, path, sha256="a" * 64, image_id="img-" + "a" * 16)
    with pytest.raises(ValueError, match="Duplicate image path"):
        load_manifest(write_manifest(tmp_path, [first, second]), tmp_path, ("Forest",))


def test_duplicate_id_rejected_on_hash_prefix_collision(tmp_path):
    first = make_entry(tmp_path, make_image(tmp_path, "first.png"))
    digest = first.sha256[:16] + ("0" if first.sha256[16] != "0" else "1") + first.sha256[17:]
    second = make_entry(tmp_path, make_image(tmp_path, "second.png", 1), sha256=digest,
                        image_id=first.image_id)
    with pytest.raises(ValueError, match="Duplicate image_id"):
        load_manifest(write_manifest(tmp_path, [first, second]), tmp_path, ("Forest",))


@pytest.mark.parametrize("contents", ["", "\n  \n", "not json\n", "{}\n"])
def test_empty_or_invalid_manifest(tmp_path, contents):
    path = tmp_path / "manifest.jsonl"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path, tmp_path, ("Forest",))


def test_wrong_id_unknown_label_missing_file_and_invalid_image(tmp_path):
    path = make_image(tmp_path, "image.png")
    entry = make_entry(tmp_path, path)
    for changed, message in (
        (entry.model_copy(update={"image_id": "img-" + "f" * 16}), "image_id"),
        (entry.model_copy(update={"label": "Unknown"}), "Unknown mapped label"),
        (entry.model_copy(update={"path": "missing.png"}), "does not exist"),
    ):
        with pytest.raises(ValueError, match=message):
            load_manifest(write_manifest(tmp_path, [changed]), tmp_path, ("Forest",))
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Invalid manifest line"):
        load_manifest(write_manifest(tmp_path, [make_entry(tmp_path, path)]), tmp_path, ("Forest",))


def test_exif_transpose_and_rgb_conversion(tmp_path):
    path = tmp_path / "oriented.png"
    with Image.new("L", (2, 3), 100) as image:
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif)
    with load_image(make_entry(tmp_path, path), tmp_path) as image:
        assert image.size == (3, 2)
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (100, 100, 100)
        assert not image.getexif()


@pytest.mark.parametrize("extension", ["gif", "tiff"])
def test_reject_multiple_frames(tmp_path, extension):
    path = tmp_path / f"animated.{extension}"
    with Image.new("RGB", (4, 3), "red") as first:
        with Image.new("RGB", (4, 3), "blue") as second:
            first.save(path, save_all=True, append_images=[second])
    entry = make_entry(tmp_path, path)
    with pytest.raises(ValueError, match="multiple-frame"):
        load_image(entry, tmp_path)
    with pytest.raises(ValueError, match="multiple-frame"):
        load_manifest(write_manifest(tmp_path, [entry]), tmp_path, ("Forest",))


def test_prepare_deterministic_stratified_disjoint_splits(tmp_path):
    for class_index, label in enumerate(("Forest", "River")):
        for index in reversed(range(11)):
            make_image(tmp_path, f"dataset/{label}/nested/{index:02}.png", class_index * 20 + index)
    options = dict(root=tmp_path, dataset_dir="dataset", taxonomy="folders", seed=42,
                   heldout_fraction=0.2, train_fraction=0.3, validation_fraction=0.1)
    audit = prepare_manifest(output="manifests/first.jsonl", **options)
    prepare_manifest(output="manifests/second.jsonl", **options)
    first = tmp_path / "manifests/first.jsonl"
    assert first.read_bytes() == (tmp_path / "manifests/second.jsonl").read_bytes()
    assert audit["labels"] == ["Forest", "River"]
    assert audit["split_counts"] == {"heldout": 4, "train": 6, "validation": 2, "development": 10}
    entries = load_manifest(first, tmp_path, tuple(audit["labels"]))
    for label in audit["labels"]:
        assert audit["class_split_counts"][label] == {
            "heldout": 2, "train": 3, "validation": 1, "development": 5,
        }
    split_hashes = {
        split: {entry.sha256 for entry in entries if entry.split == split}
        for split in audit["split_counts"]
    }
    for left, hashes in split_hashes.items():
        for right, other in split_hashes.items():
            if left != right:
                assert hashes.isdisjoint(other)
    assert len({entry.image_id for entry in entries}) == 22


def test_prepare_rejects_duplicates_before_writing(tmp_path):
    make_image(tmp_path, "dataset/Forest/a.png")
    make_image(tmp_path, "dataset/River/b.png")
    with pytest.raises(ValueError, match="Duplicate content"):
        prepare_manifest(tmp_path, "dataset", "manifest.jsonl", "folders", 42)
    assert not (tmp_path / "manifest.jsonl").exists()


def test_prepare_refuses_overwrite(tmp_path):
    make_image(tmp_path, "dataset/Forest/a.png")
    output = tmp_path / "manifest.jsonl"
    output.write_text("existing content", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_manifest(tmp_path, "dataset", "manifest.jsonl", "folders", 42)
    assert output.read_text(encoding="utf-8") == "existing content"


def test_failed_atomic_publication_cleans_up(tmp_path, monkeypatch):
    make_image(tmp_path, "dataset/Forest/a.png")

    def fail_link(source, destination):
        raise OSError("publication failed")

    monkeypatch.setattr("navi_pipeline.data.os.link", fail_link)
    with pytest.raises(OSError, match="publication failed"):
        prepare_manifest(tmp_path, "dataset", "out/manifest.jsonl", "folders", 42)
    assert list((tmp_path / "out").iterdir()) == []


def test_atomic_publication_does_not_overwrite_racing_writer(tmp_path, monkeypatch):
    make_image(tmp_path, "dataset/Forest/a.png")

    def racing_link(source, destination):
        destination.write_text("other writer", encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr("navi_pipeline.data.os.link", racing_link)
    with pytest.raises(FileExistsError):
        prepare_manifest(tmp_path, "dataset", "out/manifest.jsonl", "folders", 42)
    assert (tmp_path / "out/manifest.jsonl").read_text(encoding="utf-8") == "other writer"
    assert len(list((tmp_path / "out").iterdir())) == 1


@pytest.mark.parametrize("fractions", [(-0.1, 0, 0), (0.5, 0.4, 0.2), (float("nan"), 0, 0),
                                      (0, float("inf"), 0)])
def test_invalid_fractions(tmp_path, fractions):
    with pytest.raises(ValueError, match="fractions"):
        prepare_manifest(tmp_path, "dataset", "manifest.jsonl", "folders", 42, *fractions)


def test_aid_mapping_and_missing_class_audit(tmp_path):
    for index, label in enumerate(("DenseResidential", "MediumResidential", "SparseResidential",
                                   "Pond", "Farmland", "Meadow", "BaseballField")):
        make_image(tmp_path, f"dataset/{label}/image.png", index)
    audit = prepare_manifest(tmp_path, "dataset", "manifest.jsonl", "aid18", 42)
    assert len(AID_LABELS) == 18
    assert audit["labels"] == ["Agricultural", "Residential", "Water Reservoir"]
    assert audit["mapped_class_counts"] == {
        "Agricultural": 2, "Residential": 3, "Water Reservoir": 1,
    }
    assert "Highway Interchange" in audit["missing_labels"]
    assert audit["missing_labels"] == sorted(set(AID_LABELS) - set(audit["labels"]))
    assert audit["excluded_class_counts"] == {"BaseballField": 1}
    assert audit["ambiguities"]
    entries = load_manifest(tmp_path / "manifest.jsonl", tmp_path, AID_LABELS)
    assert {entry.original_label for entry in entries if entry.label == "Residential"} == {
        "DenseResidential", "MediumResidential", "SparseResidential",
    }
    json.dumps(audit)


def test_eurosat_readable_labels(tmp_path):
    make_image(tmp_path, "dataset/AnnualCrop/image.png")
    audit = prepare_manifest(tmp_path, "dataset", "manifest.jsonl", "eurosat", 42)
    assert len(EUROSAT_LABELS) == 10
    assert audit["labels"] == ["Annual Crop"]
    entries = load_manifest(tmp_path / "manifest.jsonl", tmp_path, EUROSAT_LABELS)
    assert entries[0].label == "Annual Crop"
    assert entries[0].original_label == "AnnualCrop"


def test_all_canonical_aid_folders_yield_seventeen_observed_classes(tmp_path):
    folders = (
        "Airport", "BareLand", "BaseballField", "Beach", "Bridge", "Center", "Church",
        "Commercial", "DenseResidential", "Desert", "Farmland", "Forest", "Industrial",
        "Meadow", "MediumResidential", "Mountain", "Park", "Parking", "Playground", "Pond",
        "Port", "RailwayStation", "Resort", "River", "School", "SparseResidential", "Square",
        "Stadium", "StorageTanks", "Viaduct",
    )
    for index, folder in enumerate(folders):
        make_image(tmp_path, f"dataset/{folder}/image.png", index)
    audit = prepare_manifest(tmp_path, "dataset", "manifest.jsonl", "aid18", 42)
    assert audit["labels"] == sorted(set(AID_LABELS) - {"Highway Interchange"})
    assert audit["missing_labels"] == ["Highway Interchange"]
    assert audit["total"] == 23
    assert len(audit["excluded_class_counts"]) == 7


def test_prepare_rejects_invalid_image_without_publishing(tmp_path):
    image = make_image(tmp_path, "dataset/Forest/image.png")
    image.write_bytes(b"invalid image")
    with pytest.raises(OSError):
        prepare_manifest(tmp_path, "dataset", "manifest.jsonl", "folders", 42)
    assert not (tmp_path / "manifest.jsonl").exists()


def test_prepare_rejects_empty_dataset(tmp_path):
    (tmp_path / "dataset").mkdir()
    with pytest.raises(ValueError, match="no included images"):
        prepare_manifest(tmp_path, "dataset", "manifest.jsonl", "folders", 42)
    assert not (tmp_path / "manifest.jsonl").exists()


@pytest.mark.parametrize("taxonomy,folder", [("aid18", "HighwayInterchange"),
                                             ("aid18", "Intersection"), ("eurosat", "Unknown")])
def test_unknown_named_class_rejected(tmp_path, taxonomy, folder):
    make_image(tmp_path, f"dataset/{folder}/image.png")
    with pytest.raises(ValueError, match="Unknown .* class folder"):
        prepare_manifest(tmp_path, "dataset", "manifest.jsonl", taxonomy, 42)


@pytest.mark.parametrize("value", ["../escape.jsonl", "/tmp/escape.jsonl"])
def test_output_must_be_project_relative(tmp_path, value):
    make_image(tmp_path, "dataset/Forest/image.png")
    with pytest.raises(ValueError):
        prepare_manifest(tmp_path, "dataset", value, "folders", 42)


def test_symlink_files_folders_and_output_rejected(tmp_path):
    path = make_image(tmp_path, "dataset/Forest/image.png")
    link = tmp_path / "dataset/Forest/link.png"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="Symlinks"):
        prepare_manifest(tmp_path, "dataset", "manifest.jsonl", "folders", 42)
    entry = make_entry(tmp_path, path).model_copy(update={"path": "dataset/Forest/link.png"})
    with pytest.raises(ValueError, match="Symlinks"):
        load_manifest(write_manifest(tmp_path, [entry]), tmp_path, ("Forest",))
    link.unlink()
    (tmp_path / "alias").symlink_to(tmp_path / "dataset", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinks"):
        prepare_manifest(tmp_path, "alias", "other.jsonl", "folders", 42)
    with pytest.raises(ValueError, match="Symlinks"):
        prepare_manifest(tmp_path, "dataset", "alias/output.jsonl", "folders", 42)


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = make_image(tmp_path, "outside/image.png")
    (root / "escape.png").symlink_to(outside)
    entry = make_entry(tmp_path, outside).model_copy(update={"path": "escape.png"})
    with pytest.raises(ValueError, match="project-relative"):
        load_manifest(write_manifest(root, [entry]), root, ("Forest",))
    with pytest.raises(ValueError, match="inside the project"):
        load_manifest(tmp_path / "outside.jsonl", root, ("Forest",))
