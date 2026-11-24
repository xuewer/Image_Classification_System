import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(split_dir):
    result = {"total": 0, "classes": {}, "invalid_images": []}
    if not split_dir.is_dir():
        return result

    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        count = 0
        for path in class_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            count += 1
            result["total"] += 1
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as error:
                result["invalid_images"].append(
                    {"path": str(path), "error": str(error)}
                )
        result["classes"][class_dir.name] = count
    return result


def main(args):
    root = Path(args.data)
    report = {"data_dir": str(root), "splits": {}, "duplicate_hashes": [], "errors": []}
    hashes = {}

    for split in SPLITS:
        split_report = audit_split(root / split)
        report["splits"][split] = split_report
        if split_report["total"] == 0:
            report["errors"].append(f"Missing or empty split: {split}")
        for name, count in split_report["classes"].items():
            if count == 0:
                report["errors"].append(f"Empty class: {split}/{name}")
        if split_report["invalid_images"]:
            report["errors"].append(f"Invalid images in {split}")
        for path in (root / split).rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                digest = file_hash(path)
                hashes.setdefault(digest, []).append(
                    {"split": split, "path": str(path)}
                )

    report["duplicate_hashes"] = [
        {"sha256": digest, "files": paths}
        for digest, paths in hashes.items()
        if len({item["split"] for item in paths}) > 1
    ]
    report["cross_split_duplicate_file_count"] = sum(
        len(item["files"]) for item in report["duplicate_hashes"]
    )
    expected_classes = set(report["splits"]["train"]["classes"])
    for split in ("val", "test"):
        if set(report["splits"][split]["classes"]) != expected_classes:
            report["errors"].append(f"Class mismatch in {split}")
    if report["duplicate_hashes"]:
        report["errors"].append("Identical image files found across splits")
    report["passed"] = not report["errors"]

    output = Path(args.output) if args.output else root / "audit_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved audit report to:", output)
    if not report["passed"]:
        raise SystemExit("Dataset audit failed; see the report for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/split")
    parser.add_argument("--output", default=None)
    main(parser.parse_args())
