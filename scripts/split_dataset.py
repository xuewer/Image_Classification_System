import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(args):
    ratios = (args.train, args.val, args.test)
    if not all(0 < ratio < 1 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train, val, and test ratios must be positive and sum to 1")
    rng = random.Random(args.seed)

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Input directory not found: {src}")

    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("Input and output directories must not overlap")
    if dst.exists() and any(
        path.is_file() and path.name != ".gitkeep" for path in dst.rglob("*")
    ):
        raise FileExistsError(
            f"Output directory contains existing files: {dst}. Use a new output directory."
        )

    class_dirs = sorted(path for path in src.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found in {src}")

    class_files = {}
    for class_dir in class_dirs:
        files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(files) < 3:
            raise ValueError(
                f"Class '{class_dir.name}' has only {len(files)} images; "
                "at least 3 are required for train/val/test."
            )

        class_files[class_dir] = files

    for class_dir, files in class_files.items():
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, int(n * args.train))
        n_val = max(1, int(n * args.val))
        if n_train + n_val >= n:
            n_val = 1
            n_train = n - 2

        split_files = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        for split, items in split_files.items():
            out_class = dst / split / class_dir.name
            out_class.mkdir(parents=True, exist_ok=True)
            for path in items:
                target = out_class / path.relative_to(class_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

        print(
            f"{class_dir.name}: total={n}, "
            f"train={len(split_files['train'])}, "
            f"val={len(split_files['val'])}, "
            f"test={len(split_files['test'])}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if min(args.train, args.val, args.test) <= 0:
        raise ValueError("train, val, and test ratios must be positive")
    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise ValueError("train + val + test must equal 1")
    main(args)
