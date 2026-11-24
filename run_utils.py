import re
from pathlib import Path


RUN_DIR_PATTERN = re.compile(r"^test_(\d+)$")


def project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def create_run_dir(output_root):
    output_root = project_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_numbers = []
    for path in output_root.iterdir():
        if not path.is_dir():
            continue
        match = RUN_DIR_PATTERN.fullmatch(path.name)
        if match:
            run_numbers.append(int(match.group(1)))

    next_number = max(run_numbers, default=0) + 1
    while True:
        run_dir = output_root / f"test_{next_number:02d}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            next_number += 1


def latest_run_dir(output_root):
    output_root = project_path(output_root)
    runs = []
    for path in output_root.glob("test_*"):
        if not path.is_dir():
            continue
        match = RUN_DIR_PATTERN.fullmatch(path.name)
        if match:
            runs.append((int(match.group(1)), path))

    if not runs:
        return None
    return max(runs, key=lambda item: item[0])[1]


def latest_checkpoint(output_root="outputs"):
    checkpoints = []
    for run_dir in project_path(output_root).glob("test_*"):
        match = RUN_DIR_PATTERN.fullmatch(run_dir.name)
        checkpoint = run_dir / "best.pt"
        if match and checkpoint.is_file():
            checkpoints.append((int(match.group(1)), checkpoint))
    return max(checkpoints, key=lambda item: item[0])[1] if checkpoints else None
