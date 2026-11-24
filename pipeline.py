import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_utils import latest_checkpoint, project_path


def run(command):
    print("\n$", " ".join(str(item) for item in command))
    subprocess.run(command, check=True)


def main(args):
    project_dir = Path(__file__).resolve().parent
    data_dir = project_path(args.data)
    output_root = project_path(args.output)

    audit_path = output_root / "dataset_audit.json"
    run(
        [
            sys.executable,
            str(project_dir / "audit_dataset.py"),
            "--data",
            str(data_dir),
            "--output",
            str(audit_path),
        ]
    )

    train_command = [
        sys.executable,
        str(project_dir / "train.py"),
        "--data",
        str(data_dir),
        "--output",
        str(output_root),
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--min-lr",
        str(args.min_lr),
        "--device",
        args.device,
    ]
    if args.no_pretrained:
        train_command.append("--no-pretrained")
    if args.freeze_backbone:
        train_command.append("--freeze-backbone")
    previous_checkpoint = latest_checkpoint(output_root)
    run(train_command)

    checkpoint = latest_checkpoint(output_root)
    if checkpoint is None or checkpoint == previous_checkpoint:
        raise RuntimeError("Training finished without a new checkpoint")
    run_dir = checkpoint.parent

    run(
        [
            sys.executable,
            str(project_dir / "evaluate.py"),
            "--data",
            str(data_dir / "test"),
            "--checkpoint",
            str(checkpoint),
            "--device",
            args.device,
        ]
    )
    run(
        [
            sys.executable,
            str(project_dir / "error_analysis.py"),
            "--data",
            str(data_dir / "test"),
            "--checkpoint",
            str(checkpoint),
            "--top-k",
            str(args.top_k),
            "--device",
            args.device,
        ]
    )

    report_path = run_dir / "pipeline_summary.json"
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "dataset_audit": str(audit_path),
        "evaluation": str(run_dir / "evaluation"),
        "errors": str(run_dir / "errors"),
    }
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nPipeline completed:", run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run dataset audit, training, evaluation, and error analysis."
    )
    parser.add_argument("--data", default="data/split")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    main(parser.parse_args())
