import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model
from run_utils import latest_checkpoint, project_path


def append_metrics_to_analysis(output_dir, metrics):
    analysis_path = output_dir / "analysis.txt"
    if not analysis_path.exists():
        return

    text = analysis_path.read_text(encoding="utf-8")
    marker = "评估补充"
    if marker in text:
        return

    section = (
        f"\n\n{marker}\n"
        f"- Accuracy：{metrics['accuracy']:.2%}\n"
        f"- Macro Precision：{metrics['macro_precision']:.2%}\n"
        f"- Macro Recall：{metrics['macro_recall']:.2%}\n"
        f"- Macro F1：{metrics['macro_f1']:.2%}\n"
        "详细分类报告和混淆矩阵保存在 evaluation/ 目录。\n"
    )
    analysis_path.write_text(text + section, encoding="utf-8")


def main(args):
    checkpoint_path = project_path(args.checkpoint) if args.checkpoint else latest_checkpoint("outputs")
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(
            "No checkpoint found. Train a model first or pass --checkpoint explicitly."
        )

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes = ckpt["classes"]
    image_size = ckpt.get("image_size", 224)
    model_name = ckpt.get("model_name", "resnet18")

    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    data_path = project_path(args.data)
    ds = datasets.ImageFolder(data_path, transform=tfm)
    if not ds.samples:
        raise ValueError(
            f"No images found in test dataset: {data_path}. "
            "Check the --data path and class subdirectories."
        )
    if ds.classes != classes:
        raise ValueError(
            "Dataset classes do not match checkpoint classes. "
            f"Dataset: {ds.classes}; checkpoint: {classes}"
        )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        len(classes),
        pretrained=False,
        model_name=model_name,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")
    print("\nClassification Report:")
    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out = (
        checkpoint_path.parent / "evaluation"
        if args.output is None
        else project_path(args.output)
    )
    out.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "num_samples": int(len(y_true)),
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
    }

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(
            classification_report(
                y_true,
                y_pred,
                target_names=classes,
                zero_division=0,
            )
        )
    append_metrics_to_analysis(checkpoint_path.parent, metrics)

    print("Saved evaluation results to:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to best.pt from the latest test_XX run.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Evaluation output directory. Defaults to checkpoint_dir/evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
