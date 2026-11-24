import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

from model import build_model
from run_utils import latest_checkpoint, project_path


def main(args):
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
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
        raise ValueError(f"No images found in error-analysis dataset: {data_path}")
    if ds.classes != classes:
        raise ValueError(
            f"Dataset classes do not match checkpoint classes: "
            f"{ds.classes} != {classes}"
        )
    model = build_model(
        len(classes),
        pretrained=False,
        model_name=model_name,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    mistakes = []

    with torch.no_grad():
        for idx in range(len(ds)):
            image_tensor, label = ds[idx]
            logits = model(image_tensor.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0]
            pred = int(probs.argmax())
            conf = float(probs[pred])

            if pred != label:
                path = Path(ds.samples[idx][0])
                mistakes.append((conf, path, classes[label], classes[pred]))

    mistakes.sort(key=lambda x: x[0], reverse=True)

    out = (
        checkpoint_path.parent / "errors"
        if args.output is None
        else project_path(args.output)
    )
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out}. Use --output with a new directory."
        )
    out.mkdir(parents=True, exist_ok=True)

    for rank, (conf, path, true_name, pred_name) in enumerate(mistakes[:args.top_k], 1):
        safe_true = true_name.replace("/", "_")
        safe_pred = pred_name.replace("/", "_")
        target = out / (
            f"{rank:03d}__{safe_true}__{safe_pred}__"
            f"{conf:.3f}__{path.name}"
        )
        shutil.copy2(path, target)

    summary = {
        "total_samples": len(ds),
        "misclassified": len(mistakes),
        "error_rate": len(mistakes) / len(ds),
        "checkpoint": str(checkpoint_path),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Total test samples:", len(ds))
    print("Misclassified:", len(mistakes))
    print("Saved top errors to:", out)


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
        help="Error output directory. Defaults to checkpoint_dir/errors.",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
