import argparse
import json
import os
import random
from datetime import date
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets
from model import build_model
from run_utils import create_run_dir, project_path


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        progress = tqdm(loader, leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


def plot_history(history, output_dir):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    axes[2].plot(epochs, [row["lr"] for row in history], label="lr")
    axes[2].set_title("Learning rate")
    axes[2].set_xlabel("Epoch")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def write_analysis_report(
    output_dir,
    args,
    classes,
    train_size,
    val_size,
    test_size,
    history,
    best_val_acc,
):
    first = history[0]
    final = history[-1]
    best_epoch = max(history, key=lambda row: row["val_acc"])
    mode = (
        "ImageNet 预训练迁移学习"
        if not args.no_pretrained
        else "从零训练"
    )
    metrics_path = output_dir / "evaluation" / "metrics.json"
    metrics = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    def percent(value):
        return f"{value * 100:.2f}%"

    lines = [
        "图像分类训练结果分析",
        f"运行目录：{output_dir}",
        f"分析日期：{date.today().isoformat()}",
        "",
        "一、实验配置",
        "- 模型：ResNet18/ResNet50 可配置",
        f"- 实际模型：{args.model}",
        f"- 训练方式：{mode}",
        f"- 训练轮数：{args.epochs}",
        f"- batch size：{args.batch_size}",
        f"- 初始学习率：{args.lr:g}",
        f"- 最小学习率：{args.min_lr:g}",
        "- 学习率策略：CosineAnnealingLR",
        f"- 设备：{args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}",
        f"- 随机种子：{args.seed}",
        "",
        "二、数据规模",
        f"- 类别：{', '.join(classes)}",
        f"- 训练集：{train_size} 张",
        f"- 验证集：{val_size} 张",
        f"- 测试集：{test_size} 张",
        "",
        "三、每轮训练结果",
        "轮次 | 训练损失 | 训练准确率 | 验证损失 | 验证准确率 | 学习率",
    ]
    for row in history:
        lines.append(
            f"{row['epoch']} | {row['train_loss']:.4f} | "
            f"{percent(row['train_acc'])} | {row['val_loss']:.4f} | "
            f"{percent(row['val_acc'])} | {row['lr']:.8g}"
        )

    lines.extend(
        [
            "",
            "四、结果摘要",
            f"- 最佳验证结果：第 {best_epoch['epoch']} 轮，"
            f"验证准确率 {percent(best_val_acc)}",
            f"- 训练损失：{first['train_loss']:.4f} -> {final['train_loss']:.4f}",
            f"- 验证损失：{first['val_loss']:.4f} -> {final['val_loss']:.4f}",
            f"- 训练准确率：{percent(first['train_acc'])} -> "
            f"{percent(final['train_acc'])}",
            f"- 验证准确率：{percent(first['val_acc'])} -> "
            f"{percent(final['val_acc'])}",
            "",
            "五、学习率与数据风险说明",
            "CosineAnnealingLR 会按余弦曲线降低学习率。旧版 eta_min=0 "
            "会导致最后一次记录为 0，这是调度器行为，不是数据错误。",
            f"当前版本 eta_min={args.min_lr:g}，history.json 中 lr 是本轮学习率，"
            "next_lr 是 scheduler.step() 后的下一轮学习率。",
            "当前数据集已经检查完全相同文件的跨 split 重复；但 Fruits-360 "
            "可能存在同一拍摄序列的近似图片跨 split，因此测试指标可能偏乐观。",
            "建议增加真实手机拍摄的外部测试集，或按拍摄序列/水果实例进行分组切分。",
        ]
    )

    if metrics:
        lines.extend(
            [
                "",
                "六、测试集评估",
                f"- Accuracy：{metrics['accuracy']:.2%}",
                f"- Macro Precision：{metrics['macro_precision']:.2%}",
                f"- Macro Recall：{metrics['macro_recall']:.2%}",
                f"- Macro F1：{metrics['macro_f1']:.2%}",
            ]
        )

    lines.extend(
        [
            "",
            "七、后续建议",
            "1. 使用真实手机拍摄图像进行外部测试。",
            "2. 对混淆矩阵和错误样本进行定性分析。",
            "3. 比较从零训练与 ImageNet 预训练在外部数据上的表现。",
            "4. 使用多个随机种子重复实验并报告均值和标准差。",
        ]
    )
    (output_dir / "analysis.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(args):
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0")
    if args.lr <= 0:
        raise ValueError("--lr must be greater than 0")
    if args.min_lr < 0 or args.min_lr >= args.lr:
        raise ValueError("--min-lr must be >= 0 and smaller than --lr")

    seed_everything(args.seed)
    data_dir = project_path(args.data)
    output_dir = create_run_dir(project_path(args.output))
    print("Output directory:", output_dir)

    requested_device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({requested_device}), but CUDA is not available."
        )
    device = torch.device(requested_device)
    print("Device:", device)

    train_ds, val_ds, test_ds = build_datasets(data_dir, args.image_size)
    print("Classes:", train_ds.classes)
    print("Train:", len(train_ds))
    print("Val:", len(val_ds))
    print("Test:", len(test_ds))

    (output_dir / "classes.json").write_text(
        json.dumps(train_ds.classes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        len(train_ds.classes),
        pretrained=not args.no_pretrained,
        model_name=args.model,
        freeze_backbone=args.freeze_backbone,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    best_val_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "next_lr": next_lr,
        }
        history.append(row)
        print(
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"lr={current_lr:.8f}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "classes": train_ds.classes,
            "image_size": args.image_size,
            "model_name": args.model,
            "pretrained": not args.no_pretrained,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, output_dir / "best.pt")
            print("Saved best model.")

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        plot_history(history, output_dir)
    except Exception as error:
        (output_dir / "training_curves_error.txt").write_text(
            f"{type(error).__name__}: {error}\n",
            encoding="utf-8",
        )
    write_analysis_report(
        output_dir,
        args,
        train_ds.classes,
        len(train_ds),
        len(val_ds),
        len(test_ds),
        history,
        best_val_acc,
    )

    print("\nBest validation accuracy:", best_val_acc)
    print("Best checkpoint:", output_dir / "best.pt")
    print("Analysis report:", output_dir / "analysis.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/split")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    main(parser.parse_args())
