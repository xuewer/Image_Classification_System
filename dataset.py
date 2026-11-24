from pathlib import Path

from torchvision import datasets, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size=224):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_tfms, eval_tfms


def _check_split_directory(data_dir, split):
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset split directory: {split_dir}")
    return split_dir


def build_datasets(data_dir, image_size=224):
    data_dir = Path(data_dir)
    train_tfms, eval_tfms = build_transforms(image_size)

    train_dir = _check_split_directory(data_dir, "train")
    val_dir = _check_split_directory(data_dir, "val")
    test_dir = _check_split_directory(data_dir, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

    if not train_ds.samples:
        raise ValueError(f"No training images found in {train_dir}")
    if not val_ds.samples:
        raise ValueError(f"No validation images found in {val_dir}")
    if not test_ds.samples:
        raise ValueError(f"No test images found in {test_dir}")

    expected_classes = train_ds.classes
    for name, dataset in (("val", val_ds), ("test", test_ds)):
        if dataset.classes != expected_classes:
            raise ValueError(
                f"Class mismatch in {name}: "
                f"expected {expected_classes}, got {dataset.classes}"
            )

    return train_ds, val_ds, test_ds


def summarize_datasets(train_ds, val_ds, test_ds):
    return {
        "classes": train_ds.classes,
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
        "total": len(train_ds) + len(val_ds) + len(test_ds),
    }
