#!/usr/bin/env python3
"""Detect crystals in vapor-diffusion images using ResNet features + Logistic Regression.

This script expects the MARCO-style folder layout that ships with the workshop:
materials/session2/data/marco-protein-crystal-image-recognition/
    └── train/
        ├── Clear/
        ├── Crystals/
        ├── Other/
        └── Precipitate/

Usage:
    python session2_crystal_classifier.py --data-root materials/session2/data/marco-protein-crystal-image-recognition/train

The code extracts 512-dim embeddings from a frozen ResNet18 backbone and fits a
sklearn Logistic Regression classifier.  Results are printed to stdout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['Crystals', 'Other', 'Precipitate', 'Clear']


def parse_args():
    parser = argparse.ArgumentParser(description="Crystal detection via ResNet features")
    parser.add_argument(
        "--data-root",
        type=str,
        default="materials/session2/data/marco-protein-crystal-image-recognition",
        help="Path containing MARCO TFRecord shards or ImageFolder data",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images (useful for quick demos)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("session2_crystal_classifier.pt"),
        help="Where to save the trained LogisticRegression (pickle format)",
    )
    parser.add_argument(
        "--converted-dir",
        type=Path,
        default=None,
        help="Optional directory to store converted ImageFolder data",
    )
    return parser.parse_args()


def get_feature_extractor(device: torch.device) -> nn.Module:
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet.to(device)
    resnet.eval()
    return resnet


def convert_tfrecords_if_needed(data_root: Path, split: str, limit: int | None, converted_dir: Path | None) -> Path:
    """Convert MARCO TFRecord shards into ImageFolder layout if necessary."""
    split_dir = data_root / split
    if split_dir.exists() and any(split_dir.iterdir()):
        return split_dir

    shards = sorted(data_root.glob(f"{split}-*"))
    if not shards:
        raise FileNotFoundError(f"No ImageFolder folders or TFRecord shards found under {data_root}")

    dest = converted_dir or (data_root / f"converted_{split}")
    for cls in CLASS_NAMES:
        (dest / cls).mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(shards)} TFRecord shards into {dest} ...")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("tensorflow is required to decode MARCO TFRecord shards") from exc

    record_count = 0
    for shard in shards:
        for raw_example in tf.compat.v1.io.tf_record_iterator(str(shard)):
            example = tf.train.Example.FromString(raw_example)
            img_bytes = example.features.feature['image/encoded'].bytes_list.value[0]
            label = example.features.feature['image/class/label'].int64_list.value[0]
            class_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else f"class_{label}"
            out_path = dest / class_name / f"img_{record_count:07d}.jpg"
            with open(out_path, 'wb') as f:
                f.write(img_bytes)
            record_count += 1
            if limit and record_count >= limit:
                print(f"Reached conversion limit of {limit} images")
                return dest
    print(f"Converted {record_count} images")
    return dest


def main():
    args = parse_args()
    raw_root = Path(args.data_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {raw_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_root = convert_tfrecords_if_needed(raw_root, 'train', args.limit, args.converted_dir)
    dataset = ImageFolder(root=str(image_root), transform=transform)
    if args.limit and args.limit < len(dataset):
        subset_indices = np.random.choice(len(dataset), args.limit, replace=False)
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    classes = dataset.dataset.classes if isinstance(dataset, torch.utils.data.Subset) else dataset.classes
    print(f"Loaded {len(dataset)} images across classes: {classes}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count() or 4)
    feature_extractor = get_feature_extractor(device)

    print("Extracting ResNet18 features...")
    features, labels = [], []
    with torch.no_grad():
        for images, batch_labels in tqdm(loader):
            images = images.to(device)
            embeddings = feature_extractor(images)
            features.append(embeddings.cpu().numpy())
            labels.append(batch_labels.numpy())
    X = np.vstack(features)
    y = np.concatenate(labels)
    print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))

    import joblib

    joblib.dump({"model": clf, "classes": classes}, args.output)
    print(f"Saved classifier to {args.output}")

if __name__ == "__main__":
    main()
