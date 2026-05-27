"""Predict and check a small random subset of captcha images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from charset import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES
from dataset import decode_indices, load_label_rows
from model import CRNN
from predict import load_model, preprocess_image, predict_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify model predictions on a random subset of images")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--images", type=Path, default=Path("Images"))
    parser.add_argument("--num-samples", type=int, default=15, help="Number of random samples to check (e.g. 10-20)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Labels CSV not found: {args.labels}")
    if not args.images.exists():
        raise FileNotFoundError(f"Images folder not found: {args.images}")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model = load_model(args.checkpoint, device)

    # Load all label rows
    samples = load_label_rows(args.labels)
    if not samples:
        print("No samples found in labels file.")
        return

    # Choose random subset
    num_to_sample = min(args.num_samples, len(samples))
    sampled = random.sample(samples, num_to_sample)

    print(f"\nChecking {num_to_sample} random images:")
    print("=" * 85)
    print(f"{'Filename':<45} | {'True Label':<12} | {'Predicted':<12} | {'Status':<8}")
    print("-" * 85)

    correct_count = 0
    for filename, true_label in sampled:
        image_path = args.images / filename
        if not image_path.exists():
            print(f"{filename:<45} | {true_label:<12} | {'[FILE MISSING]':<12} | ❌")
            continue

        pred_label = predict_text(model, image_path, device)
        is_match = (pred_label == true_label)
        status_str = "✅ Match" if is_match else "❌ Fail"
        if is_match:
            correct_count += 1

        print(f"{filename:<45} | {true_label:<12} | {pred_label:<12} | {status_str}")

    print("=" * 85)
    subset_acc = correct_count / num_to_sample if num_to_sample else 0.0
    print(f"Subset Accuracy: {correct_count}/{num_to_sample} ({subset_acc:.2%})")


if __name__ == "__main__":
    main()
