"""Evaluate a trained CRNN captcha model checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from charset import BLANK_INDEX, NUM_CLASSES
from dataset import create_dataloaders, decode_indices
from model import CRNN


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cpu")


def greedy_decode(log_probs: torch.Tensor) -> list[str]:
    predictions = log_probs.argmax(dim=2).transpose(0, 1)
    decoded: list[str] = []
    for sequence in predictions:
        decoded.append(decode_indices(sequence.tolist()))
    return decoded


@torch.no_grad()
def evaluate_split(
    model: CRNN,
    loader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> tuple[float, float, float, list[dict]]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    exact = 0
    correct_chars = 0
    total_chars = 0
    total = 0
    sample_offset = 0
    
    failures = []

    for images, targets, target_lengths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        log_probs = model(images)
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=device,
        )
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        total_loss += loss.item()
        total_batches += 1

        batch_size = images.size(0)
        batch_samples = loader.dataset.samples[sample_offset : sample_offset + batch_size]
        sample_offset += batch_size
        predictions = greedy_decode(log_probs.float().cpu())

        for (filename, label), prediction in zip(batch_samples, predictions):
            total += 1
            is_exact = (prediction == label)
            if is_exact:
                exact += 1
            else:
                failures.append({
                    "filename": filename,
                    "label": label,
                    "prediction": prediction
                })
            
            for pred_char, true_char in zip(prediction, label):
                if pred_char == true_char:
                    correct_chars += 1
            total_chars += max(len(label), 1)

    avg_loss = total_loss / max(total_batches, 1)
    seq_acc = exact / total if total else 0.0
    char_acc = correct_chars / total_chars if total_chars else 0.0
    return avg_loss, seq_acc, char_acc, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate captcha CRNN checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--images", type=Path, default=Path("Images"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = resolve_device(args.device)
    
    # Load samples and split manually to create non-shuffled, non-augmented dataloaders for evaluation
    from dataset import load_label_rows, split_samples, CaptchaDataset, collate_fn
    from torch.utils.data import DataLoader

    samples = load_label_rows(args.labels)
    if not samples:
        raise ValueError(f"No labeled samples found in {args.labels}")

    train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio)
    pin_memory = torch.cuda.is_available()

    eval_train_loader = DataLoader(
        CaptchaDataset(train_samples, args.images, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        CaptchaDataset(val_samples, args.images, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    print("\nEvaluating training split...")
    train_loss, train_seq, train_char, train_fails = evaluate_split(model, eval_train_loader, criterion, device)
    
    print("Evaluating validation split...")
    val_loss, val_seq, val_char, val_fails = evaluate_split(model, val_loader, criterion, device)

    # Results Table
    print("\n" + "="*60)
    print(f"{'Dataset Split':<20} | {'Loss':<10} | {'Seq Acc (Exact)':<16} | {'Char Acc':<10}")
    print("-"*60)
    print(f"{'Train':<20} | {train_loss:<10.4f} | {train_seq:<16.2%} | {train_char:<10.2%}")
    print(f"{'Validation':<20} | {val_loss:<10.4f} | {val_seq:<16.2%} | {val_char:<10.2%}")
    print("="*60)
    
    # Print some incorrect prediction examples
    if val_fails:
        print(f"\nSample of incorrect predictions in validation set ({len(val_fails)} total failures):")
        for i, fail in enumerate(val_fails[:10]):
            print(f"  Image: {fail['filename']} | True: {fail['label']:<8} | Predicted: {fail['prediction']:<8}")


if __name__ == "__main__":
    main()
