"""Train CRNN captcha model on GPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
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
def evaluate(
    model: CRNN,
    loader,
    criterion: nn.CTCLoss,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    exact = 0
    correct_chars = 0
    total_chars = 0
    total = 0
    sample_offset = 0

    for images, targets, target_lengths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        with autocast(device.type, enabled=use_amp):
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

        for (_, label), prediction in zip(batch_samples, predictions):
            total += 1
            if prediction == label:
                exact += 1
            for pred_char, true_char in zip(prediction, label):
                if pred_char == true_char:
                    correct_chars += 1
            total_chars += max(len(label), 1)

    avg_loss = total_loss / max(total_batches, 1)
    seq_acc = exact / total if total else 0.0
    char_acc = correct_chars / total_chars if total_chars else 0.0
    return avg_loss, seq_acc, char_acc


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    use_amp = device.type == "cuda"

    train_loader, val_loader = create_dataloaders(
        labels_path=args.labels,
        images_dir=args.images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
    )

    model = CRNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(device.type, enabled=use_amp)

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"

    best_seq_acc = 0.0
    print(f"Training on {device} with {len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for images, targets, target_lengths in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device.type, enabled=use_amp):
                log_probs = model(images)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=log_probs.size(0),
                    dtype=torch.long,
                    device=device,
                )
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, seq_acc, char_acc = evaluate(model, val_loader, criterion, device, use_amp)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | seq_acc={seq_acc:.3f} | char_acc={char_acc:.3f}"
        )

        if seq_acc >= best_seq_acc:
            best_seq_acc = seq_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "seq_acc": seq_acc,
                    "char_acc": char_acc,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"Saved checkpoint to {best_path} (seq_acc={seq_acc:.3f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train captcha CRNN on GPU")
    parser.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--images", type=Path, default=Path("Images"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
