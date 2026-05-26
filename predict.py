"""Run inference with a trained captcha model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from charset import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES
from dataset import decode_indices
from model import CRNN


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("L")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).unsqueeze(0)


def load_model(checkpoint_path: Path, device: torch.device) -> CRNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    model = CRNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_text(model: CRNN, image_path: Path, device: torch.device) -> str:
    tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        log_probs = model(tensor)
    sequence = log_probs.argmax(dim=2).squeeze(1).tolist()
    return decode_indices(sequence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict captcha text")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--image", type=Path, default=None, help="Single image path")
    parser.add_argument("--folder", type=Path, default=None, help="Folder of images")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.image is None and args.folder is None:
        raise ValueError("Provide --image or --folder")

    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)

    paths: list[Path]
    if args.image is not None:
        paths = [args.image]
    else:
        paths = sorted(args.folder.glob("*.png"))

    for path in paths:
        text = predict_text(model, path, device)
        print(f"{path.name}: {text}")


if __name__ == "__main__":
    main()
