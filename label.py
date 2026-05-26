"""Manual captcha labeling tool."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox


def load_labels(labels_path: Path) -> dict[str, str]:
    if not labels_path.exists():
        return {}

    labels: dict[str, str] = {}
    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get("filename", "").strip()
            label = row.get("label", "").strip().upper()
            if filename and label:
                labels[filename] = label
    return labels


def save_labels(labels_path: Path, labels: dict[str, str]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label"])
        writer.writeheader()
        for filename in sorted(labels):
            writer.writerow({"filename": filename, "label": labels[filename]})


def get_unlabeled_images(images_dir: Path, labels: dict[str, str]) -> list[Path]:
    images = sorted(images_dir.glob("*.png"))
    return [path for path in images if path.name not in labels]


class LabelApp:
    def __init__(
        self,
        images: list[Path],
        labels: dict[str, str],
        labels_path: Path,
        start_index: int = 0,
    ) -> None:
        self.images = images
        self.labels = labels
        self.labels_path = labels_path
        self.index = start_index
        self.saved_count = 0

        self.root = tk.Tk()
        self.root.title("Captcha Labeler")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("640x320")

        self.progress_var = tk.StringVar()
        self.input_var = tk.StringVar()
        self.input_var.trace_add("write", self._normalize_input)

        progress = tk.Label(
            self.root,
            textvariable=self.progress_var,
            fg="#cccccc",
            bg="#1e1e1e",
            font=("Segoe UI", 11),
        )
        progress.pack(pady=(12, 6))

        self.image_label = tk.Label(self.root, bg="#1e1e1e")
        self.image_label.pack(pady=8, expand=True)

        entry = tk.Entry(
            self.root,
            textvariable=self.input_var,
            font=("Consolas", 24),
            justify="center",
            width=12,
        )
        entry.pack(pady=8)
        entry.focus_set()

        help_text = tk.Label(
            self.root,
            text="Enter label and press Enter to save | Backspace to skip | Esc to quit",
            fg="#888888",
            bg="#1e1e1e",
            font=("Segoe UI", 10),
        )
        help_text.pack(pady=(0, 12))

        self.root.bind("<Return>", self._on_submit)
        self.root.bind("<BackSpace>", self._on_skip)
        self.root.bind("<Escape>", self._on_quit)

        self._show_current()

    def _normalize_input(self, *_args) -> None:
        value = self.input_var.get().upper()
        filtered = "".join(char for char in value if char.isalnum())
        if filtered != value:
            self.input_var.set(filtered)

    def _show_current(self) -> None:
        if self.index >= len(self.images):
            messagebox.showinfo("Done", f"Labeled {self.saved_count} images this session.")
            self.root.destroy()
            return

        image_path = self.images[self.index]
        image = Image.open(image_path).convert("RGB")
        scale = 3
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
        photo = ImageTk.PhotoImage(image)

        self.image_label.configure(image=photo)
        self.image_label.image = photo
        self.progress_var.set(
            f"{self.index + 1}/{len(self.images)}  |  {image_path.name}  |  total saved: {len(self.labels)}"
        )
        self.input_var.set("")

    def _persist(self) -> None:
        save_labels(self.labels_path, self.labels)

    def _on_submit(self, _event=None) -> None:
        label = self.input_var.get().strip().upper()
        if not label:
            return

        image_path = self.images[self.index]
        self.labels[image_path.name] = label
        self._persist()
        self.saved_count += 1
        self.index += 1
        self._show_current()

    def _on_skip(self, _event=None) -> None:
        self.index += 1
        self._show_current()

    def _on_quit(self, _event=None) -> None:
        self._persist()
        self.root.destroy()

    def run(self) -> None:
        if not self.images:
            print("No unlabeled images found.")
            return
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label captcha images")
    parser.add_argument("--images", type=Path, default=Path("Images"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--limit", type=int, default=None, help="Max images to label this session")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.images.exists():
        print(f"Images directory not found: {args.images}", file=sys.stderr)
        return 1

    labels = load_labels(args.labels)
    unlabeled = get_unlabeled_images(args.images, labels)
    if args.limit is not None:
        unlabeled = unlabeled[: args.limit]

    print(f"Already labeled: {len(labels)}")
    print(f"Remaining this session: {len(unlabeled)}")

    app = LabelApp(unlabeled, labels, args.labels)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
