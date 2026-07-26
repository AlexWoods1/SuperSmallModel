"""Download and write local train/val text files for STLM."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def download_shakespeare() -> str:
    with urllib.request.urlopen(SHAKESPEARE_URL) as resp:
        return resp.read().decode("utf-8")


def download_tinystories_streaming(max_chars: int, split: str = "train") -> str:
    # * Stream so we never materialize the full dataset in RAM.
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    chunks: list[str] = []
    n = 0
    for row in ds:
        text = row["text"]
        chunks.append(text)
        n += len(text) + 1  # +1 for the join newline
        if n >= max_chars:
            break
    return "\n".join(chunks)[:max_chars]


def split_train_val(text: str, train_frac: float) -> tuple[str, str]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    # * Split on a newline boundary near the cut so we don't bisect a story mid-line.
    cut = int(len(text) * train_frac)
    cut = text.rfind("\n", 0, cut)
    if cut <= 0:
        cut = int(len(text) * train_frac)
    return text[:cut], text[cut:]


def write_splits(out_dir: Path, train: str, val: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text(train, encoding="utf-8")
    (out_dir / "val.txt").write_text(val, encoding="utf-8")
    print(f"Wrote {out_dir / 'train.txt'} ({len(train):,} chars)")
    print(f"Wrote {out_dir / 'val.txt'} ({len(val):,} chars)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare STLM text datasets.")
    p.add_argument(
        "--dataset",
        choices=("shakespeare", "tinystories"),
        required=True,
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument(
        "--max-chars",
        type=int,
        default=20_000_000,
        help="Cap for tinystories (ignored for shakespeare)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "shakespeare":
        text = download_shakespeare()
    else:
        text = download_tinystories_streaming(args.max_chars)

    train, val = split_train_val(text, args.train_frac)
    write_splits(args.out, train, val)


if __name__ == "__main__":
    main()
