"""Dataset loading and batching for STLM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stlm.config import Config
from stlm.tokenizer import Tokenizer


def _bin_path(txt_path: Path) -> Path:
    return txt_path.with_suffix(".bin")


def _cache_is_fresh(txt_path: Path, bin_path: Path, tokenizer_path: Path) -> bool:
    if not bin_path.is_file():
        return False
    bin_mtime = bin_path.stat().st_mtime
    return (
        bin_mtime >= txt_path.stat().st_mtime
        and bin_mtime >= tokenizer_path.stat().st_mtime
    )


def _encode_and_cache(
    txt_path: Path,
    bin_path: Path,
    tokenizer: Tokenizer,
) -> torch.Tensor:
    """Encode text to token ids and write a uint16 .bin cache."""
    text = txt_path.read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    if not ids:
        raise ValueError(f"Encoded empty sequence from {txt_path}")
    if max(ids) > np.iinfo(np.uint16).max:
        raise ValueError(f"Token id {max(ids)} exceeds uint16; increase cache dtype")
    arr = np.array(ids, dtype=np.uint16)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(bin_path)
    print(f"Wrote {bin_path} ({len(arr):,} tokens)")
    return torch.from_numpy(arr.astype(np.int64))


def _load_or_encode(
    txt_path: Path,
    tokenizer_path: Path,
    tokenizer: Tokenizer,
) -> torch.Tensor:
    if not txt_path.is_file():
        raise FileNotFoundError(f"Missing text file: {txt_path}")
    bin_path = _bin_path(txt_path)
    if _cache_is_fresh(txt_path, bin_path, tokenizer_path):
        arr = np.fromfile(bin_path, dtype=np.uint16)
        print(f"Loaded {bin_path} ({len(arr):,} tokens)")
        return torch.from_numpy(arr.astype(np.int64))
    return _encode_and_cache(txt_path, bin_path, tokenizer)


class TextDataset:
    def __init__(
        self, data_dir: Path, tokenizer_path: Path, device: str = "cpu"
    ) -> None:
        data_dir = Path(data_dir)
        tokenizer_path = Path(tokenizer_path)
        tok = Tokenizer.load(tokenizer_path)
        self.tokenizer = tok
        self.device = device
        self.train = _load_or_encode(data_dir / "train.txt", tokenizer_path, tok)
        self.val = _load_or_encode(data_dir / "val.txt", tokenizer_path, tok)

    def get_batch(
        self, split: str, batch_size: int, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        # * Random starts; need block_size+1 tokens so y can shift by one.
        high = len(data) - block_size
        if high <= 0:
            raise ValueError(
                f"Split '{split}' too short ({len(data)} tokens) for "
                f"block_size={block_size}"
            )
        # * Vectorized gather: one (B, T+1) index tensor instead of Python stack loops.
        starts = torch.randint(high, (batch_size,))
        offsets = torch.arange(block_size + 1)
        seq = data[starts[:, None] + offsets[None, :]]
        x = seq[:, :-1].contiguous()
        y = seq[:, 1:].contiguous()
        return x.to(self.device), y.to(self.device)

    @staticmethod
    def build_dataloaders(cfg: Config, device: str = "cpu") -> TextDataset:
        return TextDataset(cfg.data.data_dir, cfg.paths.tokenizer_path, device)
