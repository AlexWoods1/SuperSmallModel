"""Tests for training helpers and a short synthetic training run."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from stlm.config import Config, ModelConfig, TrainConfig, load_config
from stlm.data import TextDataset
from stlm.model import GPT
from stlm.tokenizer import Tokenizer
from stlm.train import get_lr, save_checkpoint, set_seed, train


def _write_tiny_corpus(data_dir: Path, tokenizer_path: Path) -> Tokenizer:
    # * Varied lines so BPE cannot collapse an entire split into one token.
    lines = [
        f"story {i}: the quick brown fox jumps over lazy dog {i}\n" for i in range(100)
    ]
    text = "".join(lines)
    cut = int(len(text) * 0.8)
    train_text, val_text = text[:cut], text[cut:]
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (data_dir / "val.txt").write_text(val_text, encoding="utf-8")

    tok = Tokenizer()
    tok.train(train_text, vocab_size=288)
    tok.save(tokenizer_path)
    return tok


def _tiny_train_yaml(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    tok_path = data_dir / "tokenizer.json"
    out_dir = tmp_path / "checkpoints"
    _write_tiny_corpus(data_dir, tok_path)

    cfg = {
        "model": {
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
            "block_size": 8,
            "vocab_size": 288,
            "dropout": 0.0,
            "bias": True,
        },
        "train": {
            "batch_size": 2,
            "max_steps": 2,
            "learning_rate": 3.0e-4,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.99,
            "warmup_steps": 1,
            "grad_clip": 1.0,
            "eval_interval": 2,
            "eval_batches": 1,
            "sample_interval": 100,
            "checkpoint_interval": 2,
            "seed": 0,
        },
        "data": {
            "dataset": "synthetic",
            "data_dir": str(data_dir),
            "train_frac": 0.9,
        },
        "paths": {
            "out_dir": str(out_dir),
            "tokenizer_path": str(tok_path),
        },
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


def test_load_config_smoke_yaml() -> None:
    cfg = load_config(Path("configs/smoke.yaml"))
    assert isinstance(cfg, Config)
    assert cfg.model.n_layer == 2
    assert cfg.train.max_steps == 200
    assert cfg.data.dataset == "shakespeare"


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_model_config_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            n_layer=1,
            n_head=3,
            n_embd=16,
            block_size=8,
            vocab_size=32,
        )


def test_get_lr_warmup_and_floor() -> None:
    cfg = TrainConfig(
        batch_size=1,
        max_steps=10,
        learning_rate=1.0,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.99,
        warmup_steps=2,
        grad_clip=1.0,
        eval_interval=1,
        eval_batches=1,
        sample_interval=1,
        checkpoint_interval=1,
        seed=0,
    )
    assert get_lr(0, cfg) == 0.0
    assert get_lr(1, cfg) == pytest.approx(0.5)
    assert get_lr(2, cfg) == pytest.approx(1.0)
    assert get_lr(10, cfg) == pytest.approx(0.1)
    # * Mid-decay is between peak and floor.
    mid = get_lr(6, cfg)
    assert 0.1 < mid < 1.0


def test_save_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(_tiny_train_yaml(tmp_path))
    model = GPT(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=3, cfg=cfg)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["step"] == 3
    assert "model" in blob
    assert "optimizer" in blob
    restored = GPT(cfg.model)
    restored.load_state_dict(blob["model"])


def test_train_two_steps_on_synthetic_data(tmp_path: Path) -> None:
    config_path = _tiny_train_yaml(tmp_path)
    set_seed(0)
    train(config_path)
    ckpt = tmp_path / "checkpoints" / "ckpt_2.pt"
    assert ckpt.is_file()


def test_text_dataset_batch_shapes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    tok_path = data_dir / "tokenizer.json"
    _write_tiny_corpus(data_dir, tok_path)
    ds = TextDataset(data_dir, tok_path, device="cpu")
    x, y = ds.get_batch("train", batch_size=4, block_size=8)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_text_dataset_rejects_short_split(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    tok_path = data_dir / "tokenizer.json"
    data_dir.mkdir(parents=True)
    (data_dir / "train.txt").write_text("hi", encoding="utf-8")
    (data_dir / "val.txt").write_text("bye", encoding="utf-8")
    tok = Tokenizer()
    tok.save(tok_path)
    ds = TextDataset(data_dir, tok_path, device="cpu")
    with pytest.raises(ValueError, match="too short"):
        ds.get_batch("train", batch_size=1, block_size=64)
