"""Tests for text sampling / generation."""

from __future__ import annotations

from pathlib import Path

import torch

from stlm.config import Config, DataConfig, ModelConfig, PathsConfig, TrainConfig
from stlm.data import TextDataset
from stlm.generate import generate, load_checkpoint
from stlm.model import GPT
from stlm.tokenizer import Tokenizer
from stlm.train import sample_text, save_checkpoint


def _build_dataset(tmp_path: Path) -> TextDataset:
    lines = [f"once upon a time in land {i} there lived a fox\n" for i in range(80)]
    text = "".join(lines)
    cut = int(len(text) * 0.8)
    train_text, val_text = text[:cut], text[cut:]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (data_dir / "val.txt").write_text(val_text, encoding="utf-8")

    tok_path = data_dir / "tokenizer.json"
    tok = Tokenizer()
    tok.train(train_text, vocab_size=288)
    tok.save(tok_path)
    return TextDataset(data_dir, tok_path, device="cpu")


def _tiny_model(vocab_size: int, *, block_size: int = 16) -> GPT:
    cfg = ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=16,
        block_size=block_size,
        vocab_size=vocab_size,
        dropout=0.0,
        bias=True,
    )
    return GPT(cfg)


def _write_checkpoint(tmp_path: Path, data: TextDataset) -> Path:
    """Save a minimal training checkpoint compatible with load_checkpoint."""
    model = _tiny_model(data.tokenizer.vocab_size)
    cfg = Config(
        model=model.config,
        train=TrainConfig(
            batch_size=1,
            max_steps=1,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.99,
            warmup_steps=0,
            grad_clip=1.0,
            eval_interval=1,
            eval_batches=1,
            sample_interval=1,
            checkpoint_interval=1,
            seed=0,
        ),
        data=DataConfig(
            dataset="synthetic",
            data_dir=tmp_path / "data",
            train_frac=0.9,
        ),
        paths=PathsConfig(
            out_dir=tmp_path / "checkpoints",
            tokenizer_path=tmp_path / "data" / "tokenizer.json",
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    path = tmp_path / "checkpoints" / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=1, cfg=cfg)
    return path


def test_sample_text_returns_nonempty_string(tmp_path: Path) -> None:
    torch.manual_seed(0)
    data = _build_dataset(tmp_path)
    model = _tiny_model(data.tokenizer.vocab_size)
    out = sample_text(model, data, max_new_tokens=8, temperature=1.0)
    assert isinstance(out, str)
    assert len(out) > 0


def test_sample_text_grows_with_max_new_tokens(tmp_path: Path) -> None:
    """sample_text appends exactly max_new_tokens ids before decoding."""
    torch.manual_seed(0)
    data = _build_dataset(tmp_path)
    model = _tiny_model(data.tokenizer.vocab_size)

    lengths: list[int] = []
    original_decode = data.tokenizer.decode

    def counting_decode(ids: list[int]) -> str:
        lengths.append(len(ids))
        return original_decode(ids)

    data.tokenizer.decode = counting_decode  # type: ignore[method-assign]

    prompt_len = len(data.tokenizer.encode("\n"))
    max_new_tokens = 11
    sample_text(model, data, max_new_tokens=max_new_tokens, temperature=1.0)

    assert lengths == [prompt_len + max_new_tokens]


def test_load_checkpoint_restores_model_and_tokenizer(tmp_path: Path) -> None:
    data = _build_dataset(tmp_path)
    ckpt_path = _write_checkpoint(tmp_path, data)
    model, tok = load_checkpoint(ckpt_path, device="cpu")

    assert isinstance(model, GPT)
    assert isinstance(tok, Tokenizer)
    assert tok.vocab_size == data.tokenizer.vocab_size
    assert model.config.vocab_size == data.tokenizer.vocab_size
    assert next(model.parameters()).device.type == "cpu"


def test_generate_returns_nonempty_string(tmp_path: Path) -> None:
    torch.manual_seed(0)
    data = _build_dataset(tmp_path)
    model = _tiny_model(data.tokenizer.vocab_size)
    out = generate(
        model,
        data.tokenizer,
        "once upon",
        max_tokens=8,
        temperature=1.0,
    )
    assert isinstance(out, str)
    assert len(out) > 0


def test_generate_grows_with_max_tokens(tmp_path: Path) -> None:
    """generate appends exactly max_tokens ids before decoding."""
    torch.manual_seed(0)
    data = _build_dataset(tmp_path)
    model = _tiny_model(data.tokenizer.vocab_size)
    prompt = "once upon"

    lengths: list[int] = []
    original_decode = data.tokenizer.decode

    def counting_decode(ids: list[int]) -> str:
        lengths.append(len(ids))
        return original_decode(ids)

    data.tokenizer.decode = counting_decode  # type: ignore[method-assign]

    prompt_len = len(data.tokenizer.encode(prompt))
    max_tokens = 9
    generate(model, data.tokenizer, prompt, max_tokens=max_tokens, temperature=1.0)

    assert lengths == [prompt_len + max_tokens]


def test_generate_respects_top_k(tmp_path: Path) -> None:
    torch.manual_seed(0)
    data = _build_dataset(tmp_path)
    model = _tiny_model(data.tokenizer.vocab_size)
    out = generate(
        model,
        data.tokenizer,
        "fox",
        max_tokens=5,
        temperature=1.0,
        top_k=5,
    )
    assert isinstance(out, str)
    assert len(out) > 0


def test_generate_from_loaded_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    data = _build_dataset(tmp_path)
    ckpt_path = _write_checkpoint(tmp_path, data)
    model, tok = load_checkpoint(ckpt_path, device="cpu")
    out = generate(model, tok, "once", max_tokens=4, temperature=0.8, top_k=10)
    assert isinstance(out, str)
    assert len(out) > 0
