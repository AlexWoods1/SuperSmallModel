"""Tests for the GPT model and causal attention."""

from __future__ import annotations

import pytest
import torch

from stlm.config import ModelConfig
from stlm.model import CausalSelfAttention, GPT


def _tiny_config(**overrides: object) -> ModelConfig:
    cfg = {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 16,
        "block_size": 8,
        "vocab_size": 64,
        "dropout": 0.0,
        "bias": True,
    }
    cfg.update(overrides)
    return ModelConfig(**cfg)  # type: ignore[arg-type]


def _expected_param_count(cfg: ModelConfig) -> int:
    """Closed-form count with tied lm_head / wte and Linear+LayerNorm biases."""
    n_embd = cfg.n_embd
    emb = cfg.vocab_size * n_embd + cfg.block_size * n_embd
    # * c_attn (3x) + c_proj, each with bias when cfg.bias is True.
    bias_scale = 1 if cfg.bias else 0
    attn = (
        n_embd * (3 * n_embd)
        + bias_scale * (3 * n_embd)
        + n_embd * n_embd
        + bias_scale * n_embd
    )
    mlp = (
        n_embd * (4 * n_embd)
        + bias_scale * (4 * n_embd)
        + (4 * n_embd) * n_embd
        + bias_scale * n_embd
    )
    # * LayerNorm always has weight + bias in this model.
    ln = 2 * n_embd
    block = attn + mlp + 2 * ln
    return emb + cfg.n_layer * block + ln


def test_param_count_matches_formula() -> None:
    cfg = _tiny_config()
    model = GPT(cfg)
    assert model.count_parameters() == _expected_param_count(cfg)


def test_param_count_without_bias() -> None:
    cfg = _tiny_config(bias=False)
    model = GPT(cfg)
    assert model.count_parameters() == _expected_param_count(cfg)


def test_causal_mask_is_lower_triangular() -> None:
    cfg = _tiny_config(block_size=5)
    attn = CausalSelfAttention(cfg)
    mask = attn.causal_mask[0, 0]
    assert torch.equal(mask, torch.tril(torch.ones_like(mask)))


def test_past_logits_independent_of_future_tokens() -> None:
    """Changing the last token must not change logits at earlier positions."""
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = GPT(cfg)
    model.eval()

    x1 = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size

    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    assert torch.allclose(logits1[:, :-1], logits2[:, :-1], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(logits1[:, -1], logits2[:, -1], atol=1e-5, rtol=1e-5)


def test_forward_shapes_and_loss() -> None:
    cfg = _tiny_config()
    model = GPT(cfg)
    b, t = 2, 5
    idx = torch.randint(0, cfg.vocab_size, (b, t))
    targets = torch.randint(0, cfg.vocab_size, (b, t))

    logits, loss = model(idx, targets)
    assert logits.shape == (b, t, cfg.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0


def test_forward_without_targets_returns_no_loss() -> None:
    cfg = _tiny_config()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 3))
    logits, loss = model(idx)
    assert logits.shape == (1, 3, cfg.vocab_size)
    assert loss is None


def test_forward_rejects_sequence_longer_than_block_size() -> None:
    cfg = _tiny_config(block_size=4)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    with pytest.raises(ValueError, match="block size"):
        model(idx)
