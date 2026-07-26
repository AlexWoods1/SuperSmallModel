"""Training loop for STLM."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from stlm.config import Config, TrainConfig, load_config
from stlm.data import TextDataset
from stlm.model import GPT


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to 10% of peak LR."""
    min_lr = 0.1 * cfg.learning_rate
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    if step >= cfg.max_steps:
        return min_lr
    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (cfg.learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data: TextDataset,
    cfg: Config,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.train.eval_batches)
        for i in range(cfg.train.eval_batches):
            xb, yb = data.get_batch(split, cfg.train.batch_size, cfg.model.block_size)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample_text(
    model: GPT,
    data: TextDataset,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
) -> str:
    """Generate a short continuation from a newline prompt."""
    model.eval()
    device = next(model.parameters()).device
    prompt = "\n"
    idx = torch.tensor([data.tokenizer.encode(prompt)], dtype=torch.long, device=device)
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return data.tokenizer.decode(idx[0].tolist())


def save_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": asdict(cfg),
            "tokenizer_path": str(cfg.paths.tokenizer_path),
        },
        path,
    )


def train(config_path: str | Path = "configs/smoke.yaml") -> None:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = load_config(path)

    device = get_device()
    set_seed(cfg.train.seed)
    print(f"Training on {device} for {cfg.train.max_steps} steps")

    data = TextDataset.build_dataloaders(cfg, device)
    cfg.model.vocab_size = data.tokenizer.vocab_size
    print(f"Vocab size: {cfg.model.vocab_size}")

    model = GPT(cfg.model).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.beta1, cfg.train.beta2),
        weight_decay=cfg.train.weight_decay,
    )

    cfg.paths.out_dir.mkdir(parents=True, exist_ok=True)
    model.train()

    for step in range(1, cfg.train.max_steps + 1):
        lr = get_lr(step, cfg.train)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb = data.get_batch("train", cfg.train.batch_size, cfg.model.block_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps:
            losses = estimate_loss(model, data, cfg)
            print(
                f"step {step:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | lr {lr:.2e}"
            )

        if cfg.train.sample_interval > 0 and step % cfg.train.sample_interval == 0:
            sample = sample_text(model, data)
            print("--- sample ---")
            # * Windows consoles may not support all UTF-8 replacement chars.
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            printable = (
                sample[:500]
                .encode(encoding, errors="replace")
                .decode(encoding, errors="replace")
            )
            print(printable)
            print("--------------")

        if step % cfg.train.checkpoint_interval == 0 or step == cfg.train.max_steps:
            ckpt_path = cfg.paths.out_dir / f"ckpt_{step}.pt"
            save_checkpoint(
                ckpt_path, model=model, optimizer=optimizer, step=step, cfg=cfg
            )
            print(f"Saved {ckpt_path}")

    print("Training complete")


if __name__ == "__main__":
    train()
