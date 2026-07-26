# Training

How to prepare data, train STLM, and what to expect on a laptop CPU.

## Prerequisites

- Python **3.14+**
- [uv](https://docs.astral.sh/uv/)
- From repo root: `uv sync`

Optional GPU/MPS is auto-detected; CPU is the documented path.

## 1. Prepare text

```powershell
uv run python scripts/prepare_data.py --dataset shakespeare --out data/smoke
```

Writes `data/smoke/train.txt` and `val.txt` (~1.1M characters total for Tiny Shakespeare).

TinyStories (streamed subset):

```powershell
uv run python scripts/prepare_data.py --dataset tinystories --out data/tinystories --max-chars 20000000
```

## 2. Tokenizer

Train a byte-level BPE and save JSON (example: vocab 1024):

```powershell
uv run python -c "from pathlib import Path; from stlm.tokenizer import Tokenizer; t=Tokenizer(); t.train(Path('data/smoke/train.txt').read_text(encoding='utf-8'), 1024); t.save(Path('data/smoke/tokenizer.json')); print(t.vocab_size)"
```

Point `paths.tokenizer_path` in your YAML at that file. Training **overrides** `model.vocab_size` from the tokenizer.

## 3. Train

Smoke (fast sanity check):

```powershell
uv run stlm train --config configs/smoke.yaml
```

Toy CPU (release-quality for this repo):

```powershell
uv run stlm train --config configs/toy_cpu.yaml
```

First run encodes text and writes `train.bin` / `val.bin` next to the `.txt` files. Later runs reload the bins if they are newer than the text and tokenizer.

### What the loop does

- Device: CUDA → MPS → CPU
- AdamW + grad clip
- Linear **warmup**, then **cosine** decay to 10% of peak LR
- Periodic train/val loss, sample printouts, checkpoints under `paths.out_dir`

Checkpoint payload: `model`, `optimizer`, `step`, full `config` dict, `tokenizer_path`.

## Configs


|                 | Smoke               | Toy CPU                   |
| --------------- | ------------------- | ------------------------- |
| Layers / embd   | 2 / 64              | 4 / 128                   |
| Context         | 64                  | 256                       |
| Steps           | ~200                | 5000                      |
| Batch           | 16                  | 16                        |
| Out dir         | `checkpoints/smoke` | `checkpoints/toy_cpu`     |
| Wall time (CPU) | minutes             | on the order of **hours** |


Keep `n_embd % n_head == 0`.

## Healthy loss (Shakespeare, vocab 1024)

Rough guide from the bundled toy run:


| Step  | Val loss (approx.) |
| ----- | ------------------ |
| ~250  | ~5.3               |
| ~2000 | ~4.3               |
| ~5000 | **~4.04**          |


Random-guess CE for vocab 1024 is ln(1024)  = **~**6.9. If loss never leaves ~6–7, check tokenizer path, vocab override, and that bins match the current tokenizer (delete `*.bin` and retrain if you changed merges).

## Troubleshooting


| Symptom                             | Likely cause                                    |
| ----------------------------------- | ----------------------------------------------- |
| `Config file not found`             | Run from repo root                              |
| OOM                                 | Lower `batch_size` or `block_size`              |
| Gibberish forever                   | Too few steps / too small model — try `toy_cpu` |
| `Token id exceeds uint16`           | Vocab > 65535 (not used in v1)                  |
| Stale tokens after tokenizer change | Delete `data/**/*.bin` and train again          |


## Tests

```powershell
uv run pytest -q
```

Unit tests cover model shapes, tokenizer round-trips, short train steps, and generate — they do not replace a full `toy_cpu` run.