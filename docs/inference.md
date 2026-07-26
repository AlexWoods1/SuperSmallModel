# Inference

Generate text from a trained checkpoint with the `stlm generate` CLI (or `stlm.generate` in Python).

## Checkpoint format

Produced by training (`ckpt_{step}.pt`):


| Key              | Contents                                            |
| ---------------- | --------------------------------------------------- |
| `model`          | `state_dict` for `GPT`                              |
| `optimizer`      | AdamW state (not needed for generate)               |
| `step`           | Training step                                       |
| `config`         | Full config `asdict` (includes `model` hyperparams) |
| `tokenizer_path` | Path to `tokenizer.json` used at train time         |


`load_checkpoint` rebuilds `ModelConfig` from `config["model"]`, loads weights, and loads the tokenizer from `tokenizer_path` (resolved relative to your current working directory — run from the **repo root**).

## CLI

```powershell
uv run stlm generate `
  --checkpoint checkpoints/toy_cpu/ckpt_5000.pt `
  --prompt "MENENIUS:" `
  --max-tokens 200 `
  --temperature 0.8 `
  --top-k 40 `
  --device cpu
```


| Flag            | Default  | Meaning                                |
| --------------- | -------- | -------------------------------------- |
| `--checkpoint`  | required | Path to `.pt`                          |
| `--prompt`      | newline  | UTF-8 prompt string                    |
| `--max-tokens`  | 200      | New tokens to sample                   |
| `--temperature` | 0.8      | Softmax temperature (`→0` ≈ greedy)    |
| `--top-k`       | off      | Keep only top-k logits before sampling |
| `--device`      | `cpu`    | `cpu` / `cuda` / `mps`                 |


## Sampling algorithm

1. Encode the prompt with the checkpoint’s tokenizer.
2. Each step, crop context to the last `block_size` tokens.
3. Forward → take logits at the last position.
4. Divide by temperature; optional top-k filter (other logits → `-inf`).
5. Softmax → `multinomial` → append one token.
6. Decode the full id sequence (prompt + continuation).

Code: `[src/stlm/generate.py](../src/stlm/generate.py)`.

### Temperature and top-k

- **Lower temperature** (e.g. `0.5`): sharper, more repetitive, often “safer” diction.
- **Higher temperature** (e.g. `1.2`): more random; more broken words on a tiny model.
- **Top-k** (e.g. `40`): truncates the long tail of rare tokens; often improves readability slightly.

## Python API

```python
from stlm.generate import load_checkpoint, generate

model, tok = load_checkpoint("checkpoints/toy_cpu/ckpt_5000.pt", device="cpu")
print(generate(model, tok, "MENENIUS:", max_tokens=200, temperature=0.8, top_k=40))
```

## Quality expectations

The bundled **~957k-param** Shakespeare model produces recognizable dialogue structure (speaker names, line breaks) but frequent misspellings and nonsense words.

If output is pure noise or crashes:

1. Confirm the checkpoint path and that training finished (`ckpt_5000.pt`).
2. Ensure `tokenizer.json` still exists at the path stored in the checkpoint.
3. Match train/generate device dtype expectations (v1 is fp32).

## Releases

When distributing weights, ship **checkpoint +** `tokenizer.json` together (and keep relative paths consistent, or document that users should place the tokenizer where the checkpoint expects). See `scripts/export_release.py` once packaged for GitHub Releases.