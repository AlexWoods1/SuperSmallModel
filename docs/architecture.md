# Architecture

STLM is a **decoder-only GPT** for next-token prediction. The interesting bits are implemented by hand in PyTorch — not via `transformers` model classes or `nn.MultiheadAttention`.

## Stack


| Built here                                 | Used as a library                         |
| ------------------------------------------ | ----------------------------------------- |
| Causal self-attention (QKV, mask, softmax) | `torch.nn` Linear / Embedding / LayerNorm |
| Transformer block (pre-norm + residuals)   | AdamW, autodiff                           |
| Byte-level BPE train / encode / decode     | PyYAML for configs                        |
| Train + generate loops                     |                                           |


## Forward path

```text
token ids (B, T)
    → token emb + position emb
    → dropout
    → N × Block
         ln → CausalSelfAttention → residual
         ln → MLP (4× GELU)       → residual
    → final LayerNorm
    → LM head (weights tied with token emb)
    → logits (B, T, vocab_size)
```

If targets are provided, loss is token-level cross-entropy (ignore index `-1`).

## Causal self-attention

For each position t, the model may only attend to positions \le t (no future leakage).

1. Project x to fused QKV (`n_embd → 3·n_embd`), then split.
2. Reshape into `n_head` heads with `head_dim = n_embd / n_head`.
3. Scores: QK^\top / \sqrt{d}.
4. Apply a lower-triangular **causal mask** (registered buffer, not a learned parameter).
5. Softmax → dropout → weighted sum of V → merge heads → output projection.

Code: `[src/stlm/model.py](../src/stlm/model.py)` (`CausalSelfAttention`).

## Block (pre-norm)

```text
x ← x + Attn(LN(x))
x ← x + MLP(LN(x))
```

Pre-norm (LayerNorm before sublayers) is the modern default and trains more stably than post-norm for deep stacks. STLM is shallow (2–4 layers) but uses the same pattern as larger GPTs.

## Embeddings and weight tying

- **Token embedding** `wte`: `vocab_size × n_embd`
- **Position embedding** `wpe`: `block_size × n_embd` (learned, absolute positions `0…T-1`)
- **LM head** shares `wte.weight` (fewer parameters, common for small LMs)

Sequences longer than `block_size` are rejected at forward time; generation crops to the last `block_size` tokens.

## Tokenizer

Byte-level BPE (`[src/stlm/tokenizer.py](../src/stlm/tokenizer.py)`):

- Base vocab = 256 UTF-8 bytes.
- `train` repeatedly merges the most frequent adjacent pair until `vocab_size`.
- `encode` / `decode` replay those merges; state is saved as `tokenizer.json` (`merges` only).

Toy run uses **vocab 1024** (~768 merges on Shakespeare).

## Config knobs

See YAML under `configs/`. Important fields:


| Field               | Role                                        |
| ------------------- | ------------------------------------------- |
| `n_layer`           | Depth                                       |
| `n_embd` / `n_head` | Width (must divide evenly)                  |
| `block_size`        | Context length T                            |
| `vocab_size`        | Overridden at train time from the tokenizer |
| `dropout`           | Attention / residual / embedding dropout    |




## Reference sizes


| Config                 | Approx. params | Intent                           |
| ---------------------- | -------------- | -------------------------------- |
| `configs/smoke.yaml`   | ~100k–200k     | Pipeline check in minutes        |
| `configs/toy_cpu.yaml` | **~957k**      | Laptop train; release checkpoint |


