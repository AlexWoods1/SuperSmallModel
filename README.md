# SuperSmallModel (STLM)

A **~1M-parameter GPT** built from scratch in pure PyTorch — byte-level BPE, training loop, and a CLI so anyone can **download & run** or **train on a laptop**.

Portfolio project: end-to-end language modeling without Hugging Face model wrappers.

## Results (toy CPU run)


|          |                                            |
| -------- | ------------------------------------------ |
| Params   | **957,184**                                |
| Config   | 4 layers × 128 embd, block 256, vocab 1024 |
| Data     | Tiny Shakespeare                           |
| Val loss | **~4.04** after 5k steps (from ~5.3 @ 250) |


Example (`stlm generate --prompt "MENENIUS:"`):

```text
MENENIUS: you must peak there's presen's ent.

LEONTES:
We have gentleman on
pised for anishonour will not befaith appy and sul, thee,
...
SICINIUS:
Wish swit,
Do honour: forthere not duke of your highs, afts
```

It's not 10 trillion parameter claude mythos, It's under a million parameters. It's a demo. GPU training would see better performance.

## Quickstart

**Requires:** Python 3.14+, [uv](https://docs.astral.sh/uv/)

```powershell
git clone https://github.com/AlexWoods1/SuperSmallModel.git
cd SuperSmallModel
uv sync
```



### Generate (pretrained checkpoint)

train locally or download a [release](https://github.com/AlexWoods1/SuperSmallModel/releases):

```powershell
uv run stlm generate `
  --checkpoint checkpoints/toy_cpu/ckpt_5000.pt `
  --prompt "MENENIUS:" `
  --max-tokens 200 `
  --temperature 0.8 `
  --top-k 40
```



### Train yourself

Smoke (minutes):

```powershell
uv run python scripts/prepare_data.py --dataset shakespeare --out data/smoke
# tokenizer + bins are created on first train if needed
uv run stlm train --config configs/smoke.yaml
```

Full toy run (hours on CPU):

```powershell
uv run stlm train --config configs/toy_cpu.yaml
```



## What’s inside


| Piece                          | Path                    |
| ------------------------------ | ----------------------- |
| GPT (manual causal attention)  | `src/stlm/model.py`     |
| Byte-level BPE                 | `src/stlm/tokenizer.py` |
| Data + `.bin` cache            | `src/stlm/data.py`      |
| Train (AdamW, warmup + cosine) | `src/stlm/train.py`     |
| Generate                       | `src/stlm/generate.py`  |
| CLI                            | `uv run stlm …`         |


Docs: [architecture](docs/architecture.md) · [training](docs/training.md) · [inference](docs/inference.md)

## Tests

```powershell
uv run pytest -q
```



## License

MIT — see [LICENSE](LICENSE).

---

Built as an open, handcrafted mini-LLM for learning and hiring portfolios. Feedback welcome via GitHub. Some of the tests and docs were written by cursor. 